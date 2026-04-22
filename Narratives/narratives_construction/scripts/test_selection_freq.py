import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import statsmodels.api as sm
import os

# Try to import linearmodels for robust Panel Data fixed effects
try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("Note: 'linearmodels' package not found. To get cleaner fixed-effects output, you can run: pip install linearmodels")
    print("Falling back to statsmodels within-transformation (demeaning) approach...\n")

# 1. Define Paths (Update these if your files are in different locations)
H5_PATH = Path(r'C:\Users\jonat\Lasso_paper\Results\Estimation\Cross_Sectional\betas.h5')
FREQ_CSV_PATH = Path(r'C:\Users\jonat\Lasso_paper\Narratives\narratives_construction\scripts\academic_topics.csv')

# ==========================================
# SANITY CHECK 1: Ensure files exist
# ==========================================
if not H5_PATH.exists():
    raise FileNotFoundError(f"HDF5 file not found at: {H5_PATH}")
if not FREQ_CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found at: {FREQ_CSV_PATH}")

# 2. Load LASSO selection frequencies from betas.h5
print("Loading betas from HDF5...")
with h5py.File(H5_PATH, 'r') as f:
    # SANITY CHECK 2: Validate internal HDF5 keys
    if 'betas' not in f or 'topics' not in f:
        raise KeyError("HDF5 file must contain 'betas' and 'topics' datasets.")
        
    T_betas = f['betas'][:]
    topics_arr = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in f['topics'][:]]
    
    # Try to extract stock IDs if they are saved in the H5 file
    if 'stocks' in f:
        stock_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['stocks'][:]]
    else:
        stock_ids = [f"Stock_{i}" for i in range(T_betas.shape[0])]

# SANITY CHECK 3: Shape matching
if T_betas.shape[1] != len(topics_arr):
    raise ValueError(f"Shape mismatch: betas array has {T_betas.shape[1]} topics but topics_arr has {len(topics_arr)}.")

# Calculate selection rate per topic PER STOCK (preserve stock variation)
sel = (T_betas != 0)
valid_dates = np.isfinite(T_betas).any(axis=(0,1))
sel_valid = sel[:, :, valid_dates]

# Mean over dates (axis=2) -> shape: (n_stocks, n_topics)
stock_topic_sel = sel_valid.mean(axis=2) * 100 

# Create a Panel DataFrame (Rows = Stock + Topic combination)
print("Building stock-topic panel dataset...")
panel_df = pd.DataFrame(stock_topic_sel, index=stock_ids, columns=topics_arr)
# Melt from wide to long format
panel_df = panel_df.reset_index().melt(id_vars='index', var_name='Topic_LASSO', value_name='Selection_Rate_Pct')
panel_df.rename(columns={'index': 'Stock'}, inplace=True)

# 3. Filter out Macro variables
MACRO_SET = {
    '10-Year_Breakeven_Inflation_Rate', '10-Year_Treasury_Yield',
    '3-Month_Treasury_Yield', 'Fed_Funds_Effective_Rate',
    'Financial_Stress_Index', 'High_Yield_Option-Adjusted_Spread',
    'Trade_Weighted_USD_Index', 'USD_to_JPY', 'VIX_Volatility_Index',
    'WTI_Crude_Oil'
}

panel_df = panel_df[~panel_df['Topic_LASSO'].isin(MACRO_SET)].copy()

# 4. Load the frequency table
print("Loading frequency table...")
freq_df = pd.read_csv(FREQ_CSV_PATH)

# SANITY CHECK 4: CSV Structure
if 'Topic' not in freq_df.columns or 'Frequency' not in freq_df.columns:
    raise ValueError("Frequency CSV must contain 'Topic' and 'Frequency' columns.")

# 5. Robust Merging
def normalize_string(s):
    return str(s).lower().replace(' ', '').replace('_', '').replace('/', '')

panel_df['norm_name'] = panel_df['Topic_LASSO'].apply(normalize_string)
freq_df['norm_name'] = freq_df['Topic'].apply(normalize_string)

# Merge datasets
merged_df = pd.merge(panel_df, freq_df, on='norm_name', how='left')

# Create predictors
merged_df['In_List'] = merged_df['Frequency'].notna().astype(int)
merged_df['Frequency_Count'] = merged_df['Frequency'].fillna(0)

# ==========================================
# VALIDITY CHECK: Print merge success rates
# ==========================================
unique_topics = merged_df['Topic_LASSO'].nunique()
matched_topics = merged_df[merged_df['In_List'] == 1]['Topic_LASSO'].nunique()

print("\n" + "-"*40)
print("MERGE VALIDITY CHECK:")
print(f"Total Unique Topics in LASSO output: {unique_topics}")
print(f"Topics successfully matched to CSV : {matched_topics}")
if matched_topics == 0:
    print("WARNING: Zero topics matched. Check your CSV contents and normalization.")
print("-"*40)

# Drop any accidental NaNs in calculation columns
merged_df = merged_df.dropna(subset=['Selection_Rate_Pct', 'In_List', 'Frequency_Count'])

# ==========================================
# 6. Run Regressions
# ==========================================

print("\n" + "="*80)
print("PART A: POOLED OLS (All stocks together, Clustered Standard Errors)")
print("="*80)

# Model 1 (Pooled)
print("\n--- Model 1 (Pooled): Does being on the list predict a higher selection rate? ---")
X1 = sm.add_constant(merged_df['In_List'])
y = merged_df['Selection_Rate_Pct']
# Clustered standard errors by Stock
model1_pooled = sm.OLS(y, X1).fit(cov_type='cluster', cov_kwds={'groups': merged_df['Stock']})
print(model1_pooled.summary())

# Model 2 (Pooled)
print("\n--- Model 2 (Pooled): Does frequency count predict a higher selection rate? ---")
X2 = sm.add_constant(merged_df['Frequency_Count'])
model2_pooled = sm.OLS(y, X2).fit(cov_type='cluster', cov_kwds={'groups': merged_df['Stock']})
print(model2_pooled.summary())


print("\n" + "="*80)
print("PART B: FIXED EFFECTS OLS (Controlling for baseline stock differences)")
print("="*80)

if HAS_LINEARMODELS:
    # FIX: linearmodels requires the second index level to be numeric or datetime.
    # Convert the string 'Topic_LASSO' into a numeric 'Topic_ID'
    merged_df['Topic_ID'] = merged_df['Topic_LASSO'].astype('category').cat.codes
    
    # Set a MultiIndex (Entity=Stock, "Time"=Topic_ID) required for linearmodels
    panel_data = merged_df.set_index(['Stock', 'Topic_ID'])

    print("\n--- Model 1 (Stock Fixed Effects): In_List ---")
    mod1_fe = PanelOLS(panel_data['Selection_Rate_Pct'], 
                       sm.add_constant(panel_data['In_List']), 
                       entity_effects=True) # Enables Stock Fixed Effects
    res1_fe = mod1_fe.fit(cov_type='clustered', cluster_entity=True)
    print(res1_fe.summary)

    print("\n--- Model 2 (Stock Fixed Effects): Frequency_Count ---")
    mod2_fe = PanelOLS(panel_data['Selection_Rate_Pct'], 
                       sm.add_constant(panel_data['Frequency_Count']), 
                       entity_effects=True)
    res2_fe = mod2_fe.fit(cov_type='clustered', cluster_entity=True)
    print(res2_fe.summary)

else:
    # Fallback to statsmodels using the "Within Transformation" (Demeaning)
    print("Using De-meaned approach for Fixed Effects (Within Transformation)")
    
    # Calculate within-stock deviations (Value - Stock Mean)
    merged_df['Sel_Rate_Demeaned'] = merged_df.groupby('Stock')['Selection_Rate_Pct'].transform(lambda x: x - x.mean())
    merged_df['In_List_Demeaned'] = merged_df.groupby('Stock')['In_List'].transform(lambda x: x - x.mean())
    merged_df['Freq_Demeaned'] = merged_df.groupby('Stock')['Frequency_Count'].transform(lambda x: x - x.mean())

    print("\n--- Model 1 (Stock Fixed Effects Fallback): In_List ---")
    # No constant needed because data is strictly demeaned
    mod1_fe = sm.OLS(merged_df['Sel_Rate_Demeaned'], merged_df['In_List_Demeaned']).fit(cov_type='cluster', cov_kwds={'groups': merged_df['Stock']})
    print(mod1_fe.summary())

    print("\n--- Model 2 (Stock Fixed Effects Fallback): Frequency_Count ---")
    mod2_fe = sm.OLS(merged_df['Sel_Rate_Demeaned'], merged_df['Freq_Demeaned']).fit(cov_type='cluster', cov_kwds={'groups': merged_df['Stock']})
    print(mod2_fe.summary())