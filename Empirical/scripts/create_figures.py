import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# --- 1. Path & Environment Setup ---

# Setup Directories
REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "Results" / "Estimation" / "Cross_Sectional"
# Using the updated path from your prompt
CODES_FILE = REPO_ROOT / "Data" / "data_raw" / "industry_codes.csv"
SAVE_DIR = Path(r"C:\Users\jonat\Lasso_paper\Results\Figures")

# Ensure the save directory exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Journal of Finance Style Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', # LaTeX-like math font
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 300
})
sns.set_style("white")

# --- 2. Data Loading & Preparation ---

def load_all_data():
    """Loads model results and industry mapping."""
    h5_path = DATA_DIR / 'betas.h5'
    
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing data file at: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        data = {
            'betas': f['betas'][:],               
            'stocks': f['stocks'][:].astype(str),
            'dates': pd.to_datetime(f['dates'][:].astype(str)),
            'r2_in_s1': f['r2_in'][:],            
            'r2_in_s2': f['stage2_r2_in'][:],     
            'targets': f['targets'][:],           
            'preds_s1': f['predictions'][:],      
            'preds_s2': f['stage2_predictions'][:] 
        }
    
    codes = pd.read_csv(CODES_FILE).dropna(subset=['HSICCD'])
    codes['PERMNO'] = codes['PERMNO'].astype(str)
    
    return data, codes

def sic_to_industry_name(sic):
    """Maps SIC codes to broad industry categories."""
    if pd.isna(sic): return "Unknown"
    s = int(sic)
    if 100 <= s <= 999:    return "Agri, Fish, Forest"
    if 1000 <= s <= 1499:  return "Mining"
    if 1500 <= s <= 1799:  return "Construction"
    if 2000 <= s <= 3999:  return "Manufacturing"
    if 4000 <= s <= 4999:  return "Transport & Utils"
    if 5000 <= s <= 5199:  return "Wholesale Trade"
    if 5200 <= s <= 5999:  return "Retail Trade"
    if 6000 <= s <= 6799:  return "Finance & Insurance"
    if 7000 <= s <= 8999:  
        if 7370 <= s <= 7379: return "Tech/Software"
        return "Services"
    if 9000 <= s <= 9999:  return "Public Admin/Misc"
    return "Unknown"

# --- 3. Core Calculation Functions ---

def calc_market_oos_r2(targets, predictions, dates):
    """Daily cross-sectional OOS R2 relative to daily mean."""
    sse = np.nansum((targets - predictions)**2, axis=0)
    sst = np.nansum((targets - np.nanmean(targets, axis=0))**2, axis=0)
    return pd.Series(1 - (sse / sst), index=dates)

def get_active_counts(betas):
    """Returns a matrix of (stocks x days) with active predictor counts."""
    is_active = np.nan_to_num(betas, nan=0.0) != 0
    active_per_day = np.sum(is_active, axis=1).astype(float)
    has_data = np.isfinite(betas).any(axis=1)
    active_per_day[~has_data] = np.nan
    return active_per_day

# --- 4. Plotting & Saving Functions ---

def save_fig(name):
    path = SAVE_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")

def plot_performance_timeseries(daily_is, daily_oos, title_prefix, fig_num):
    """Generates Figure 1 and 2 style performance panels without Figure headers."""
    ma_window = 21
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for ax, data, label, sub_title in zip([ax1, ax2], [daily_is, daily_oos], 
                                          ['In-Sample $R^2$', 'Out-of-Sample $R^2$'],
                                          ['Panel A', 'Panel B']):
        ma = data.rolling(window=ma_window, min_periods=1).mean()
        
        # JF Style: Grayscale/Clean lines
        ax.plot(data.index, data, color='#E0E0E0', lw=0.7, alpha=0.7)
        ax.plot(ma.index, ma, color='black', lw=1.5, label='21-Day MA')
        ax.axhline(data.mean(), color='black', linestyle='--', lw=1.0, label=f'Mean ({data.mean():.4f})')
        
        # JF Style: Left-aligned bold panel labels
        ax.set_title(f'{sub_title}: {title_prefix} {label}', loc='left', fontweight='bold', pad=10)
        ax.set_ylabel(label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='in', length=4)
        ax.legend(frameon=False, loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    save_fig(f"Results_{title_prefix.replace(' ', '_')}")
    plt.close()

def plot_industry_analysis(raw, codes_df):
    """JF Style Industry Bar Charts."""
    codes_df = codes_df.copy()
    codes_df['Industry'] = codes_df['HSICCD'].apply(sic_to_industry_name)
    ind_map = codes_df.set_index('PERMNO')['Industry'].to_dict()
    
    stock_perf = pd.DataFrame({'PERMNO': raw['stocks'], 'IS_R2': np.nanmean(raw['r2_in_s1'], axis=1)})
    stock_perf['Industry'] = stock_perf['PERMNO'].map(ind_map).fillna('Unknown')
    
    global_mean = np.nanmean(raw['targets'], axis=0)
    results = []
    for ind, group in stock_perf.groupby('Industry'):
        idx = group.index.values
        sse = np.nansum((raw['targets'][idx, :] - raw['preds_s1'][idx, :])**2)
        sst = np.nansum((raw['targets'][idx, :] - global_mean)**2)
        results.append({'Industry': ind, 'IS_R2': group['IS_R2'].mean(), 'OOS_R2': 1 - (sse/sst)})

    df_plot = pd.DataFrame(results).set_index('Industry').sort_values('OOS_R2', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    pct_fmt = FuncFormatter(lambda x, p: f"{x*100:.1f}%")
    
    for ax, col, title, panel in zip([ax1, ax2], ['IS_R2', 'OOS_R2'], 
                                   ['In-Sample $R^2$', 'Out-of-Sample $R^2$'], 
                                   ['Panel A', 'Panel B']):
        ax.barh(np.arange(len(df_plot)), df_plot[col], color='dimgray', height=0.7)
        ax.axvline(df_plot[col].mean(), color='black', ls='--', lw=1.2)
        ax.set_title(f'{panel}: {title}', loc='left', fontweight='bold')
        ax.xaxis.set_major_formatter(pct_fmt)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', left=False)

    ax1.set_yticks(np.arange(len(df_plot)))
    ax1.set_yticklabels(df_plot.index)
    ax1.invert_yaxis()
    plt.tight_layout()
    save_fig("Industry_Cross_Section")
    plt.close()

def plot_predictor_persistence(betas):
    """JF Style Persistence Plot."""
    is_active = np.nan_to_num(betas, nan=0.0) != 0
    durations = []
    for i in range(is_active.shape[0]):
        padded = np.pad(is_active[i], ((0,0), (1,1)), mode='constant')
        diffs = np.diff(padded.astype(int), axis=1)
        starts, ends = np.where(diffs == 1), np.where(diffs == -1)
        durs = ends[1] - starts[1]
        if durs.size > 0: durations.append(durs)
    
    max_d = max(d.max() for d in durations)
    x = np.arange(1, max_d + 1)
    surv_matrix = np.array([1.0 - (np.cumsum(np.bincount(d, minlength=max_d+2))/len(d))[x] for d in durations])
    avg_s, lo, hi = np.mean(surv_matrix, axis=0), np.percentile(surv_matrix, 2.5, axis=0), np.percentile(surv_matrix, 97.5, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = avg_s > 0.001
    ax.plot(x[mask], avg_s[mask], color='black', lw=1.5)
    ax.fill_between(x[mask], lo[mask], hi[mask], color='lightgray', alpha=0.5)
    
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1.1)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, p: f'{y*100:g}%'))
    ax.set_title("Predictor Survival Probability", loc='left', fontweight='bold')
    ax.set_xlabel('Consecutive Days (m)')
    ax.set_ylabel('Probability Active > m days')
    sns.despine()
    plt.tight_layout()
    save_fig("Predictor_Persistence")
    plt.close()

def plot_active_counts_over_time(active_counts, dates):
    """JF Style Activity Timeseries."""
    mean_a = np.nanmean(active_counts, axis=0)
    std_a = np.nanstd(active_counts, axis=0)
    ci = 1.96 * (std_a / np.sqrt(active_counts.shape[0]))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dates, mean_a - ci, mean_a + ci, color='lightgray', alpha=0.6)
    ax.plot(dates, mean_a, color='black', lw=1.2)
    ax.axhline(mean_a.mean(), color='black', ls='--', lw=1)
    
    ax.set_ylabel(r'$n_{active}$ [#/day]')
    ax.set_title("Average Number of Active Predictors per Day", loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='in')
    plt.tight_layout()
    save_fig("Active_Predictors_Time")
    plt.close()

# --- 5. Main Execution ---

if __name__ == "__main__":
    print("Starting Journal of Finance Style Figure Generation...")
    
    # Load Data
    raw, codes = load_all_data()

    # Calculate Performance Metrics
    s1_is = pd.Series(np.nanmean(raw['r2_in_s1'], axis=0), index=raw['dates'])
    s2_is = pd.Series(np.nanmean(raw['r2_in_s2'], axis=0), index=raw['dates'])
    s1_oos = calc_market_oos_r2(raw['targets'], raw['preds_s1'], raw['dates'])
    s2_oos = calc_market_oos_r2(raw['targets'], raw['preds_s2'], raw['dates'])
    active_mat = get_active_counts(raw['betas'])

    # Plot and Save in High-Res
    plot_performance_timeseries(s1_is, s1_oos, "Stage 1 Lasso", 1)
    plot_performance_timeseries(s2_is, s2_oos, "Stage 2 ALM", 2)
    plot_industry_analysis(raw, codes)
    plot_predictor_persistence(raw['betas'])
    plot_active_counts_over_time(active_mat, raw['dates'])
    
    print(f"\nProcessing Complete. Figures available in: {SAVE_DIR}")