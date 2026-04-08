"""
Industry–topic narrative alignment test.

Tests whether the LASSO selects topics that are narratively coherent with
each stock's industry, i.e., whether statistical selection and economic
narrative are correlated.

Methodology:
  1. Load per-stock-per-topic selection rates from the beta tensor
  2. Map stocks (PERMNOs) → SIC codes → 8 broad sector groups
  3. Define a set of "narratively relevant" topics for each sector
  4. For each stock, compare mean selection rate on own-sector vs
     other-sector topics
  5. T-test (paired, across stocks) + panel OLS with stock and topic FEs

Output:
  industry_topic_alignment.png   — bar chart of own vs other selection rates
  industry_topic_alignment.csv   — per-stock own/other rates
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
from scipy import stats
import warnings

TENSOR_PATH  = '../../Results/Estimation/Cross_Sectional_holdout/betas.h5'
IND_PATH     = '../../Data/data_raw/industry_codes.csv'
OUT_DIR      = '../../Results/Estimation/Cross_Sectional_holdout'

# ── 1. Load tensor ─────────────────────────────────────────────────────────────
print("Loading tensor …")
with h5py.File(TENSOR_PATH, 'r') as f:
    betas  = f['betas'][:]
    stocks = f['stocks'][:].astype(str)   # PERMNOs
    topics = f['topics'][:].astype(str)

print(f"Tensor: {betas.shape}  ({len(stocks)} stocks × {len(topics)} features)")

# Per-stock-per-topic mean selection rate (average over valid dates)
valid_dates = np.isfinite(betas).any(axis=(0, 1))
sel = (betas[:, :, valid_dates] != 0).astype(np.float32)
per_stock_feat_rate = sel.mean(axis=2)   # (stocks, features)

# ── 2. SIC → broad sector mapping ─────────────────────────────────────────────
ind_df = pd.read_csv(IND_PATH)
# Keep the most recent SIC code per PERMNO
ind_df = ind_df.groupby('PERMNO')['HSICCD'].last().reset_index()
ind_df['PERMNO'] = ind_df['PERMNO'].astype(str)

def sic_to_sector(sic):
    s = int(sic)
    if 1000 <= s <= 1499 or 2900 <= s <= 2999:   return 'Energy & Mining'
    if 6000 <= s <= 6999:                          return 'Financials'
    if s in range(3570, 3580) or s in range(3660, 3700) or s in range(7370, 7380):
        return 'Technology'
    if (2830 <= s <= 2836) or (8000 <= s <= 8099): return 'Healthcare'
    if (2000 <= s <= 2399) or (5200 <= s <= 5999): return 'Consumer'
    if (2400 <= s <= 2829 and not (2830 <= s <= 2836)) or \
       (3000 <= s <= 3569) or (3700 <= s <= 3799):
        return 'Industrials'
    if 4800 <= s <= 4999:                          return 'Utilities & Telecom'
    return 'Other'

ind_df['sector'] = ind_df['HSICCD'].apply(sic_to_sector)
print("\nSector distribution:")
print(ind_df['sector'].value_counts())

# ── 3. Industry → relevant topics ─────────────────────────────────────────────
# Topic names in the tensor use underscores; map accordingly
SECTOR_TOPICS = {
    'Energy & Mining': [
        'WTI_Crude_Oil', 'Oil_drilling', 'Oil_market', 'Commodities',
        'Mining', 'Chemicals/paper'
    ],
    'Financials': [
        'Bank_loans', 'Investment_banking', 'Financial_crisis', 'Mortgages',
        'Credit_cards', 'Credit_ratings', 'Insurance', 'Pensions',
        'Nonperforming_loans', 'Savings_&_loans', 'Acquired_investment_banks',
        'Financial_reports', 'Private_equity/hedge_funds'
    ],
    'Technology': [
        'Internet', 'Software', 'Computers', 'Mobile_devices',
        'Microchips', 'Electronics'
    ],
    'Healthcare': [
        'Pharma', 'Disease', 'Health_insurance', 'Biology/chemistry/physics'
    ],
    'Consumer': [
        'Retail', 'Foods/consumer_goods', 'Soft_drinks', 'Fast_food',
        'Luxury/beverages', 'Marketing', 'Revenue_growth', 'Product_prices'
    ],
    'Industrials': [
        'Machinery', 'Automotive', 'Aerospace/defense', 'Steel',
        'Chemicals/paper', 'Rail/trucking/shipping'
    ],
    'Utilities & Telecom': [
        'Utilities', 'Phone_companies', 'Cable'
    ],
}

# Validate all topic names exist in tensor
topic_set = set(topics)
for sector, tlist in SECTOR_TOPICS.items():
    bad = [t for t in tlist if t not in topic_set]
    if bad:
        print(f"WARNING — {sector}: missing topics {bad}")
        SECTOR_TOPICS[sector] = [t for t in tlist if t in topic_set]

topic_idx = {t: i for i, t in enumerate(topics)}

# ── 4. Per-stock own-sector vs other-sector selection rates ───────────────────
permno_to_sector = dict(zip(ind_df['PERMNO'], ind_df['sector']))

results = []
for s_idx, permno in enumerate(stocks):
    sector = permno_to_sector.get(permno, None)
    if sector is None or sector == 'Other':
        continue
    own_topics   = SECTOR_TOPICS.get(sector, [])
    if len(own_topics) == 0:
        continue
    own_idx   = [topic_idx[t] for t in own_topics if t in topic_idx]
    other_idx = [i for i in range(len(topics)) if i not in set(own_idx)]

    own_rate   = per_stock_feat_rate[s_idx, own_idx].mean()
    other_rate = per_stock_feat_rate[s_idx, other_idx].mean()

    results.append({
        'permno': permno,
        'sector': sector,
        'own_rate': own_rate,
        'other_rate': other_rate,
        'diff': own_rate - other_rate,
        'n_own_topics': len(own_idx),
    })

res_df = pd.DataFrame(results)
print(f"\nStocks with sector match: {len(res_df)}")
print(f"\nOverall:")
print(f"  Mean own-sector   selection rate: {res_df['own_rate'].mean()*100:.4f}%")
print(f"  Mean other-sector selection rate: {res_df['other_rate'].mean()*100:.4f}%")
print(f"  Mean difference: {res_df['diff'].mean()*100:.4f}%")

# Paired t-test
t, p = stats.ttest_rel(res_df['own_rate'], res_df['other_rate'])
print(f"\nPaired t-test (own vs other):  t = {t:.3f},  p = {p:.4f}")

# Wilcoxon signed-rank (non-parametric)
w, p_w = stats.wilcoxon(res_df['diff'])
print(f"Wilcoxon signed-rank:          W = {w:.0f},  p = {p_w:.4f}")

# By sector
print(f"\nBy sector:")
print(f"{'Sector':<22} {'N':>5} {'Own %':>8} {'Other %':>9} {'Diff pp':>8} {'t-stat':>8} {'p':>7}")
print("-" * 70)
for sector, grp in res_df.groupby('sector'):
    t_s, p_s = stats.ttest_rel(grp['own_rate'], grp['other_rate'])
    print(f"{sector:<22} {len(grp):>5} {grp['own_rate'].mean()*100:>8.4f} "
          f"{grp['other_rate'].mean()*100:>9.4f} {grp['diff'].mean()*100:>8.4f} "
          f"{t_s:>8.3f} {p_s:>7.4f}")

# ── 5. Panel OLS with stock + topic FEs ───────────────────────────────────────
print("\nBuilding panel for FE regression …")
# For efficiency, only include stocks in res_df and all topics
panel_rows = []
sector_map = dict(zip(res_df['permno'], res_df['sector']))
for s_idx, permno in enumerate(stocks):
    sector = sector_map.get(permno, None)
    if sector is None:
        continue
    own_set = set(SECTOR_TOPICS.get(sector, []))
    for t_idx, topic in enumerate(topics):
        rate = per_stock_feat_rate[s_idx, t_idx]
        own  = int(topic in own_set)
        panel_rows.append((permno, topic, rate, own, sector))

panel_df = pd.DataFrame(panel_rows, columns=['permno', 'topic', 'sel_rate', 'own_sector', 'sector'])
print(f"Panel shape: {panel_df.shape}")

# Demean by stock and topic FEs (within estimator)
panel_df['sel_mean_stock'] = panel_df.groupby('permno')['sel_rate'].transform('mean')
panel_df['sel_mean_topic'] = panel_df.groupby('topic')['sel_rate'].transform('mean')
panel_df['sel_grand']      = panel_df['sel_rate'].mean()
panel_df['sel_within']     = (panel_df['sel_rate']
                               - panel_df['sel_mean_stock']
                               - panel_df['sel_mean_topic']
                               + panel_df['sel_grand'])

panel_df['own_mean_stock'] = panel_df.groupby('permno')['own_sector'].transform('mean')
panel_df['own_mean_topic'] = panel_df.groupby('topic')['own_sector'].transform('mean')
panel_df['own_grand']      = panel_df['own_sector'].mean()
panel_df['own_within']     = (panel_df['own_sector']
                               - panel_df['own_mean_stock']
                               - panel_df['own_mean_topic']
                               + panel_df['own_grand'])

# OLS on within-transformed data
from numpy.linalg import lstsq
X_fe = panel_df['own_within'].values.reshape(-1, 1)
y_fe = panel_df['sel_within'].values
beta_fe, _, _, _ = lstsq(np.hstack([np.ones((len(X_fe), 1)), X_fe]), y_fe, rcond=None)
resid = y_fe - beta_fe[0] - beta_fe[1] * X_fe[:, 0]
n, k = len(y_fe), 2
se = np.sqrt((resid**2).sum() / (n - k) / (X_fe[:, 0].var() * n))
t_fe = beta_fe[1] / se
p_fe = 2 * stats.t.sf(abs(t_fe), df=n - k)
print(f"\nPanel OLS (stock + topic FEs):")
print(f"  β(own_sector) = {beta_fe[1]*100:.5f} pp   t = {t_fe:.3f}   p = {p_fe:.4f}")
print(f"  [Interpretation: own-sector topics selected {beta_fe[1]*100:.4f} pp more, conditional on stock and topic means]")

# ── 6. Figures ─────────────────────────────────────────────────────────────────
SECTOR_ORDER = [s for s in [
    'Energy & Mining', 'Financials', 'Technology', 'Healthcare',
    'Consumer', 'Industrials', 'Utilities & Telecom'
] if s in res_df['sector'].unique()]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: own vs other by sector
ax = axes[0]
x = np.arange(len(SECTOR_ORDER))
w = 0.35
own_means   = [res_df[res_df['sector']==s]['own_rate'].mean()*100 for s in SECTOR_ORDER]
other_means = [res_df[res_df['sector']==s]['other_rate'].mean()*100 for s in SECTOR_ORDER]
own_se      = [res_df[res_df['sector']==s]['own_rate'].sem()*100 for s in SECTOR_ORDER]
other_se    = [res_df[res_df['sector']==s]['other_rate'].sem()*100 for s in SECTOR_ORDER]

b1 = ax.bar(x - w/2, own_means,   w, yerr=own_se,   capsize=3,
            color='#1f77b4', alpha=0.85, label='Own-sector topics')
b2 = ax.bar(x + w/2, other_means, w, yerr=other_se, capsize=3,
            color='#aec7e8', alpha=0.85, label='Other-sector topics')
ax.set_xticks(x)
ax.set_xticklabels(SECTOR_ORDER, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Mean LASSO selection rate (%)')
ax.set_title('Own-sector vs other-sector topic selection rates\n(bars = mean ± SE across stocks in sector)')
ax.legend(fontsize=9)
# Annotate p-values
for i, sector in enumerate(SECTOR_ORDER):
    grp = res_df[res_df['sector'] == sector]
    _, p_s = stats.ttest_rel(grp['own_rate'], grp['other_rate'])
    stars = '***' if p_s < 0.01 else '**' if p_s < 0.05 else '*' if p_s < 0.10 else ''
    if stars:
        ypos = max(own_means[i], other_means[i]) + max(own_se[i], other_se[i]) + 0.0005
        ax.text(i, ypos*100 if ypos < 0.01 else ypos, stars,
                ha='center', fontsize=10, color='black')

# Right: distribution of diff (own - other)
ax2 = axes[1]
colors_map = {
    'Energy & Mining': '#d62728', 'Financials': '#1f77b4', 'Technology': '#2ca02c',
    'Healthcare': '#9467bd', 'Consumer': '#ff7f0e', 'Industrials': '#8c564b',
    'Utilities & Telecom': '#17becf',
}
for sector in SECTOR_ORDER:
    grp = res_df[res_df['sector'] == sector]
    ax2.hist(grp['diff'] * 100, bins=30, alpha=0.5,
             label=f"{sector} (n={len(grp)})",
             color=colors_map.get(sector, 'gray'))
ax2.axvline(0, color='black', lw=1.5, ls='--')
ax2.axvline(res_df['diff'].mean()*100, color='red', lw=1.5, ls='-',
            label=f"Overall mean = {res_df['diff'].mean()*100:.4f} pp\nt={t:.2f}, p={p:.3f}")
ax2.set_xlabel('Own-sector rate − Other-sector rate (pp)')
ax2.set_ylabel('Number of stocks')
ax2.set_title('Distribution of narrative alignment gap (own − other)')
ax2.legend(fontsize=7, ncol=2)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'industry_topic_alignment.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {out}")

# Save per-stock results
res_df.to_csv(os.path.join(OUT_DIR, 'industry_topic_alignment.csv'), index=False)
print(f"Saved industry_topic_alignment.csv")

print(f"\n{'='*60}")
print("SUMMARY FOR PAPER")
print(f"{'='*60}")
print(f"Stocks analysed          : {len(res_df)}")
print(f"Mean own-sector rate     : {res_df['own_rate'].mean()*100:.4f}%")
print(f"Mean other-sector rate   : {res_df['other_rate'].mean()*100:.4f}%")
print(f"Mean diff                : {res_df['diff'].mean()*100:.4f} pp")
print(f"% stocks with own > other: {(res_df['diff']>0).mean()*100:.1f}%")
print(f"Paired t-test            : t={t:.3f}, p={p:.4f}")
print(f"Wilcoxon                 : W={w:.0f}, p={p_w:.4f}")
print(f"Panel FE β               : {beta_fe[1]*100:.5f} pp, t={t_fe:.3f}, p={p_fe:.4f}")
