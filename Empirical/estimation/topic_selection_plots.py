"""
Topic selection frequency analysis.

Loads the beta tensor produced by cross_sectional_estimation.py and generates
plots examining which topics / macro series get selected by the LASSO.

The model predicts that, after global standardisation of X, selection probability
should be driven purely by statistical properties (e.g. correlation with returns),
with no systematic preference for any particular topic ex ante.

Outputs (saved in the same directory as the tensor):
  topic_selection_frequency.png   — ranked bar chart, all 190 features
  topic_selection_concentration.png — Lorenz curve + macro vs topics boxplot
  topic_selection_heatmap.png     — stock × topic heat-map (top-active stocks)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import h5py
from scipy.stats import entropy as scipy_entropy

TENSOR_DIR = '../../Results/Estimation/Cross_Sectional_holdout'
OUT_DIR    = TENSOR_DIR          # save plots alongside the tensor

# ── Load tensor ───────────────────────────────────────────────────────────────
print("Loading tensor …")
with h5py.File(os.path.join(TENSOR_DIR, 'betas.h5'), 'r') as f:
    betas  = f['betas'][:]             # (n_stocks, n_topics, n_dates) float32
    stocks = f['stocks'][:].astype(str)
    topics = f['topics'][:].astype(str)
    dates  = pd.to_datetime(f['dates'][:].astype(str))

print(f"Tensor: {betas.shape}  ({len(stocks)} stocks × {len(topics)} features × {len(dates)} dates)")

# ── Mark macro features (first 10 before underscore replacement) ──────────────
MACRO_KEYWORDS = ['Fed_Funds', '10-Year_Treasury', 'VIX', 'Financial_Stress',
                  'High_Yield', 'Trade_Weighted', '3-Month_Treasury',
                  'WTI_Crude', 'Breakeven', 'USD_to']
is_macro = np.array([any(kw in t for kw in MACRO_KEYWORDS) for t in topics])
# Also catch by position: first 10 topics alphabetically may differ; use name
# The tensor topics are sorted alphabetically by the estimation script
# Let's just flag by checking if they appear in the original first-10
MACRO_SET = {
    '10-Year_Breakeven_Inflation_Rate', '10-Year_Treasury_Yield',
    '3-Month_Treasury_Yield', 'Fed_Funds_Effective_Rate',
    'Financial_Stress_Index', 'High_Yield_Option-Adjusted_Spread',
    'Trade_Weighted_USD_Index', 'USD_to_JPY', 'VIX_Volatility_Index',
    'WTI_Crude_Oil'
}
is_macro = np.array([t in MACRO_SET for t in topics])
print(f"Macro features: {is_macro.sum()}  |  Topic features: {(~is_macro).sum()}")

# ── Selection indicator: betas != 0 ──────────────────────────────────────────
sel = (betas != 0).astype(np.float32)   # (stocks, features, dates)

# Per-feature mean selection rate across stocks and dates
# Ignore NaN windows (padding); betas==0 could be genuine or padding
# Use only dates where at least one stock has a non-nan beta
valid_dates = np.isfinite(betas).any(axis=(0,1))   # (dates,)
sel_valid = sel[:, :, valid_dates]                  # (stocks, features, valid_dates)

# mean over dates then over stocks
per_stock_feat_rate = sel_valid.mean(axis=2)        # (stocks, features)
mean_sel = per_stock_feat_rate.mean(axis=0)         # (features,)
std_sel  = per_stock_feat_rate.std(axis=0)

# fraction of stocks that EVER select each feature
ever_sel = (per_stock_feat_rate > 0).mean(axis=0)

# per-stock total selection rate
per_stock_rate = sel_valid.mean(axis=(1, 2))        # (stocks,)

sort_idx = np.argsort(mean_sel)[::-1]
topics_sorted  = topics[sort_idx]
mean_sel_sorted = mean_sel[sort_idx]
is_macro_sorted = is_macro[sort_idx]

# null uniform rate
null_rate = mean_sel.sum() / len(topics)
print(f"\nOverall mean selection rate: {mean_sel.sum():.4f}")
print(f"Null uniform rate per feature: {null_rate:.6f}")
print(f"Top feature / null ratio: {mean_sel_sorted[0] / null_rate:.1f}×")
print(f"\nTop 15 most selected features:")
for i in range(15):
    tag = '[MACRO]' if is_macro_sorted[i] else '[topic]'
    print(f"  {i+1:2d}. {topics_sorted[i]:<45s} {mean_sel_sorted[i]*100:.3f}%  {tag}")

# ── FIGURE 1: Ranked bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

colors = ['#d62728' if m else '#1f77b4' for m in is_macro_sorted]
ax = axes[0]
ax.bar(range(len(mean_sel_sorted)), mean_sel_sorted * 100,
       color=colors, alpha=0.85, width=1.0)
ax.axhline(null_rate * 100, color='black', ls='--', lw=1.5,
           label=f'Uniform null ({null_rate*100:.4f}%)')
ax.set_xlabel('Feature rank (1 = most selected)')
ax.set_ylabel('Mean selection rate (%)')
ax.set_title('Selection frequency by feature (all 190, ranked)')
legend_handles = [
    mpatches.Patch(color='#d62728', label=f'Macro series ({is_macro.sum()})'),
    mpatches.Patch(color='#1f77b4', label=f'News topics ({(~is_macro).sum()})'),
    plt.Line2D([0],[0], color='k', ls='--', lw=1.5, label='Uniform null'),
]
ax.legend(handles=legend_handles, fontsize=9)

# Right panel: top 40 named
n_top = 40
top_names   = topics_sorted[:n_top]
top_vals    = mean_sel_sorted[:n_top]
top_colors  = ['#d62728' if m else '#1f77b4' for m in is_macro_sorted[:n_top]]

ax2 = axes[1]
ax2.barh(range(n_top), top_vals[::-1] * 100, color=top_colors[::-1], alpha=0.85)
ax2.set_yticks(range(n_top))
ax2.set_yticklabels([t.replace('_', ' ') for t in top_names[::-1]], fontsize=7)
ax2.axvline(null_rate * 100, color='black', ls='--', lw=1.5, label='Uniform null')
ax2.set_xlabel('Mean selection rate (%)')
ax2.set_title('Top 40 most-selected features')
ax2.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'topic_selection_frequency.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {out}")

# ── FIGURE 2: Concentration (Lorenz) + macro vs topics boxplot ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lorenz curve
vals_sorted = np.sort(mean_sel)
cum = np.cumsum(vals_sorted) / vals_sorted.sum()
x_pct = np.linspace(0, 100, len(cum))

n = len(vals_sorted)
gini = 1 - 2 * np.trapz(cum, np.linspace(0, 1, n))

ax = axes[0]
ax.plot(x_pct, cum * 100, color='#1f77b4', lw=2, label='Actual')
ax.plot([0, 100], [0, 100], 'k--', lw=1.2, label='Uniform')
ax.fill_between(x_pct, cum * 100, np.linspace(0, 100, n), alpha=0.15, color='#1f77b4')
ax.set_xlabel('Cumulative share of features (%)')
ax.set_ylabel('Cumulative share of total selections (%)')
ax.set_title('Lorenz curve of selection concentration across features')
ax.text(0.05, 0.88, f'Gini = {gini:.3f}', transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', fc='white', ec='gray'))
ax.legend()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)

# Boxplot macro vs topics
macro_rates = mean_sel[is_macro] * 100
topic_rates = mean_sel[~is_macro] * 100

ax2 = axes[1]
bp = ax2.boxplot([macro_rates, topic_rates],
                 labels=[f'Macro\n(N={is_macro.sum()})', f'News topics\n(N={(~is_macro).sum()})'],
                 patch_artist=True, notch=False,
                 boxprops=dict(alpha=0.6),
                 medianprops=dict(color='black', lw=2))
bp['boxes'][0].set_facecolor('#d62728')
bp['boxes'][1].set_facecolor('#1f77b4')
ax2.axhline(null_rate * 100, color='k', ls='--', lw=1.5, label='Uniform null')
ax2.set_ylabel('Mean selection rate (%)')
ax2.set_title('Selection rate: macro series vs.\ news topics')
ax2.legend(fontsize=9)

mean_macro = macro_rates.mean()
mean_topic = topic_rates.mean()
print(f"Mean selection rate — macro: {mean_macro:.4f}%  topics: {mean_topic:.4f}%  ratio: {mean_macro/mean_topic:.2f}×")

plt.tight_layout()
out = os.path.join(OUT_DIR, 'topic_selection_concentration.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# ── FIGURE 3: Stock × feature heat-map (top 50 most active stocks) ───────────
n_show = min(50, len(stocks))
total_sel_per_stock = per_stock_feat_rate.sum(axis=1)
top_stock_idx = np.argsort(total_sel_per_stock)[::-1][:n_show]

heatmap_data = per_stock_feat_rate[top_stock_idx, :]   # (n_show, features)
heatmap_df   = pd.DataFrame(heatmap_data,
                             index=stocks[top_stock_idx],
                             columns=[t.replace('_', ' ') for t in topics])

fig, ax = plt.subplots(figsize=(22, 10))
sns.heatmap(heatmap_df, cmap='YlOrRd', ax=ax,
            xticklabels=True, yticklabels=False,
            linewidths=0, cbar_kws={'label': 'Mean selection rate'})
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
# Highlight macro columns with a border
n_cols = len(topics)
for j, t in enumerate(topics):
    if t in MACRO_SET:
        ax.add_patch(plt.Rectangle((j, 0), 1, n_show,
                                   fill=False, edgecolor='blue', lw=1.5))
ax.set_title(f'LASSO selection rate: top {n_show} most active stocks × all features\n'
             f'(blue borders = macro series)', fontsize=11)
ax.set_xlabel('Feature')
ax.set_ylabel('Stock (ranked by total selection activity)')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'topic_selection_heatmap.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# ── Print summary stats for paper ────────────────────────────────────────────
top10_share = mean_sel_sorted[:10].sum() / mean_sel.sum() * 100
print(f"\n{'='*60}")
print("SUMMARY FOR PAPER")
print(f"{'='*60}")
print(f"N stocks analysed       : {len(stocks)}")
print(f"N features              : {len(topics)}  ({is_macro.sum()} macro + {(~is_macro).sum()} topics)")
print(f"Gini coefficient        : {gini:.3f}")
print(f"Top-10 share of total   : {top10_share:.1f}%")
print(f"Mean sel rate macro     : {mean_macro:.4f}%")
print(f"Mean sel rate topics    : {mean_topic:.4f}%")
print(f"Macro / topic ratio     : {mean_macro/mean_topic:.2f}×")
print(f"% features never selected: {(mean_sel == 0).mean()*100:.1f}%")
print(f"\nTop 10:")
for i in range(10):
    tag = '[MACRO]' if is_macro_sorted[i] else '[topic]'
    print(f"  {i+1:2d}. {topics_sorted[i].replace('_',' '):<45s} {mean_sel_sorted[i]*100:.3f}%  {tag}")
