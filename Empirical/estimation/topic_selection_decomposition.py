"""
Variance decomposition of topic selection rates.

Decomposes Var(s_bar_{ij}) into:
  - Stock fixed effects  (some stocks select more overall)
  - Topic fixed effects  (some topics are more universally selected)
  - Residual             (idiosyncratic stock x topic)

Also reports:
  - Distribution of topic-level mean selection rates
  - Top/bottom 20 topics by mean selection rate
  - Scatter: topic mean vs cross-stock std
  - Gini coefficient of topic popularity
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TENSOR = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled/betas.h5')
OUT    = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled')

# ── 1. Load tensor and compute s_bar_{ij} ─────────────────────────────────────
print("Loading tensor …")
with h5py.File(TENSOR, 'r') as f:
    stocks = f['stocks'][:].astype(str)
    topics = f['topics'][:].astype(str)
    n_s, n_t, n_d = f['betas'].shape
    # mean selection rate per (stock, topic) pair
    S = np.zeros((n_s, n_t), dtype=np.float64)
    for start in range(0, n_s, 100):
        end = min(start + 100, n_s)
        S[start:end] = (f['betas'][start:end] != 0).mean(axis=2)

print(f"  S shape: {S.shape}  ({n_s} stocks x {n_t} topics)")
print(f"  Overall mean s_bar: {S.mean()*100:.4f}%")
print(f"  Overall std  s_bar: {S.std()*100:.4f}%")

# ── 2. Variance decomposition via two-way within-transformation ───────────────
# Iterative demeaning (Gauss-Seidel) to project out both FEs simultaneously
def two_way_demean(X, tol=1e-10, max_iter=500):
    """Alternating projections to remove row and column means."""
    R = X - X.mean()          # start from grand-demeaned
    for _ in range(max_iter):
        R_new = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()
        # refine
        row_means = R_new.mean(axis=1, keepdims=True)
        col_means = R_new.mean(axis=0, keepdims=True)
        R_new2 = R_new - row_means - col_means + R_new.mean()
        if np.max(np.abs(R_new2 - R_new)) < tol:
            return R_new2
        R_new = R_new2
    return R_new

grand_mean   = S.mean()
TSS          = np.var(S)                                         # total variance

# Stock FE only: residual = s_ij - s_i.
stock_means  = S.mean(axis=1, keepdims=True)                     # (n_s, 1)
resid_stock  = S - stock_means
R2_stock     = 1 - np.var(resid_stock) / TSS

# Topic FE only: residual = s_ij - s_.j
topic_means  = S.mean(axis=0, keepdims=True)                     # (1, n_t)
resid_topic  = S - topic_means
R2_topic     = 1 - np.var(resid_topic) / TSS

# Both FEs: two-way within residual
resid_both   = two_way_demean(S)
R2_both      = 1 - np.var(resid_both) / TSS

# Marginal contributions (order-independent Shapley-style split)
stock_marginal = R2_both - R2_topic     # stock FE on top of topic FE
topic_marginal = R2_both - R2_stock     # topic FE on top of stock FE
# Simple sequential (stock first)
topic_seq    = R2_both - R2_stock
stock_seq    = R2_stock
resid_share  = 1 - R2_both

print(f"\n── Variance decomposition ──────────────────────────────────")
print(f"  R²(stock FE only)  : {R2_stock*100:.2f}%")
print(f"  R²(topic FE only)  : {R2_topic*100:.2f}%")
print(f"  R²(both FEs)       : {R2_both*100:.2f}%")
print(f"  Residual (idiosync): {resid_share*100:.2f}%")
print(f"  Marginal topic FE  : {topic_marginal*100:.2f}%  (on top of stock FE)")
print(f"  Marginal stock FE  : {stock_marginal*100:.2f}%  (on top of topic FE)")

# ── 3. Topic-level statistics ─────────────────────────────────────────────────
topic_mean   = S.mean(axis=0)          # mean selection rate across stocks
topic_std    = S.std(axis=0)           # cross-stock std
topic_cv     = topic_std / (topic_mean + 1e-12)  # coefficient of variation

# Gini coefficient of topic popularity
def gini(x):
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum(np.arange(1, n+1) * x) / (n * x.sum())) - (n+1)/n

G = gini(topic_mean)
print(f"\n── Topic-level statistics ──────────────────────────────────")
print(f"  Topic mean sel rate: min={topic_mean.min()*100:.4f}%  "
      f"max={topic_mean.max()*100:.4f}%  std={topic_mean.std()*100:.4f}%")
print(f"  Gini coefficient   : {G:.4f}  (0=uniform, 1=maximally concentrated)")

# Top 20 and bottom 20
order_desc = np.argsort(topic_mean)[::-1]
print(f"\n  Top 20 topics by mean selection rate:")
for k in order_desc[:20]:
    print(f"    {topics[k]:<50s}  {topic_mean[k]*100:.4f}%  (σ={topic_std[k]*100:.4f}%)")
print(f"\n  Bottom 10 topics by mean selection rate:")
for k in order_desc[-10:]:
    print(f"    {topics[k]:<50s}  {topic_mean[k]*100:.4f}%")

# Correlation between topic mean and cross-stock std
rho_mean_std, p_ms = spearmanr(topic_mean, topic_std)
print(f"\n  Spearman ρ(topic mean, cross-stock std) = {rho_mean_std:.3f}  p={p_ms:.4f}")

# ── 4. Save results ───────────────────────────────────────────────────────────
topic_df = pd.DataFrame({
    'topic':      topics,
    'mean_sel':   topic_mean,
    'std_sel':    topic_std,
    'cv_sel':     topic_cv,
    'rank':       np.argsort(np.argsort(-topic_mean)) + 1,
})
topic_df.to_csv(os.path.join(OUT, 'topic_selection_rates.csv'), index=False)

# ── 5. Figures ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A: variance decomposition bar ───────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
components = {
    'Stock FE\n(overall propensity)': R2_stock * 100,
    'Topic FE\n(universal popularity)': topic_seq * 100,
    'Residual\n(idiosyncratic)': resid_share * 100,
}
colors_a = ['steelblue', 'darkorange', 'silver']
bars = ax_a.bar(range(3), list(components.values()), color=colors_a,
                edgecolor='white', width=0.55)
ax_a.set_xticks(range(3))
ax_a.set_xticklabels(list(components.keys()), fontsize=10)
ax_a.set_ylabel('Share of total variance (%)')
ax_a.set_title('Variance decomposition of $\\bar{s}_{ij}$\n'
               f'(R² both FEs = {R2_both*100:.1f}%)', fontsize=11)
for bar, val in zip(bars, components.values()):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
              f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_a.set_ylim(0, max(components.values()) * 1.25)

# ── Panel B: Lorenz curve of topic popularity ─────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
tm_sorted = np.sort(topic_mean)
cum_share  = np.cumsum(tm_sorted) / tm_sorted.sum()
pop_share  = np.arange(1, n_t + 1) / n_t
ax_b.plot(pop_share * 100, cum_share * 100, color='steelblue', lw=2,
          label=f'Lorenz curve (Gini={G:.3f})')
ax_b.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.5, label='Perfect equality')
ax_b.fill_between(pop_share * 100, pop_share * 100, cum_share * 100,
                  alpha=0.15, color='steelblue')
ax_b.set_xlabel('Cumulative share of topics (%)')
ax_b.set_ylabel('Cumulative share of selections (%)')
ax_b.set_title('Lorenz curve of topic selection popularity\n'
               f'(Gini = {G:.3f})', fontsize=11)
ax_b.legend(fontsize=9)
ax_b.set_xlim(0, 100); ax_b.set_ylim(0, 100)

# ── Panel C: distribution of topic mean selection rates ───────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.hist(topic_mean * 100, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax_c.axvline(topic_mean.mean() * 100, color='red', ls='--', lw=1.5,
             label=f'Mean = {topic_mean.mean()*100:.3f}%')
ax_c.axvline(np.median(topic_mean) * 100, color='darkorange', ls='--', lw=1.5,
             label=f'Median = {np.median(topic_mean)*100:.3f}%')
ax_c.set_xlabel('Mean selection rate across stocks (%)')
ax_c.set_ylabel('Number of topics')
ax_c.set_title('Distribution of topic-level mean selection rates', fontsize=11)
ax_c.legend(fontsize=9)

# ── Panel D: scatter mean vs cross-stock std ──────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.scatter(topic_mean * 100, topic_std * 100, alpha=0.55, s=25,
             color='steelblue', edgecolors='none')
# Annotate top 10 topics
top10 = np.argsort(topic_mean)[-10:]
for k in top10:
    ax_d.annotate(topics[k].replace('_', ' ')[:25],
                  (topic_mean[k]*100, topic_std[k]*100),
                  fontsize=6, alpha=0.8,
                  xytext=(4, 0), textcoords='offset points')
ax_d.set_xlabel('Topic mean selection rate (%)')
ax_d.set_ylabel('Cross-stock std of selection rate (%)')
ax_d.set_title(f'Topic mean vs cross-stock heterogeneity\n'
               f'(Spearman ρ={rho_mean_std:.3f})', fontsize=11)

# ── Panel E: top 20 topics horizontal bar ─────────────────────────────────────
ax_e = fig.add_subplot(gs[2, :])
top20_idx   = order_desc[:20][::-1]   # ascending so top is at chart top
top20_names = [topics[k].replace('_', ' ') for k in top20_idx]
top20_means = topic_mean[top20_idx] * 100
top20_stds  = topic_std[top20_idx] * 100

# Color by CV: low CV = universally popular, high CV = concentrated in few stocks
cv_vals = topic_cv[top20_idx]
cmap    = plt.cm.RdYlGn_r
norm    = plt.Normalize(cv_vals.min(), cv_vals.max())
colors_e = [cmap(norm(v)) for v in cv_vals]

bars_e = ax_e.barh(range(20), top20_means, xerr=top20_stds,
                   color=colors_e, edgecolor='white', alpha=0.85,
                   error_kw=dict(elinewidth=0.8, capsize=3, ecolor='grey'))
ax_e.set_yticks(range(20))
ax_e.set_yticklabels(top20_names, fontsize=8)
ax_e.set_xlabel('Mean selection rate across stocks (%)')
ax_e.set_title('Top 20 topics by mean selection rate   '
               '(error bars = cross-stock std;  colour = CV: green=universal, red=concentrated)',
               fontsize=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax_e, label='Coefficient of variation', shrink=0.6, pad=0.01)

out_fig = os.path.join(OUT, 'topic_selection_decomposition.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {out_fig}")
