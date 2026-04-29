"""
Time-series decomposition of LASSO selection rates.

Two exercises:
  1. Regression: bar_s_t = a + b1*sigma_r_t + b2*sigma_x_t + eps_t
     Tests whether time variation in aggregate selection is explained by
     (a) return volatility [+] and (b) predictor volatility [-].
     Under the null model with constant sigma_u, bar_s_t is FLAT — any
     significant beta_1 or beta_2 points beyond the basic noise mechanism.

  2. Oracle simulation: replace actual returns with iid N(0, sigma_u_i^2)
     noise for N_SIM stocks, rerun rolling LASSO with same (lambda_i, window),
     and compare the simulated selection-rate time series to the actual one.
     Model-implied selection should be nearly flat; systematic spikes in the
     actual series reveal extra-model variation (time-varying volatility,
     correlated noise, genuine predictability episodes).
"""

import os, sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TENSOR  = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled/betas.h5')
FEAT    = os.path.join(ROOT, 'Data/clean_data/final_macro_topic_features.csv')
SUMMARY = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled/summary.csv')
OUT     = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled')

N_SIM  = 150    # stocks to include in oracle simulation
N_JOBS = -1     # parallel workers

# ── 1. Load tensor ─────────────────────────────────────────────────────────────
print("Loading tensor …")
with h5py.File(TENSOR, 'r') as f:
    dates_raw = [d.decode() for d in f['dates'][:]]
    stocks    = f['stocks'][:].astype(str)
    topics    = f['topics'][:].astype(str)
    n_s, n_t, n_d = f['betas'].shape

    # Per-date aggregate selection rate: mean over (stocks, topics) of 1[beta!=0]
    # Load in batches to avoid OOM
    sel_sum = np.zeros(n_d, dtype=np.float64)
    targets = np.zeros((n_s, n_d), dtype=np.float32)
    BATCH = 100
    for start in range(0, n_s, BATCH):
        end = min(start + BATCH, n_s)
        chunk = f['betas'][start:end]                         # (batch, topics, dates)
        sel_sum += (chunk != 0).sum(axis=(0, 1))
        targets[start:end] = f['targets'][start:end]

agg_rate = sel_sum / (n_s * n_t)   # (n_dates,)
dates    = pd.to_datetime(dates_raw)

print(f"Tensor: ({n_s}, {n_t}, {n_d})  aggregate mean sel = {agg_rate.mean()*100:.4f}%")

# ── 2. Cross-sectional return volatility: sigma_r_t ────────────────────────────
# Cross-sectional std of returns across stocks at each date
sigma_r = targets.std(axis=0)    # (n_dates,)

# ── 3. Predictor volatility: sigma_x_t ────────────────────────────────────────
# Average absolute value of raw topic innovations at each date
# Use the full feature matrix (before within-window standardisation)
print("Loading features …")
feat_df = pd.read_csv(FEAT, index_col=0)
feat_df.index = pd.to_datetime(feat_df.index)
feat_df = feat_df.reindex(dates).ffill().bfill()  # align; fill any date gaps

# sigma_x_t = cross-feature std of topic innovations at date t
# Proxy for "how active/noisy are topics today"
sigma_x = feat_df.abs().mean(axis=1).values    # (n_dates,)

# ── 4. Newey-West regression ───────────────────────────────────────────────────
print("\nRunning regression bar_s_t ~ sigma_r_t + sigma_x_t …")

def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values

# Smooth to reduce microstructure noise
y  = smooth(agg_rate * 100)   # selection rate in %
x1 = smooth(sigma_r)
x2 = smooth(sigma_x)

# Standardise regressors for interpretable betas (use nanmean/nanstd)
x1_std = (x1 - np.nanmean(x1)) / np.nanstd(x1)
x2_std = (x2 - np.nanmean(x2)) / np.nanstd(x2)

X = np.column_stack([np.ones(len(y)), x1_std, x2_std])

# Drop any NaN rows
mask = np.isfinite(y) & np.isfinite(x1_std) & np.isfinite(x2_std)
y_c = y[mask]; X_c = X[mask]
print(f"  Non-NaN observations: {mask.sum()} / {len(y)}")

# OLS via normal equations (more robust than lstsq with near-singular)
b = np.linalg.solve(X_c.T @ X_c, X_c.T @ y_c)
resid_c = y_c - X_c @ b
resid   = y - X @ b   # full series (may have NaN at edges from smoothing)
resid   = np.where(np.isfinite(resid), resid, 0.0)
n, k = int(mask.sum()), 3

# Newey-West covariance (bandwidth = 20 lags)
NW_LAGS = 20
def newey_west(X, e, nlags):
    n, k = X.shape
    S = (e[:, None] * X).T @ (e[:, None] * X) / n
    for lag in range(1, nlags + 1):
        w = 1 - lag / (nlags + 1)
        Gamma = (e[lag:, None] * X[lag:]).T @ (e[:-lag, None] * X[:-lag]) / n
        S += w * (Gamma + Gamma.T)
    XX_inv = np.linalg.inv(X.T @ X / n)
    return XX_inv @ S @ XX_inv / n

V_nw = newey_west(X_c, resid_c, NW_LAGS)
se_nw = np.sqrt(np.diag(V_nw))
t_nw  = b / se_nw
p_nw  = 2 * stats.t.sf(np.abs(t_nw), df=n - k)
R2    = 1 - resid_c.var() / y_c.var()

print(f"\n  R² = {R2:.4f}")
print(f"  {'Variable':<22} {'beta':>8} {'SE(NW)':>9} {'t':>8} {'p':>8}")
print(f"  {'-'*58}")
labels = ['Intercept', 'σ_r (return vol)', 'σ_x (predictor act.)']
for i, lab in enumerate(labels):
    stars = '***' if p_nw[i]<0.01 else '**' if p_nw[i]<0.05 else '*' if p_nw[i]<0.10 else ''
    print(f"  {lab:<22} {b[i]:>8.4f} {se_nw[i]:>9.4f} {t_nw[i]:>8.3f} {p_nw[i]:>7.4f} {stars}")

# Also: Spearman correlations
rho_r, p_r = stats.spearmanr(agg_rate, sigma_r)
rho_x, p_x = stats.spearmanr(agg_rate, sigma_x)
print(f"\n  Spearman ρ(sel, σ_r) = {rho_r:+.4f}  p={p_r:.4f}")
print(f"  Spearman ρ(sel, σ_x) = {rho_x:+.4f}  p={p_x:.4f}")

# ── 5. Oracle simulation ───────────────────────────────────────────────────────
print(f"\nOracle simulation: {N_SIM} stocks, iid noise …")

summary = pd.read_csv(SUMMARY)
summary['stock'] = summary['stock'].astype(str)

# Pick N_SIM stocks with non-degenerate params
sim_stocks = summary[summary['avg_selection_rate'] > 0.001].sample(
    min(N_SIM, len(summary)), random_state=42)

# Align features to dates
feat_arr = feat_df.values.astype(np.float32)   # (n_dates, n_topics)
topic_to_col = {t: i for i, t in enumerate(feat_df.columns)}
feat_cols = [topic_to_col.get(t, -1) for t in topics]
feat_aligned = np.zeros((n_d, n_t), dtype=np.float32)
for j, col in enumerate(feat_cols):
    if col >= 0:
        feat_aligned[:, j] = feat_arr[:, col]

# Reload actual returns for simulation stocks
stock_to_idx = {s: i for i, s in enumerate(stocks)}

def simulate_stock(row, seed):
    """Run rolling LASSO on iid-noise returns; return selection rate per date."""
    permno   = str(row['stock'])
    lam      = float(row['eff_lambda'])
    vol      = float(row['stock_vol'])
    window   = int(row['window_size'])
    s_idx    = stock_to_idx.get(permno)
    if s_idx is None:
        return None

    rng = np.random.default_rng(seed)
    # Simulate iid returns with same volatility
    r_sim = rng.normal(0, vol, size=n_d).astype(np.float32)

    sel_t = np.zeros(n_d, dtype=np.float32)
    for t in range(window, n_d):
        X_w = feat_aligned[t - window:t, :].copy()
        y_w = r_sim[t - window:t]
        # Within-window standardise X
        mu  = X_w.mean(axis=0); sd = X_w.std(axis=0)
        sd[sd < 1e-8] = 1.0
        X_w = (X_w - mu) / sd
        try:
            mdl = Lasso(alpha=lam, fit_intercept=True, max_iter=500, warm_start=False)
            mdl.fit(X_w, y_w)
            sel_t[t] = (mdl.coef_ != 0).sum()
        except Exception:
            pass
    return sel_t / n_t    # selection rate (fraction of topics)

print("  Running parallel LASSO on simulated returns …")
results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(simulate_stock)(row, seed=i)
    for i, (_, row) in enumerate(sim_stocks.iterrows())
)
results = [r for r in results if r is not None]
sim_sel = np.stack(results, axis=0)   # (N_SIM, n_dates)
sim_mean = sim_sel.mean(axis=0)       # (n_dates,) simulated aggregate sel rate

print(f"  Simulated mean sel rate: {sim_mean.mean()*100:.4f}%  (actual: {agg_rate.mean()*100:.4f}%)")
print(f"  Simulated std: {sim_mean.std()*100:.4f}%  (actual std: {agg_rate.std()*100:.4f}%)")
print(f"  Ratio actual/simulated std: {agg_rate.std()/sim_mean.std():.2f}x")
rho_os, p_os = stats.spearmanr(agg_rate, sim_mean)
print(f"  Spearman ρ(actual, simulated): {rho_os:.4f}  p={p_os:.4f}")

# ── 6. Figures ─────────────────────────────────────────────────────────────────
events = {
    '2011-08-05': 'S&P\ndngrade',
    '2013-05-22': 'Taper\ntantrum',
    '2015-08-24': 'China\ncrash',
    '2016-06-23': 'Brexit',
    '2016-11-08': 'US\nelection',
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

def add_events(ax, ymin, ymax):
    for dt_str, label in events.items():
        dt = pd.Timestamp(dt_str)
        if dt < dates[0] or dt > dates[-1]: continue
        ax.axvline(dt, color='red', lw=0.8, ls='--', alpha=0.5)
        ax.text(dt, ymax*0.96, label, fontsize=6, ha='center', va='top',
                color='darkred', bbox=dict(boxstyle='round,pad=0.1',
                fc='white', ec='none', alpha=0.7))

w = 10   # smoothing window

# Panel 1: actual vs simulated selection rate
ax = axes[0]
ax.plot(dates, smooth(agg_rate*100, w), color='steelblue', lw=1.8, label='Actual')
ax.fill_between(dates,
    smooth(np.percentile(sim_sel*100, 10, axis=0), w),
    smooth(np.percentile(sim_sel*100, 90, axis=0), w),
    alpha=0.25, color='orange', label='Simulated p10–p90')
ax.plot(dates, smooth(sim_mean*100, w), color='orange', lw=1.5, ls='--',
        label='Simulated mean (iid noise)')
add_events(ax, *ax.get_ylim())
ax.set_ylabel('Aggregate selection rate (%)')
ax.set_title('Actual vs simulated (iid noise) aggregate selection rate')
ax.legend(fontsize=9)

# Panel 2: return vol and predictor activity
ax2 = axes[1]
ax2.plot(dates, smooth(sigma_r*100, w), color='steelblue', lw=1.5,
         label='σ_r: cross-sect. return vol (%)')
ax2b = ax2.twinx()
ax2b.plot(dates, smooth(sigma_x, w), color='orange', lw=1.4, ls='--',
          label='σ_x: avg topic activity (abs)')
ax2b.set_ylabel('Avg topic |innovation|', color='darkorange')
ax2b.tick_params(axis='y', labelcolor='darkorange')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, fontsize=9)
ax2.set_ylabel('Cross-sectional return std (%)')
ax2.set_title(f'Return volatility (+) and predictor activity (−): '
              f'ρ(sel,σ_r)={rho_r:+.3f}, ρ(sel,σ_x)={rho_x:+.3f}')
add_events(ax2, *ax2.get_ylim())

# Panel 3: regression residuals
ax3 = axes[2]
resid_smooth = smooth(resid, w)
ax3.plot(dates, resid_smooth, color='steelblue', lw=1.2)
ax3.axhline(0, color='black', lw=0.8, ls=':')
ax3.fill_between(dates, resid_smooth, 0,
                 where=resid_smooth>0, alpha=0.3, color='steelblue')
ax3.fill_between(dates, resid_smooth, 0,
                 where=resid_smooth<0, alpha=0.3, color='salmon')
ax3.set_ylabel('Regression residual (pp)')
ax3.set_xlabel('Date')
ax3.set_title(f'Unexplained time variation in selection rate  '
              f'(R²={R2:.3f}; β_σr={b[1]:+.3f}***, β_σx={b[2]:+.3f})')
add_events(ax3, *ax3.get_ylim())

for ax_ in axes:
    ax_.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
out_fig = os.path.join(OUT, 'selection_rate_decomposition.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {out_fig}")

# Save results table
reg_df = pd.DataFrame({
    'variable': labels,
    'beta': b, 'se_nw': se_nw, 't': t_nw, 'p': p_nw
})
reg_df.to_csv(os.path.join(OUT, 'selection_rate_regression.csv'), index=False)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"R² of regression           : {R2:.4f}")
print(f"β(σ_r)  t={t_nw[1]:.2f}  p={p_nw[1]:.4f}")
print(f"β(σ_x)  t={t_nw[2]:.2f}  p={p_nw[2]:.4f}")
print(f"Actual/simulated std ratio : {agg_rate.std()/sim_mean.std():.2f}x")
print(f"ρ(actual, simulated)       : {rho_os:.4f}  p={p_os:.4f}")
