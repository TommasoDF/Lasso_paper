"""
GARCH oracle simulation.

Tests whether replacing iid noise with a common GARCH(1,1) volatility factor
reproduces the time-series variation in aggregate LASSO selection rates.

Steps:
  1. Fit GARCH(1,1) to market (VW) returns → conditional volatility σ_{m,t}
  2. Decompose each stock's return: r_{i,t} = β_i r_{m,t} + ε_{i,t}
     → market-beta component + idiosyncratic residual σ̃_i η_{i,t}
  3. Oracle simulation: r^{sim}_{i,t} = σ_{m,t} η^{(m)}_{i,t} + σ̃_i η^{(i)}_{i,t}
  4. Run rolling LASSO on simulated returns; compare selection rate time series
     to actual and to iid baseline
  5. Re-run the regression using GARCH σ_{m,t} instead of raw cross-sect. std

Output:
  garch_selection_oracle.png
  garch_vol_comparison.csv
"""

import os, sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from arch import arch_model
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TENSOR  = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled/betas.h5')
MKT_CSV = os.path.join(ROOT, 'Data/data_raw/market_return.csv')
FEAT    = os.path.join(ROOT, 'Data/clean_data/final_macro_topic_features.csv')
SUMMARY = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled/summary.csv')
OUT     = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled')

N_SIM  = 150
N_JOBS = -1

# ── 1. Load tensor: actual selection rates + returns ───────────────────────────
print("Loading tensor …")
with h5py.File(TENSOR, 'r') as f:
    dates_raw = [d.decode() for d in f['dates'][:]]
    stocks    = f['stocks'][:].astype(str)
    topics    = f['topics'][:].astype(str)
    n_s, n_t, n_d = f['betas'].shape
    sel_sum = np.zeros(n_d, dtype=np.float64)
    targets = np.zeros((n_s, n_d), dtype=np.float32)
    for start in range(0, n_s, 100):
        end = min(start + 100, n_s)
        sel_sum += (f['betas'][start:end] != 0).sum(axis=(0, 1))
        targets[start:end] = f['targets'][start:end]

agg_rate = sel_sum / (n_s * n_t)
dates    = pd.to_datetime(dates_raw)

# ── 2. Fit GARCH(1,1) to market returns ────────────────────────────────────────
print("Fitting GARCH(1,1) to market returns …")
mkt = pd.read_csv(MKT_CSV)
mkt['DATE'] = pd.to_datetime(mkt['DATE'])
mkt = mkt.set_index('DATE').sort_index()
mkt_ret = mkt['vwretx'] * 100   # percent returns for GARCH numerical stability

# Fit on full sample
gm = arch_model(mkt_ret, vol='Garch', p=1, q=1, dist='normal', rescale=False)
res = gm.fit(disp='off', show_warning=False)
print(f"  GARCH params: ω={res.params['omega']:.6f}, α={res.params['alpha[1]']:.4f}, "
      f"β={res.params['beta[1]']:.4f}")
print(f"  Persistence α+β = {res.params['alpha[1]']+res.params['beta[1]']:.4f}")

# Align conditional volatility to tensor dates
cond_vol = res.conditional_volatility / 100   # back to decimal
cond_vol = cond_vol.reindex(dates).ffill().bfill()
sigma_m  = cond_vol.values    # (n_dates,) — GARCH conditional vol

print(f"  σ_m range: [{sigma_m.min()*100:.3f}%, {sigma_m.max()*100:.3f}%], "
      f"mean={sigma_m.mean()*100:.3f}%")

# ── 3. Market betas + idiosyncratic vol per stock ──────────────────────────────
print("Estimating market betas and idiosyncratic volatilities …")
mkt_aligned = mkt['vwretx'].reindex(dates).ffill().bfill().values   # (n_dates,)

betas_mkt = np.zeros(n_s)
sigma_idio = np.zeros(n_s)
for i in range(n_s):
    r = targets[i]
    valid = np.isfinite(r) & np.isfinite(mkt_aligned) & (r != 0)
    if valid.sum() < 100:
        sigma_idio[i] = np.nanstd(r)
        continue
    r_v = r[valid]; m_v = mkt_aligned[valid]
    beta_i = np.cov(r_v, m_v)[0, 1] / np.var(m_v)
    resid_i = r_v - beta_i * m_v
    betas_mkt[i] = beta_i
    sigma_idio[i] = resid_i.std()

print(f"  Market beta: mean={betas_mkt.mean():.3f}, median={np.median(betas_mkt):.3f}")
print(f"  Idio vol:    mean={sigma_idio.mean()*100:.3f}%, median={np.median(sigma_idio)*100:.3f}%")

# ── 4. Load features aligned to tensor dates ───────────────────────────────────
feat_df = pd.read_csv(FEAT, index_col=0)
feat_df.index = pd.to_datetime(feat_df.index)
feat_df = feat_df.reindex(dates).ffill().bfill()
sigma_x  = feat_df.abs().mean(axis=1).values

topic_to_col = {t: i for i, t in enumerate(feat_df.columns)}
feat_cols = [topic_to_col.get(t, -1) for t in topics]
feat_aligned = np.zeros((n_d, n_t), dtype=np.float32)
for j, col in enumerate(feat_cols):
    if col >= 0:
        feat_aligned[:, j] = feat_df.values[:, col]

# ── 5. Oracle simulation: GARCH noise ─────────────────────────────────────────
print(f"\nGARCH oracle simulation: {N_SIM} stocks …")
summary = pd.read_csv(SUMMARY)
summary['stock'] = summary['stock'].astype(str)
sim_stocks = summary[summary['avg_selection_rate'] > 0.001].sample(
    min(N_SIM, len(summary)), random_state=42)
stock_to_idx = {s: i for i, s in enumerate(stocks)}

def simulate_garch(row, seed, sigma_m_arr, mkt_arr, sigma_idio_arr, betas_arr):
    permno  = str(row['stock'])
    lam     = float(row['eff_lambda'])
    window  = int(row['window_size'])
    s_idx   = stock_to_idx.get(permno)
    if s_idx is None:
        return None, None

    rng = np.random.default_rng(seed)

    # GARCH simulation: r_sim = σ_m(t) * η_m + σ_idio * η_i
    # Use actual σ_m path (not re-simulate GARCH — same σ_m for all stocks)
    eta_m = rng.standard_normal(n_d)
    eta_i = rng.standard_normal(n_d)
    r_garch = (sigma_m_arr * eta_m + sigma_idio_arr[s_idx] * eta_i).astype(np.float32)

    # IID baseline (same sigma_u = total stock vol)
    sigma_u = float(np.nanstd(targets[s_idx]))
    r_iid   = (sigma_u * rng.standard_normal(n_d)).astype(np.float32)

    sel_garch = np.zeros(n_d, dtype=np.float32)
    sel_iid   = np.zeros(n_d, dtype=np.float32)

    for t in range(window, n_d):
        X_w = feat_aligned[t - window:t, :].copy()
        mu  = X_w.mean(axis=0); sd = X_w.std(axis=0)
        sd[sd < 1e-8] = 1.0
        X_w = (X_w - mu) / sd
        for r_sim, sel_arr in [(r_garch, sel_garch), (r_iid, sel_iid)]:
            y_w = r_sim[t - window:t]
            try:
                mdl = Lasso(alpha=lam, fit_intercept=True, max_iter=500)
                mdl.fit(X_w, y_w)
                nz = (mdl.coef_ != 0).sum()
            except Exception:
                nz = 0
            sel_arr[t] = nz

    return sel_garch / n_t, sel_iid / n_t

print("  Running parallel simulations …")
results = Parallel(n_jobs=N_JOBS, verbose=3)(
    delayed(simulate_garch)(
        row, seed=i, sigma_m_arr=sigma_m, mkt_arr=mkt_aligned,
        sigma_idio_arr=sigma_idio, betas_arr=betas_mkt
    )
    for i, (_, row) in enumerate(sim_stocks.iterrows())
)
garch_sels = [r[0] for r in results if r[0] is not None]
iid_sels   = [r[1] for r in results if r[1] is not None]

garch_mean = np.stack(garch_sels).mean(axis=0)
iid_mean   = np.stack(iid_sels).mean(axis=0)

def smooth(x, w=10):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values

print(f"\n  {'':30} {'mean':>8} {'std':>8} {'std ratio':>10}")
print(f"  {'Actual':30} {agg_rate.mean()*100:>8.4f} {agg_rate.std()*100:>8.4f}")
print(f"  {'GARCH simulation':30} {garch_mean.mean()*100:>8.4f} {garch_mean.std()*100:>8.4f} "
      f"{agg_rate.std()/garch_mean.std():>10.2f}x")
print(f"  {'IID simulation':30} {iid_mean.mean()*100:>8.4f} {iid_mean.std()*100:>8.4f} "
      f"{agg_rate.std()/iid_mean.std():>10.2f}x")

rho_g, p_g = stats.spearmanr(agg_rate, garch_mean)
rho_i, p_i = stats.spearmanr(agg_rate, iid_mean)
print(f"\n  ρ(actual, GARCH sim) = {rho_g:+.4f}  p={p_g:.4f}")
print(f"  ρ(actual, iid sim)   = {rho_i:+.4f}  p={p_i:.4f}")

# ── 6. Re-run regression with GARCH vol ────────────────────────────────────────
print("\nRegression with GARCH σ_m …")

def nw_reg(y, X, nlags=20):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_c = y[mask]; X_c = X[mask]
    b = np.linalg.solve(X_c.T @ X_c, X_c.T @ y_c)
    e = y_c - X_c @ b
    n, k = len(y_c), X_c.shape[1]
    S = (e[:, None]*X_c).T @ (e[:, None]*X_c) / n
    for lag in range(1, nlags+1):
        w = 1 - lag/(nlags+1)
        G = (e[lag:, None]*X_c[lag:]).T @ (e[:-lag, None]*X_c[:-lag]) / n
        S += w*(G + G.T)
    V = np.linalg.inv(X_c.T@X_c/n) @ S @ np.linalg.inv(X_c.T@X_c/n) / n
    se = np.sqrt(np.diag(V))
    t  = b/se
    p  = 2*stats.t.sf(np.abs(t), df=n-k)
    R2 = 1 - e.var()/y_c.var()
    return b, se, t, p, R2

y = smooth(agg_rate*100)

# Regression 1: raw cross-sect. vol
sigma_r = targets.std(axis=0)
x1_raw = smooth(sigma_r)
x2_raw = smooth(sigma_x)
x1s = (x1_raw - np.nanmean(x1_raw))/np.nanstd(x1_raw)
x2s = (x2_raw - np.nanmean(x2_raw))/np.nanstd(x2_raw)
b1, se1, t1, p1, R2_1 = nw_reg(y, np.column_stack([np.ones(n_d), x1s, x2s]))

# Regression 2: GARCH σ_m replacing raw vol
x1_g  = smooth(sigma_m*100)
x1gs  = (x1_g - np.nanmean(x1_g))/np.nanstd(x1_g)
b2, se2, t2, p2, R2_2 = nw_reg(y, np.column_stack([np.ones(n_d), x1gs, x2s]))

# Regression 3: GARCH σ_m only
b3, se3, t3, p3, R2_3 = nw_reg(y, np.column_stack([np.ones(n_d), x1gs]))

print(f"\n  Reg 1 (raw σ_r, σ_x):   R²={R2_1:.4f}  β_σr={b1[1]:+.4f}(t={t1[1]:.2f})  β_σx={b1[2]:+.4f}(t={t1[2]:.2f})")
print(f"  Reg 2 (GARCH σ_m, σ_x): R²={R2_2:.4f}  β_σm={b2[1]:+.4f}(t={t2[1]:.2f})  β_σx={b2[2]:+.4f}(t={t2[2]:.2f})")
print(f"  Reg 3 (GARCH σ_m only): R²={R2_3:.4f}  β_σm={b3[1]:+.4f}(t={t3[1]:.2f})")

# ── 7. Figures ─────────────────────────────────────────────────────────────────
events = {
    '2011-08-05': 'S&P\ndngrade',
    '2013-05-22': 'Taper\ntantrum',
    '2015-08-24': 'China\ncrash',
    '2016-06-23': 'Brexit',
    '2016-11-08': 'US\nelection',
}
def add_events(ax):
    yl = ax.get_ylim()
    for dt_str, label in events.items():
        dt = pd.Timestamp(dt_str)
        if dt < dates[0] or dt > dates[-1]: continue
        ax.axvline(dt, color='red', lw=0.8, ls='--', alpha=0.5)
        ax.text(dt, yl[1]*0.96, label, fontsize=6, ha='center', va='top',
                color='darkred', bbox=dict(boxstyle='round,pad=0.1', fc='white',
                ec='none', alpha=0.7))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: actual vs GARCH vs iid
ax = axes[0]
ax.plot(dates, smooth(agg_rate*100), color='steelblue', lw=1.8, label='Actual')
ax.plot(dates, smooth(garch_mean*100), color='darkorange', lw=1.5, ls='-.',
        label=f'GARCH sim (ρ={rho_g:+.3f})')
ax.plot(dates, smooth(iid_mean*100), color='grey', lw=1.2, ls='--',
        label=f'IID sim (ρ={rho_i:+.3f})')
ax.set_ylabel('Aggregate selection rate (%)')
ax.set_title('Actual vs GARCH-noise vs iid-noise oracle selection rates')
ax.legend(fontsize=9)
add_events(ax)

# Panel 2: GARCH conditional vol
ax2 = axes[1]
ax2.plot(dates, smooth(sigma_m*100), color='darkorange', lw=1.5,
         label='GARCH σ_m (conditional vol)')
ax2b = ax2.twinx()
ax2b.plot(dates, smooth(agg_rate*100), color='steelblue', lw=1.2, ls='--',
          alpha=0.7, label='Actual selection rate')
ax2b.set_ylabel('Selection rate (%)', color='steelblue')
ax2b.tick_params(axis='y', labelcolor='steelblue')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, fontsize=9)
ax2.set_ylabel('GARCH σ_m (%)')
rho_gm, _ = stats.spearmanr(
    pd.Series(sigma_m).reindex(range(n_d)).ffill().values,
    agg_rate)
ax2.set_title(f'GARCH conditional volatility vs actual selection rate  '
              f'(Spearman ρ={rho_gm:+.3f})')
add_events(ax2)

# Panel 3: R² comparison across regression specs
ax3 = axes[2]
specs  = ['IID sim\n(baseline)', 'Raw σ_r + σ_x\n(regression)',
          'GARCH σ_m + σ_x\n(regression)', 'GARCH σ_m only\n(regression)']
r2s    = [iid_mean.std()**2 / agg_rate.std()**2, R2_1, R2_2, R2_3]
colors = ['grey', 'steelblue', 'darkorange', 'darkred']
bars = ax3.bar(range(len(specs)), [r*100 for r in r2s], color=colors, alpha=0.8, edgecolor='white')
ax3.set_xticks(range(len(specs)))
ax3.set_xticklabels(specs, fontsize=9)
ax3.set_ylabel('Variance explained (%)')
ax3.set_title('How much of time variation in selection rate is explained?')
for bar, r2 in zip(bars, r2s):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{r2*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylim(0, max(r2s)*130)

for ax_ in axes[:2]:
    ax_.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_.xaxis.set_major_locator(mdates.YearLocator())
    ax_.set_xlim(dates[0], dates[-1])

plt.tight_layout()
out_fig = os.path.join(OUT, 'garch_selection_oracle.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {out_fig}")

# Save vol series for reference
pd.DataFrame({
    'date': dates,
    'sigma_m_garch': sigma_m,
    'sigma_r_crosssect': sigma_r,
    'sigma_x_topic': sigma_x,
    'agg_sel_rate': agg_rate,
}).to_csv(os.path.join(OUT, 'garch_vol_comparison.csv'), index=False)
print("Saved garch_vol_comparison.csv")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"GARCH α+β (persistence) : {res.params['alpha[1]']+res.params['beta[1]']:.4f}")
print(f"Std ratio actual/GARCH  : {agg_rate.std()/garch_mean.std():.2f}x")
print(f"Std ratio actual/iid    : {agg_rate.std()/iid_mean.std():.2f}x")
print(f"ρ(actual, GARCH sim)    : {rho_g:+.4f}  p={p_g:.4f}")
print(f"R²: raw vol reg         : {R2_1:.4f}")
print(f"R²: GARCH vol reg       : {R2_2:.4f}")
print(f"R²: GARCH vol only      : {R2_3:.4f}")
