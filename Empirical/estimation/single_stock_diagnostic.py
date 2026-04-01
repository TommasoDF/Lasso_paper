"""
Single-stock end-to-end diagnostic.

Checks:
  1. Data sanity (features, returns, alignment)
  2. Lambda calibration in the actual feature space
  3. Hold-out hyperparameter search (grid, then refine)
  4. Full Stage-1 + Stage-2 on estimation period
  5. Comparison to oracle values

Run from repo root:
    .venv/bin/python3 Empirical/estimation/single_stock_diagnostic.py
"""

import sys, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import itertools

# ── repo root & paths ──────────────────────────────────────────────────────
ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / 'Empirical').exists() and (p / 'Data').exists()),
    None
)
assert ROOT is not None, "Run from repo root"
sys.path.insert(0, str(ROOT / 'Empirical' / 'scripts'))

from grid_search import estimate_single_config_fast

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HOLDOUT_START = '2010-01-19'
HOLDOUT_END   = '2012-12-31'
ESTIM_START   = '2013-01-02'
ESTIM_END     = '2017-06-30'

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING AND SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: DATA LOADING AND SANITY CHECKS")
print("=" * 65)

X_all = pd.read_csv(ROOT / 'Data' / 'clean_data' / 'final_macro_topic_features.csv',
                    index_col=0, parse_dates=True)
R_all = pd.read_csv(ROOT / 'Data' / 'data_raw' / 'cross_sectional_returns.csv',
                    index_col=0, parse_dates=True)

print(f"Features  : {X_all.shape}  [{X_all.index[0].date()} – {X_all.index[-1].date()}]")
print(f"Returns   : {R_all.shape}  [{R_all.index[0].date()} – {R_all.index[-1].date()}]")

# align
common_idx = X_all.index.intersection(R_all.index)
X_all = X_all.loc[common_idx]
R_all = R_all.loc[common_idx]
print(f"Common idx: {len(common_idx)} days")

# feature stats
print(f"\nFeature value ranges (all cols):")
print(f"  mean abs value : {X_all.abs().mean().mean():.4f}")
print(f"  std            : {X_all.std().mean():.4f}")
print(f"  max abs        : {X_all.abs().max().max():.4f}")
print(f"  => Features appear to be GLOBALLY STANDARDISED (std~1)")

# oracle used raw AR1 innovations (std~0.003). confirm the difference:
oracle_std_approx = 0.003
our_std = X_all.std().mean()
print(f"\n  Oracle feature std : ~{oracle_std_approx:.4f}")
print(f"  Our feature std    : ~{our_std:.4f}  (ratio {our_std/oracle_std_approx:.0f}x)")
print(f"  => Lambda must be ~{our_std/oracle_std_approx:.0f}x LARGER than oracle values")

# return stats
r_std = R_all.std().mean()
print(f"\nReturn std (cross-sectional mean) : {r_std:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LAMBDA CALIBRATION IN ACTUAL FEATURE SPACE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: LAMBDA CALIBRATION IN ACTUAL FEATURE SPACE")
print("=" * 65)

# empirical null-distribution calibration:
# with internal standardisation (std=1 per window) and returns std~r_std,
# null OLS beta has sd ~ r_std / sqrt(window)
# set lambda = z_{1-alpha/2} * r_std / sqrt(window)

print("\nWith internal standardisation per window:")
for window in [60, 100, 200, 300]:
    for target_sel in [0.02, 0.05, 0.10]:
        K = X_all.shape[1]
        z = norm.ppf(1 - target_sel / 2)
        lam = z * r_std / np.sqrt(window)
        print(f"  window={window:3d}, target_sel={target_sel:.2f} => lambda={lam:.4e}")
    print()

# empirical check: what lambda gives 5% selection on hold-out aggregate?
print("Empirical check — selection rate on hold-out EW aggregate:")
ho_mask = (common_idx >= HOLDOUT_START) & (common_idx <= HOLDOUT_END)
X_ho = X_all.loc[ho_mask]
R_ho = R_all.loc[ho_mask]
r_agg = R_ho.mean(axis=1).dropna()
X_agg = X_ho.loc[r_agg.index]

window_test = 100
for alpha in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = estimate_single_config_fast(
            X=X_agg, y=r_agg, window_size=window_test,
            n_lags=1, lambda_val=alpha, return_details=False
        )
    sm = res.get('summary', {})
    sel = sm.get('avg_selection_rate', float('nan'))
    kt  = sm.get('kappa_tstat', float('nan'))
    kv  = sm.get('kappa', float('nan'))
    print(f"  alpha={alpha:.0e}: sel_rate={sel:.3f}  kappa={kv:.4f}  kappa_tstat={kt:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PICK A STOCK AND SPLIT DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: SINGLE STOCK SELECTION AND DATA SPLIT")
print("=" * 65)

# pick a stock with full coverage
min_obs_total = 500
eligible = [st for st in R_all.columns if R_all[st].dropna().shape[0] >= min_obs_total]
rng = np.random.default_rng(SEED)
STOCK = str(rng.choice(eligible))
print(f"Selected stock: {STOCK}  (eligible pool: {len(eligible)})")

r_stock = R_all[STOCK].dropna()
X_stock = X_all.loc[r_stock.index]

print(f"Total obs: {len(r_stock)}  [{r_stock.index[0].date()} – {r_stock.index[-1].date()}]")
print(f"Return std: {r_stock.std():.6f}  mean: {r_stock.mean():.6f}")

# hold-out split
ho_idx = r_stock.index[(r_stock.index >= HOLDOUT_START) & (r_stock.index <= HOLDOUT_END)]
est_idx = r_stock.index[(r_stock.index >= ESTIM_START) & (r_stock.index <= ESTIM_END)]

r_ho   = r_stock.loc[ho_idx];   X_ho_st  = X_stock.loc[ho_idx]
r_est  = r_stock.loc[est_idx];  X_est_st = X_stock.loc[est_idx]

print(f"\nHold-out  : {len(r_ho)} days  ({HOLDOUT_START} – {HOLDOUT_END})")
print(f"Estimation: {len(r_est)} days  ({ESTIM_START} – {ESTIM_END})")

# time alignment check
assert r_ho.index.equals(X_ho_st.index),  "Hold-out index mismatch!"
assert r_est.index.equals(X_est_st.index), "Estimation index mismatch!"
print("Index alignment: OK")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: HOLD-OUT HYPERPARAMETER SEARCH (CMCE-aligned objective)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: HOLD-OUT HYPERPARAMETER SEARCH")
print("=" * 65)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def score_from_summary(sm):
    """CMCE-aligned objective (see manuscript eq. hyperpar_objective)."""
    if sm is None:
        return np.nan
    tstat    = float(sm.get('kappa_tstat',        np.nan))
    sel_rate = float(sm.get('avg_selection_rate', np.nan))
    kappa    = float(sm.get('kappa',              np.nan))
    if not all(np.isfinite([tstat, sel_rate, kappa])):
        return np.nan
    SEL_TARGET, SEL_SCALE = 0.05, 0.04
    return (0.6 * sigmoid(tstat - 1.96)
          + 0.3 * sigmoid((sel_rate - SEL_TARGET) / SEL_SCALE)
          + 0.1 * sigmoid((kappa - 0.5) / 0.3))

# grid over the CORRECT lambda range
lambdas = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
windows = [60, 100, 150, 200, 300]
N_LAGS  = 1

print(f"\nGrid: {len(lambdas)} lambdas × {len(windows)} windows = {len(lambdas)*len(windows)} configs")
print(f"Lambda range: [{min(lambdas):.0e}, {max(lambdas):.0e}]")
print(f"Window range: [{min(windows)}, {max(windows)}]")
print()

results = []
for lam, win in itertools.product(lambdas, windows):
    if len(r_ho) < win + N_LAGS + 50:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = estimate_single_config_fast(
            X=X_ho_st, y=r_ho, window_size=win,
            n_lags=N_LAGS, lambda_val=lam, return_details=False
        )
    sm = res.get('summary', {})
    sc = score_from_summary(sm)
    results.append({
        'lambda': lam, 'window': win, 'score': sc,
        'sel_rate': sm.get('avg_selection_rate', np.nan),
        'kappa':    sm.get('kappa',              np.nan),
        'kappa_t':  sm.get('kappa_tstat',        np.nan),
        'r2_in1':   sm.get('r2_insample_stage1', np.nan),
        'r2_oos1':  sm.get('r2_oos_stage1',      np.nan),
        'r2_oos2':  sm.get('r2_oos_stage2',      np.nan),
    })

df = pd.DataFrame(results).sort_values('score', ascending=False)
print("Full grid results (sorted by score):")
print(df.to_string(index=False, float_format='{:.4f}'.format))

best = df.iloc[0]
BEST_LAMBDA = float(best['lambda'])
BEST_WINDOW = int(best['window'])
print(f"\nBest: lambda={BEST_LAMBDA:.3e}  window={BEST_WINDOW}  score={best['score']:.4f}")
print(f"      sel_rate={best['sel_rate']:.4f}  kappa={best['kappa']:.4f}  kappa_t={best['kappa_t']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FULL ESTIMATION ON ESTIMATION PERIOD (2013–2017)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: FULL ESTIMATION ON ESTIMATION PERIOD")
print("=" * 65)
print(f"lambda={BEST_LAMBDA:.3e}  window={BEST_WINDOW}  n_lags={N_LAGS}  (frozen from hold-out)")

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    res_est = estimate_single_config_fast(
        X=X_est_st, y=r_est,
        window_size=BEST_WINDOW,
        n_lags=N_LAGS,
        lambda_val=BEST_LAMBDA,
        return_details=True
    )

sm_est  = res_est['summary']
det_est = res_est['details']

print(f"\nStage 1 — Rolling LASSO (belief index)")
print(f"  R²  in-sample  : {sm_est['r2_insample_stage1']:>10.4f}")
print(f"  R²  OOS        : {sm_est['r2_oos_stage1']:>10.4f}")
print(f"  Sel rate       : {sm_est['avg_selection_rate']:>10.4f}")
print(f"  Mean active    : {det_est['num_nonzero_coefficients'].mean():>10.1f}  / {X_est_st.shape[1]} features")
print()
print(f"Stage 2 — κ regression (ALM feedback)")
print(f"  R²  in-sample  : {sm_est['r2_insample_stage2']:>10.4f}")
print(f"  R²  OOS        : {sm_est['r2_oos_stage2']:>10.4f}")
print(f"  κ̂              : {sm_est['kappa']:>10.4f}")
print(f"  κ̂  t-stat      : {sm_est['kappa_tstat']:>10.3f}")
print(f"  intercept t    : {sm_est['intercept_tstat']:>10.3f}")
print()
print(f"Observations    : {sm_est['n_observations']}  |  windows: {sm_est['n_windows']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: COMPARISON TO ORACLE VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: COMPARISON TO ORACLE BENCHMARK")
print("=" * 65)
print()
print("Oracle test results (Appendix, notebook 8_oracle_prediction.ipynb):")
print("  True kappa          : 0.3746")
print("  Stage-1 OOS R²      : -0.009 to +0.016  (near zero — expected at CMCE)")
print("  Stage-2 OOS R²      : +0.003 to +0.018  (small positive)")
print("  Selection rate      : ~10%  (target was 2%; cross-correlations inflate)")
print()
print("This stock (estimation period):")
print(f"  Stage-1 OOS R²      : {sm_est['r2_oos_stage1']:>8.4f}")
print(f"  Stage-2 OOS R²      : {sm_est['r2_oos_stage2']:>8.4f}")
print(f"  κ̂                  : {sm_est['kappa']:>8.4f}")
print(f"  κ̂  t-stat          : {sm_est['kappa_tstat']:>8.3f}")
print(f"  Selection rate      : {sm_est['avg_selection_rate']:>8.4f}")
print()
if abs(sm_est['r2_oos_stage1']) < 0.05:
    print("  Stage-1 OOS R²: COMPARABLE to oracle (near zero — CMCE-consistent)")
else:
    print(f"  Stage-1 OOS R²: {'BETTER' if sm_est['r2_oos_stage1']>0 else 'WORSE'} than oracle")

if 0 < sm_est['r2_oos_stage2'] < 0.05:
    print("  Stage-2 OOS R²: COMPARABLE to oracle (small positive)")
elif sm_est['r2_oos_stage2'] <= 0:
    print("  Stage-2 OOS R²: negative — kappa may not be identified for this stock")
else:
    print("  Stage-2 OOS R²: LARGER than oracle — unusually strong signal")

if sm_est['kappa_tstat'] > 1.96:
    print(f"  kappa t-stat {sm_est['kappa_tstat']:.2f} > 1.96: kappa SIGNIFICANT")
else:
    print(f"  kappa t-stat {sm_est['kappa_tstat']:.2f} < 1.96: kappa not significant for this stock")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 7: DIAGNOSTIC PLOTS")
print("=" * 65)

det_est = det_est.set_index('date') if 'date' in det_est.columns else det_est

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

# Panel 1: returns
axes[0].plot(r_est.index, r_est.values, lw=0.7, color='steelblue')
axes[0].axhline(0, color='k', lw=0.5, ls='--')
axes[0].set_ylabel('Return')
axes[0].set_title(f'Single-stock diagnostic: stock {STOCK}  '
                  f'(λ={BEST_LAMBDA:.1e}, s={BEST_WINDOW})')

# Panel 2: belief index f_t
axes[1].plot(det_est.index, det_est['prediction'], lw=0.8, color='darkgreen')
axes[1].axhline(0, color='k', lw=0.5, ls='--')
axes[1].set_ylabel('f_t (belief index)')

# Panel 3: sparsity
axes[2].plot(det_est.index, det_est['num_nonzero_coefficients'],
             lw=0.7, color='firebrick')
axes[2].set_ylabel('# active β')

# Panel 4: rolling in-sample R²
axes[3].plot(det_est.index, det_est['lasso_r2_in'], lw=0.7, color='purple')
axes[3].axhline(0, color='k', lw=0.5, ls='--')
axes[3].set_ylabel('In-sample R²')
axes[3].set_xlabel('Date')

plt.tight_layout()
out_fig = ROOT / 'Empirical' / 'estimation' / 'single_stock_diagnostic.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"Plot saved to {out_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: SAVE CORRECTED LAMBDA RANGE NOTE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 8: CORRECTED SEARCH RANGE FOR HOLDOUT NOTEBOOK")
print("=" * 65)
print()
print("The holdout notebook was using LAMBDA_LOW=1e-8, LAMBDA_HIGH=1e-4.")
print("With internally-standardized features (std~1) and return std~0.009,")
print("the correct range is approximately [5e-4, 5e-2].")
print()
print("To fix: in find_hyperparameters_holdout.ipynb, change:")
print("  LAMBDA_LOW, LAMBDA_HIGH = 1e-8, 1e-4")
print("to:")
print("  LAMBDA_LOW, LAMBDA_HIGH = 5e-4, 5e-2")
print()
print("Done.")
