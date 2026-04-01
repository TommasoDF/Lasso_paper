"""
Vol-scaled lambda estimation.

Uses the same (s*, n_lags) from hyperparameter tuning but replaces the
single global lambda with a per-stock effective lambda:

    eff_lam_i = BEST_LAMBDA * (vol_i / TARGET_VOL)

where TARGET_VOL = 0.0175 (median daily return vol across 1,620 stocks).

This makes the LASSO selection rate approximately vol-invariant because
X is standardised inside the rolling window but y is not.

Results are saved to cross_sectional_results_volscaled.csv alongside
the baseline cross_sectional_results.csv for comparison.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, '../scripts')
from grid_search import estimate_single_config_fast

# ── Parameters ─────────────────────────────────────────────────────────────
BEST_LAMBDA  = 0.003505898780188227
BEST_WINDOW  = 231
N_LAGS       = 1
TARGET_VOL   = 0.0175   # median daily return vol

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data …")
X_all = pd.read_csv('../../Data/clean_data/final_macro_topic_features.csv',
                    index_col=0, parse_dates=True)
R_all = pd.read_csv('../../Data/data_raw/cross_sectional_returns.csv',
                    index_col=0, parse_dates=True)

common_idx = X_all.index.intersection(R_all.index)
X_all = X_all.loc[common_idx]
R_all = R_all.loc[common_idx]
print(f"Features: {X_all.shape}  |  Returns: {R_all.shape}")

# ── Load estimation stocks (same split as baseline) ─────────────────────────
estim_df = pd.read_csv('estim_stocks.csv')
ESTIM_STOCKS = list(estim_df['stock'].astype(str).str.split('.').str[0])
print(f"Estimation stocks: {len(ESTIM_STOCKS)}")

# Match column names in R_all
r_cols = {str(c).split('.')[0]: c for c in R_all.columns}
ESTIM_STOCKS = [s for s in ESTIM_STOCKS if s in r_cols]
print(f"Matched to R_all: {len(ESTIM_STOCKS)}")

# ── Per-stock estimation ────────────────────────────────────────────────────
def run_one_volscaled(st_str):
    col = r_cols[st_str]
    r   = R_all[col].dropna()
    if len(r) < BEST_WINDOW + N_LAGS + 100:
        return None
    stock_vol = float(r.std())
    eff_lam   = BEST_LAMBDA * (stock_vol / TARGET_VOL)
    X = X_all.loc[r.index]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            res = estimate_single_config_fast(
                X=X, y=r, window_size=BEST_WINDOW,
                n_lags=N_LAGS, lambda_val=eff_lam, return_details=False
            )
            sm = res.get('summary', None)
            if sm is not None:
                sm['stock']     = st_str
                sm['eff_lambda'] = eff_lam
                sm['stock_vol']  = stock_vol
            return sm
        except Exception:
            return None


print(f"\nRunning vol-scaled estimation on {len(ESTIM_STOCKS)} stocks …")
summaries = Parallel(n_jobs=-1, backend='loky')(
    delayed(run_one_volscaled)(st) for st in ESTIM_STOCKS
)

df = pd.DataFrame([sm for sm in summaries if sm is not None])

# ── Report ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VOL-SCALED LAMBDA — ESTIMATION RESULTS")
print(f"{'='*60}")
print(f"Valid stocks      : {len(df)}")

df_non0 = df[df['avg_selection_rate'] >= 0.003]
sig = df_non0['kappa_tstat'] > 1.96
df_sig = df_non0[sig]

print(f"Non-degenerate    : {len(df_non0)}  ({len(df_non0)/len(df)*100:.1f}%)")
print(f"Sig kappa (t>1.96): {sig.sum()}  ({sig.mean()*100:.1f}% of non-degen)")

print(f"\n--- Stage-1 (all stocks) ---")
for col, label in [('r2_insample_stage1','In-sample R2'),
                   ('r2_oos_stage1',     'OOS R2'),
                   ('avg_selection_rate','Selection rate')]:
    s = df[col].dropna()
    print(f"  {label:<20}: mean={s.mean():.4f}  median={s.median():.4f}  std={s.std():.4f}  "
          f"p05={s.quantile(.05):.4f}  p95={s.quantile(.95):.4f}")

print(f"\n--- Stage-2 (sig-kappa stocks, n={len(df_sig)}) ---")
for col, label in [('r2_insample_stage2','In-sample R2'),
                   ('r2_oos_stage2',     'OOS R2'),
                   ('kappa',             'kappa'),
                   ('kappa_tstat',       't-stat kappa')]:
    s = df_sig[col].dropna()
    print(f"  {label:<20}: mean={s.mean():.4f}  median={s.median():.4f}  "
          f"min={s.min():.4f}  max={s.max():.4f}  p05={s.quantile(.05):.4f}  p95={s.quantile(.95):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
out = 'cross_sectional_results_volscaled.csv'
df.to_csv(out, index=False)
print(f"\nSaved → {out}")
