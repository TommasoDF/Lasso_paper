"""
Lambda selection via stage-2 OOS R² on real data, then full cross-sectional estimation.

Setup
-----
- Window size: 252 (fixed, 1 calendar year)
- n_lags: 1 (fixed)
- Lambda grid: glmnet-style, anchored at the data-driven λ_max (median across a sample
  of tuning stocks/windows of max|X'y|/n), then log-spaced down to λ_max * LAMBDA_EPS.
  λ_max is computed sequentially in the main process (pure numpy, no joblib).
- Each lambda is scored on N_EVAL_STOCKS randomly drawn tuning stocks using
  estimate_single_config_fast (the optimised estimator). Stage-2 OOS R² is the objective.
- Tuning/estimation split: same 20/80 cross-sectional split as the baseline
  (seed=42, min_obs=501, same eligible-stock filter).

Outputs
-------
- Data/clean_data/best_hyperparameters_stage2cv.txt   — selected hyperparameters
- Results/Estimation/Cross_Sectional_stage2cv/         — betas.h5 + summary.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

ROOT = next(
    p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "Empirical").exists() and (p / "Data").exists()
)
SCRIPTS_DIR = ROOT / "Empirical" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage1 import to_ar1_innovations, calculate_r_squared
from grid_search import estimate_single_config_fast
from cross_sectional_estimation import run_estimations, save_outputs

# ── Fixed settings ────────────────────────────────────────────────────────────
WINDOW_SIZE          = 252
N_LAGS               = 1
SEED                 = 42
TUNE_FRAC            = 0.20
WINDOW_HIGH          = 400     # for eligible-stock filter only (matches baseline)
MIN_OBS              = WINDOW_HIGH + N_LAGS + 100   # = 501

# glmnet-style grid
N_LAMBDA             = 20      # number of grid points
LAMBDA_EPS           = 1e-3    # λ_min = λ_max * LAMBDA_EPS
N_LAMBDAMAX_STOCKS   = 50      # stocks sampled to estimate λ_max (sequential)
N_LAMBDAMAX_WINDOWS  = 10      # windows sampled per stock for λ_max

# Scoring: subsample of tuning stocks per lambda (fast, mirrors Optuna baseline)
N_EVAL_STOCKS        = 30      # stocks drawn per lambda evaluation

N_JOBS               = -1      # parallelism for per-stock scoring


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    X_raw = pd.read_csv(
        ROOT / "Data" / "clean_data" / "final_macro_topic_features.csv",
        index_col=0, parse_dates=True,
    )
    R_raw = pd.read_csv(
        ROOT / "Data" / "data_raw" / "cross_sectional_returns.csv",
        index_col=0, parse_dates=True,
    )
    common = X_raw.index.intersection(R_raw.index)
    X_raw  = X_raw.loc[common]
    R_raw  = R_raw.loc[common]

    X_innov = to_ar1_innovations(X_raw)
    X_innov = X_innov.iloc[1:].apply(lambda col: winsorize(col, limits=[0.01, 0.01]))
    X_innov = X_innov.astype(float).dropna(axis=0, how="any")

    R_log = np.log(1 + R_raw)
    R_log.index = pd.to_datetime(R_log.index)

    common2 = X_innov.index.intersection(R_log.index)
    return X_innov.loc[common2], R_log.loc[common2]


# ── Stock split ───────────────────────────────────────────────────────────────

def make_stock_split(R: pd.DataFrame) -> tuple[list[str], list[str]]:
    eligible = [st for st in R.columns if R[st].dropna().shape[0] >= MIN_OBS]
    rng      = np.random.default_rng(SEED)
    perm     = list(rng.permutation(len(eligible)))
    eligible = [eligible[i] for i in perm]
    n_tune   = max(1, int(len(eligible) * TUNE_FRAC))
    return eligible[:n_tune], eligible[n_tune:]


# ── λ_max estimation — sequential, pure numpy ────────────────────────────────

def _lmax_one_stock(X_np: np.ndarray, r_np: np.ndarray,
                    rng: np.random.Generator) -> float:
    """
    λ_max = median over sampled windows of max_j |X_j' y| / n.
    With N_LAGS=1: lag-1 predictor matrix is X[:-1], target is r[1:].
    X is standardised within each window (sklearn Lasso convention).
    Pure numpy — no pandas, no joblib.
    """
    X_lag  = X_np[:-1]
    y      = r_np[1:]
    n_wins = len(y) - WINDOW_SIZE
    if n_wins <= 0:
        return np.nan

    starts = rng.choice(n_wins, size=min(N_LAMBDAMAX_WINDOWS, n_wins), replace=False)
    scaler = StandardScaler()
    vals   = []
    for t in starts:
        X_win = X_lag[t : t + WINDOW_SIZE]
        y_win = y[t : t + WINDOW_SIZE]
        if not (np.all(np.isfinite(X_win)) and np.all(np.isfinite(y_win))):
            continue
        X_s = scaler.fit_transform(X_win)
        y_c = y_win - y_win.mean()
        vals.append(float(np.max(np.abs(X_s.T @ y_c)) / WINDOW_SIZE))

    return float(np.median(vals)) if vals else np.nan


def compute_global_lambda_max(tune_stocks: list[str],
                              X: pd.DataFrame,
                              R: pd.DataFrame) -> float:
    """Median λ_max over N_LAMBDAMAX_STOCKS tuning stocks, sequential in main process."""
    rng    = np.random.default_rng(SEED + 7)
    sample = list(rng.choice(tune_stocks,
                             size=min(N_LAMBDAMAX_STOCKS, len(tune_stocks)),
                             replace=False))
    X_np  = X.to_numpy(dtype=float)
    X_idx = X.index
    vals  = []
    for i, st in enumerate(sample):
        r_s    = R[st].dropna()
        common = X_idx.intersection(r_s.index)
        if len(common) < WINDOW_SIZE + 2:
            continue
        v = _lmax_one_stock(
            X_np[X_idx.get_indexer(common)],
            r_s.loc[common].to_numpy(dtype=float),
            np.random.default_rng(SEED + 100 + i),
        )
        if np.isfinite(v):
            vals.append(v)

    if not vals:
        raise RuntimeError("Could not estimate λ_max from any tuning stock.")
    lmax = float(np.median(vals))
    print(f"  λ_max (median over {len(vals)} stocks) = {lmax:.6e}")
    print(f"  λ_min = λ_max × {LAMBDA_EPS:.0e}            = {lmax * LAMBDA_EPS:.6e}")
    return lmax


# ── Per-stock scorer (uses fast estimator) ────────────────────────────────────

def _score_one_stock(stock: str, X: pd.DataFrame, R: pd.DataFrame,
                     lambda_val: float) -> float:
    """Stage-1 + stage-2 OOS R² for one stock via estimate_single_config_fast."""
    r       = R[stock].dropna()
    X_stock = X.loc[r.index].dropna(how="any")
    r       = r.loc[X_stock.index]
    if len(r) < WINDOW_SIZE + N_LAGS + 50:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = estimate_single_config_fast(
                X=X_stock, y=r,
                window_size=WINDOW_SIZE, n_lags=N_LAGS,
                lambda_val=lambda_val,
                standardize=True, verbose=False,
                return_details=False,
            )
            sm = res.get("summary") or {}
            v  = sm.get("r2_oos_stage2", np.nan)
            return float(v) if np.isfinite(v) else np.nan
        except Exception:
            return np.nan


# ── Lambda grid search ────────────────────────────────────────────────────────

def select_lambda(tune_stocks: list[str], X: pd.DataFrame, R: pd.DataFrame) -> float:
    # Step 1: data-driven anchor (sequential, main process)
    print(f"\nEstimating λ_max from up to {N_LAMBDAMAX_STOCKS} tuning stocks …")
    lmax = compute_global_lambda_max(tune_stocks, X, R)

    # Step 2: log-spaced grid (glmnet convention)
    lambda_grid = np.exp(
        np.linspace(np.log(lmax), np.log(lmax * LAMBDA_EPS), N_LAMBDA)
    )

    print(f"\nGrid search — {N_EVAL_STOCKS} sampled tuning stocks per lambda, "
          f"{N_LAMBDA} log-spaced values (window={WINDOW_SIZE}, n_lags={N_LAGS})")
    print(f"{'#':>3}  {'λ':>12}  {'λ/λ_max':>8}  {'mean S2 OOS R²':>16}")
    print("-" * 46)

    best_lambda, best_score = lambda_grid[0], -np.inf
    rng = np.random.default_rng(SEED + 99)

    for i, lam in enumerate(lambda_grid):
        # fresh random subsample per lambda point
        sample = list(rng.choice(tune_stocks,
                                 size=min(N_EVAL_STOCKS, len(tune_stocks)),
                                 replace=False))
        scores  = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_score_one_stock)(st, X, R, lam) for st in sample
        )
        valid   = [s for s in scores if np.isfinite(s)]
        mean_r2 = float(np.mean(valid)) if valid else np.nan
        marker  = " ←" if (np.isfinite(mean_r2) and mean_r2 > best_score) else ""
        print(f"{i+1:>3}  {lam:>12.4e}  {lam/lmax:>8.4f}  {mean_r2:>16.6f}{marker}")
        if np.isfinite(mean_r2) and mean_r2 > best_score:
            best_score, best_lambda = mean_r2, lam

    print(f"\nSelected: λ = {best_lambda:.6e}  "
          f"(λ/λ_max = {best_lambda/lmax:.4f}, mean S2 OOS R² = {best_score:.6f})")
    return best_lambda


# ── Save hyperparameters ──────────────────────────────────────────────────────

def save_hyperparameters(lambda_val: float, n_tune: int, n_estim: int) -> Path:
    out = ROOT / "Data" / "clean_data" / "best_hyperparameters_stage2cv.txt"
    with open(out, "w") as f:
        f.write(f"window_size: {WINDOW_SIZE}\n")
        f.write(f"n_lags: {N_LAGS}\n")
        f.write(f"lambda: {lambda_val}\n")
        f.write(f"tune_frac: {TUNE_FRAC}\n")
        f.write(f"n_tune_stocks: {n_tune}\n")
        f.write(f"n_estim_stocks: {n_estim}\n")
        f.write(f"selection_objective: stage2_oos_r2\n")
        f.write(f"n_lambda_grid: {N_LAMBDA}\n")
        f.write(f"lambda_eps: {LAMBDA_EPS}\n")
        f.write(f"n_eval_stocks_per_lambda: {N_EVAL_STOCKS}\n")
    print(f"Saved → {out}")
    return out


# ── Full cross-sectional estimation ──────────────────────────────────────────

def run_full_estimation(estim_stocks: list[str], X: pd.DataFrame,
                        R: pd.DataFrame, lambda_val: float) -> None:
    out_dir = ROOT / "Results" / "Estimation" / "Cross_Sectional_stage2cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFull estimation — {len(estim_stocks)} stocks → {out_dir}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (tensor_betas, mat_r2_in, mat_preds, mat_targets,
         mat_lasso_intercepts, mat_stage2_preds, mat_stage2_r2_in,
         stocks, topics, date_strs, summary_df) = run_estimations(
            R[estim_stocks], X, WINDOW_SIZE, N_LAGS, lambda_val, n_jobs=N_JOBS,
        )
    save_outputs(
        tensor_betas, mat_r2_in, mat_preds, mat_targets,
        mat_lasso_intercepts, mat_stage2_preds, mat_stage2_r2_in,
        stocks, topics, date_strs, summary_df, str(out_dir),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    X, R = load_data()
    print(f"  Features: {X.shape}  |  Returns: {R.shape}")

    tune_stocks, estim_stocks = make_stock_split(R)
    print(f"  Tuning: {len(tune_stocks)} stocks  |  Estimation: {len(estim_stocks)} stocks")

    best_lambda = select_lambda(tune_stocks, X, R)
    save_hyperparameters(best_lambda, len(tune_stocks), len(estim_stocks))
    run_full_estimation(estim_stocks, X, R, best_lambda)
    print("\nDone.")


if __name__ == "__main__":
    main()
