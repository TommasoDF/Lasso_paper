"""
Optuna lambda search with the original composite objective, window=252 and n_lags=1 fixed.

Objective (identical to find_hyperparameters_holdout.ipynb):
    0.7 * sigmoid(kappa_tstat - 1.96)
  + 0.2 * exp(-((sel_rate - 0.03) / 0.02)^2)    ← bell peak at 3% sparsity
  + 0.1 * sigmoid((kappa - 0.3) / 0.2)

Hard guards (return 0):
  - avg_selection_rate < 0.3%  → f_t degenerate
  - avg_selection_rate > 20%   → dense regime

Only lambda is tuned; window=252 and n_lags=1 are fixed.
Tuning/estimation split: same 20/80 split as baseline (seed=42).

Outputs
-------
- Data/clean_data/best_hyperparameters_w252.txt
- Results/Estimation/Cross_Sectional_w252/  (betas.h5, summary.csv)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed
from scipy.stats.mstats import winsorize

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = next(
    p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "Empirical").exists() and (p / "Data").exists()
)
SCRIPTS_DIR = ROOT / "Empirical" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage1 import to_ar1_innovations
from grid_search import estimate_single_config_fast
from cross_sectional_estimation import run_estimations, save_outputs

# ── Fixed settings ────────────────────────────────────────────────────────────
WINDOW_SIZE  = 252
N_LAGS       = 1
SEED         = 42
TUNE_FRAC    = 0.20
WINDOW_HIGH  = 400
MIN_OBS      = WINDOW_HIGH + N_LAGS + 100   # = 501

# Optuna (same as baseline notebook)
N_TRIALS     = 200
LAMBDA_LOW   = 1e-3
LAMBDA_HIGH  = 0.1
N_VALIDATE   = 30

N_JOBS       = -1


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
    X_raw, R_raw = X_raw.loc[common], R_raw.loc[common]

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


# ── Original composite objective ──────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def score_from_summary(sm: dict | None) -> float:
    if sm is None:
        return 0.0
    tstat    = float(sm.get("kappa_tstat",        np.nan))
    sel_rate = float(sm.get("avg_selection_rate", np.nan))
    kappa    = float(sm.get("kappa",              np.nan))
    if not all(np.isfinite([tstat, sel_rate, kappa])):
        return 0.0
    if sel_rate < 0.003:
        return 0.0
    if sel_rate > 0.20:
        return 0.0
    sel_reward   = np.exp(-((sel_rate - 0.03) / 0.02) ** 2)
    tstat_reward = _sigmoid(tstat - 1.96)
    kappa_reward = _sigmoid((kappa - 0.3) / 0.2)
    return 0.7 * tstat_reward + 0.2 * sel_reward + 0.1 * kappa_reward


def run_one(stock: str, X: pd.DataFrame, R: pd.DataFrame,
            lambda_val: float) -> dict | None:
    r       = R[stock].dropna()
    X_stock = X.loc[r.index].dropna(how="any")
    r       = r.loc[X_stock.index]
    if len(r) < WINDOW_SIZE + N_LAGS + 100:
        return None
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
            return res.get("summary")
        except Exception:
            return None


# ── Optuna search ─────────────────────────────────────────────────────────────

def run_optuna(tune_stocks: list[str],
               X: pd.DataFrame,
               R: pd.DataFrame) -> tuple[float, optuna.Study]:
    trial_rng    = np.random.default_rng(SEED + 1)
    trial_stocks = list(trial_rng.choice(tune_stocks, size=N_TRIALS, replace=True))

    def objective(trial: optuna.Trial) -> float:
        lam = trial.suggest_float("lambda", LAMBDA_LOW, LAMBDA_HIGH, log=True)
        sm  = run_one(trial_stocks[trial.number], X, R, lam)
        return float(score_from_summary(sm))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"Running {N_TRIALS} Optuna trials "
          f"(window={WINDOW_SIZE}, n_lags={N_LAGS}, "
          f"λ ∈ [{LAMBDA_LOW:.0e}, {LAMBDA_HIGH:.0e}]) …")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_lam   = float(study.best_trial.params["lambda"])
    best_score = float(study.best_trial.value)
    print(f"\nBest: λ = {best_lam:.6e},  score = {best_score:.4f}")
    return best_lam, study


def validate(best_lam: float, tune_stocks: list[str],
             X: pd.DataFrame, R: pd.DataFrame) -> None:
    rng    = np.random.default_rng(SEED + 999)
    sample = list(rng.choice(tune_stocks,
                             size=min(N_VALIDATE, len(tune_stocks)),
                             replace=False))
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_one)(st, X, R, best_lam) for st in sample
    )
    sms = [r for r in results if r is not None]
    df  = pd.DataFrame(sms)
    print(f"\nValidation on {len(df)} tuning stocks at λ = {best_lam:.6e}:")
    for col in ["r2_oos_stage2", "kappa_tstat", "kappa", "avg_selection_rate"]:
        if col in df:
            print(f"  {col:30s}  mean={df[col].mean():.4f}  median={df[col].median():.4f}")
    sig  = df["kappa_tstat"] > 1.96
    pct0 = (df["avg_selection_rate"] < 0.003).mean()
    print(f"  sig κ (t>1.96): {sig.sum()}/{len(df)}  ({100*sig.mean():.1f}%)")
    print(f"  % zero sel:     {100*pct0:.1f}%")


# ── Save hyperparameters ──────────────────────────────────────────────────────

def save_hyperparameters(lambda_val: float, score: float,
                         n_tune: int, n_estim: int) -> Path:
    out = ROOT / "Data" / "clean_data" / "best_hyperparameters_w252.txt"
    with open(out, "w") as f:
        f.write(f"window_size: {WINDOW_SIZE}\n")
        f.write(f"n_lags: {N_LAGS}\n")
        f.write(f"lambda: {lambda_val}\n")
        f.write(f"tune_frac: {TUNE_FRAC}\n")
        f.write(f"n_tune_stocks: {n_tune}\n")
        f.write(f"n_estim_stocks: {n_estim}\n")
        f.write(f"optuna_score: {score:.6f}\n")
        f.write(f"objective: composite_kappa_tstat+sparsity+kappa_interior\n")
        f.write(f"n_trials: {N_TRIALS}\n")
    print(f"Saved → {out}")
    return out


# ── Full cross-sectional estimation ──────────────────────────────────────────

def run_full_estimation(estim_stocks: list[str], X: pd.DataFrame,
                        R: pd.DataFrame, lambda_val: float) -> None:
    out_dir = ROOT / "Results" / "Estimation" / "Cross_Sectional_w252"
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

    best_lam, study = run_optuna(tune_stocks, X, R)
    validate(best_lam, tune_stocks, X, R)
    save_hyperparameters(best_lam, study.best_trial.value,
                         len(tune_stocks), len(estim_stocks))
    run_full_estimation(estim_stocks, X, R, best_lam)
    print("\nDone.")


if __name__ == "__main__":
    main()
