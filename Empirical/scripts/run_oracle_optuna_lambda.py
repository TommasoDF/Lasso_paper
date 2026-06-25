"""
Oracle test: retrieve lambda via new Optuna objective, then re-estimate.

Starting point: identical DGP to the infeasible oracle in Appendix A4.
    kappa_true = 0.30
    lambda_true = 0.75 * lambda_base  (6.625e-5)
    sigma = 0.001, K=187, s=252, n_lags=1, T=1878

Only change vs. the infeasible oracle: instead of plugging in the known
lambda_true, we run Optuna to retrieve lambda, then run stage-1 + stage-2.

New objective (no hard guards, calibrated to oracle scale):
    1/3 * sigmoid(kappa_tstat - 1.96)          # κ t-stat: higher better
  + 1/3 * exp(-((R2_IS - R2_IS_TARGET)         # S1 IS R²: positive but small
                / R2_IS_WIDTH)^2)               #   bell peaked at 15%
  + 1/3 * sigmoid(R2_IS_S2 / R2_S2_SCALE)     # S2 IS R²: positive, larger better
                                               # (IS used here because we know the DGP)

Calibration to oracle DGP (σ=0.001):
  R2_IS_TARGET = 0.15  (at λ_true: S1 IS R² ≈ 0.19)
  R2_IS_WIDTH  = 0.08
  R2_S2_SCALE  = 0.010 (at λ_true: S2 IS R² ≈ 0.025, sigmoid ≈ 0.92)

Lambda search range: [1e-5, 1e-3]  (covers lambda_base and surroundings)

Outputs
-------
- Results/Figures/oracle_optuna_lambda_results.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Lasso

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = next(
    p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "Empirical").exists() and (p / "Data").exists()
)
SCRIPTS_DIR = ROOT / "Empirical" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage1 import lasso_rolling_window, to_ar1_innovations, calculate_r_squared
from stage2 import estimate_kappa_curve_fit, compute_alm_returns

# ── DGP — identical to infeasible oracle in A4 ────────────────────────────────
KAPPA_TRUE   = 0.30
LAMBDA_BASE  = 8.833637112956498e-05
LAMBDA_TRUE  = 0.75 * LAMBDA_BASE        # = 6.625e-5
W_TRUE       = -np.log(KAPPA_TRUE) - np.exp(-10)
SIGMA        = 0.001
WINDOW_SIZE  = 252
N_LAGS       = 1
N_TOPICS     = 187
RANDOM_SEED  = 12

# ── Optuna ────────────────────────────────────────────────────────────────────
N_TRIALS    = 150
LAMBDA_LOW  = 1e-5    # well below lambda_base
LAMBDA_HIGH = 1e-3    # well above lambda_base
SEED        = 42


# ── Data ──────────────────────────────────────────────────────────────────────

def load_topics() -> np.ndarray:
    topics = pd.read_csv(
        ROOT / "Data" / "clean_data" / "final_macro_topic_features.csv"
    )
    topics = topics[[c for c in topics.columns if not any(ch.isdigit() for ch in c)]]
    topics.set_index("date", inplace=True)
    topics = to_ar1_innovations(topics)
    topics = topics.iloc[1:].apply(lambda col: winsorize(col, limits=[0.01, 0.01]))
    topics = topics.astype(float).dropna(axis=0, how="any")
    return topics.iloc[:, :N_TOPICS].to_numpy(dtype=float)


# ── DGP — same function as oracle_lambda_stage2_cv.py ────────────────────────

def simulate_returns(X: np.ndarray, lambda_lasso: float) -> np.ndarray:
    """Exact same DGP as the infeasible oracle (no tanh, same seed)."""
    T, K = X.shape
    rng  = np.random.default_rng(RANDOM_SEED)
    beta = np.zeros((T, K))
    r    = np.zeros(T)
    r[1 : WINDOW_SIZE + 1] = rng.normal(0.0, 1e-4, WINDOW_SIZE)

    for t in range(WINDOW_SIZE + 1, T):
        X_win = X[t - WINDOW_SIZE - 1 : t - 1]
        r_win = r[t - WINDOW_SIZE : t]
        mdl   = Lasso(alpha=lambda_lasso, fit_intercept=False,
                      random_state=42, max_iter=5000)
        mdl.fit(X_win, r_win)
        beta[t] = mdl.coef_
        f_tm1   = float(beta[t - 1] @ X[t - 1])
        f_t     = float(beta[t]     @ X[t])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r[t] = (np.log(1.0 - KAPPA_TRUE * np.exp(f_tm1))
                        - np.log(1.0 - KAPPA_TRUE * np.exp(f_t))
                        + rng.normal(0.0, SIGMA))
            except Exception:
                r[t] = rng.normal(0.0, SIGMA)
    return r


# ── New oracle objective ──────────────────────────────────────────────────────
# Calibrated to oracle DGP (σ=0.001). No hard guards.

R2_IS_TARGET  = 0.15   # bell peak for S1 IS R²
R2_IS_WIDTH   = 0.08   # bell width
R2_S2_SCALE   = 0.010  # sigmoid scale for S2 IS R² (in-sample, valid since DGP known)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def composite_score(sm: dict | None) -> float:
    if sm is None:
        return 0.0
    tstat     = float(sm.get("kappa_tstat",        np.nan))
    r2_is_s1  = float(sm.get("r2_insample_stage1", np.nan))
    r2_is_s2  = float(sm.get("r2_insample_stage2", np.nan))
    if not all(np.isfinite([tstat, r2_is_s1, r2_is_s2])):
        return 0.0
    t_term  = _sigmoid(tstat - 1.96)
    s1_term = np.exp(-((r2_is_s1 - R2_IS_TARGET) / R2_IS_WIDTH) ** 2)
    s2_term = _sigmoid(r2_is_s2 / R2_S2_SCALE)
    return (t_term + s1_term + s2_term) / 3.0


def run_estimation(X: np.ndarray, r_sim: np.ndarray, lam: float) -> dict | None:
    """
    Same pipeline as oracle_lambda_stage2_cv.py (lasso_rolling_window + NLS),
    which matches the infeasible oracle and recovers kappa correctly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s1 = lasso_rolling_window(
                X=pd.DataFrame(X), y=pd.Series(r_sim),
                window_size=WINDOW_SIZE, n_lags=N_LAGS,
                lambda_mode="fixed", fixed_lambda=lam,
                standardize=False, verbose=False,
            )
        except Exception:
            return None

    preds   = np.asarray(s1["predictions"], dtype=float)[:-2]
    y_valid = np.asarray(r_sim, dtype=float)[N_LAGS:]
    y_valid = y_valid[-len(preds) - 1 : -1]
    mask    = np.isfinite(preds) & np.isfinite(y_valid)
    if mask.sum() < 50:
        return None

    preds_m   = preds[mask]
    y_m       = y_valid[mask]
    s2_df     = pd.DataFrame({"vwretd": y_m, "predictions": preds_m})
    n_features = X.shape[1] * N_LAGS   # topics × lags
    sel_rate   = float(np.mean(s1["num_nonzero_coefficients"]) / n_features)
    r2_is_s1  = float(np.nanmean(s1["insample_r_squareds"]))
    r2_oos_s1 = float(calculate_r_squared(y_m - np.nanmean(y_m), preds_m))

    try:
        popt, pcov = estimate_kappa_curve_fit(s2_df)
        kappa, ic  = float(popt[0]), float(popt[1])
        se         = float(np.sqrt(np.diag(pcov))[0])
        tstat      = kappa / se if se > 1e-30 else np.nan
        alm_is     = compute_alm_returns(preds_m, kappa, ic)
        r2_is_s2   = float(calculate_r_squared(y_m[1:], alm_is))
        # OOS: 80/20 split
        n_test = int(len(s2_df) * 0.2)
        tr = s2_df.iloc[:-n_test]; te = s2_df.iloc[-n_test:]
        popt2, _ = estimate_kappa_curve_fit(tr)
        alm_oos  = compute_alm_returns(te["predictions"].to_numpy(),
                                       float(popt2[0]), float(popt2[1]))
        r2_oos_s2 = float(calculate_r_squared(te["vwretd"].to_numpy()[1:], alm_oos))
    except Exception:
        kappa = tstat = r2_is_s2 = r2_oos_s2 = np.nan

    return {
        "kappa_tstat":        tstat,
        "kappa":              kappa,
        "avg_selection_rate": sel_rate,
        "r2_insample_stage1": r2_is_s1,
        "r2_oos_stage1":      r2_oos_s1,
        "r2_insample_stage2": r2_is_s2,
        "r2_oos_stage2":      r2_oos_s2,
    }


# ── Optuna search ─────────────────────────────────────────────────────────────

def run_optuna(X: np.ndarray, r_sim: np.ndarray) -> tuple[float, float]:
    def objective(trial: optuna.Trial) -> float:
        lam = trial.suggest_float("lambda", LAMBDA_LOW, LAMBDA_HIGH, log=True)
        sm  = run_estimation(X, r_sim, lam)
        return float(composite_score(sm))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_lam   = float(study.best_trial.params["lambda"])
    best_score = float(study.best_trial.value)
    return best_lam, best_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading topics …")
    X = load_topics()
    print(f"  X shape: {X.shape}")

    print(f"\nGenerating oracle returns "
          f"(κ={KAPPA_TRUE}, λ_true={LAMBDA_TRUE:.4e}, σ={SIGMA}, s={WINDOW_SIZE}) …")
    r_sim = simulate_returns(X, LAMBDA_TRUE)
    sel_dgp = float(np.mean(
        np.array([Lasso(alpha=LAMBDA_TRUE, fit_intercept=False, random_state=42)
                  .fit(X[t-WINDOW_SIZE-1:t-1], r_sim[t-WINDOW_SIZE:t]).coef_
                  for t in range(WINDOW_SIZE+1, min(WINDOW_SIZE+51, len(r_sim)))]) != 0
    ))
    print(f"  DGP selection rate (first 50 windows): ~{sel_dgp:.4f}")
    print(f"  λ_base = {LAMBDA_BASE:.6e}  |  λ_true = {LAMBDA_TRUE:.6e}")

    # ── Optuna ────────────────────────────────────────────────────────────────
    print(f"\nRunning Optuna ({N_TRIALS} trials, "
          f"λ ∈ [{LAMBDA_LOW:.0e}, {LAMBDA_HIGH:.0e}]) …")
    best_lam, best_score = run_optuna(X, r_sim)
    print(f"\n  Optuna selected: λ = {best_lam:.6e}  "
          f"(score = {best_score:.4f}, "
          f"ratio vs true = {best_lam/LAMBDA_TRUE:.3f}×)")

    # ── Re-estimation at selected lambda ──────────────────────────────────────
    print(f"\nRe-estimating with Optuna-selected λ = {best_lam:.6e} …")
    sm = run_estimation(X, r_sim, best_lam)

    if sm is None:
        print("  Estimation failed.")
        return

    out = {
        "lambda_optuna":   best_lam,
        "lambda_true":     LAMBDA_TRUE,
        "lambda_base":     LAMBDA_BASE,
        "lambda_ratio":    best_lam / LAMBDA_TRUE,
        "optuna_score":    best_score,
        "kappa_true":      KAPPA_TRUE,
        "kappa_hat":       float(sm.get("kappa",              np.nan)),
        "kappa_abs_err":   float(abs(sm.get("kappa", np.nan) - KAPPA_TRUE)),
        "kappa_tstat":     float(sm.get("kappa_tstat",        np.nan)),
        "s1_is_r2":        float(sm.get("r2_insample_stage1", np.nan)),
        "s1_oos_r2":       float(sm.get("r2_oos_stage1",      np.nan)),
        "s2_is_r2":        float(sm.get("r2_insample_stage2", np.nan)),
        "s2_oos_r2":       float(sm.get("r2_oos_stage2",      np.nan)),
        "sel_rate":        float(sm.get("avg_selection_rate", np.nan)),
    }

    # ── Comparison table ──────────────────────────────────────────────────────
    # Infeasible oracle numbers from existing results file
    inf = {
        "kappa_hat": 0.3287, "kappa_abs_err": 0.0287, "kappa_tstat": 9.5626,
        "s1_is_r2": 0.1880,  "s1_oos_r2": 0.0032,
        "s2_is_r2": 0.0248,  "s2_oos_r2": 0.0265,
        "sel_rate": 0.1748,  "lambda": LAMBDA_TRUE,
    }

    print(f"\n{'Metric':<22}  {'Infeasible':>12}  {'Optuna':>12}")
    print("-" * 50)
    rows = [
        ("λ",          inf["lambda"],        out["lambda_optuna"]),
        ("κ̂",          inf["kappa_hat"],      out["kappa_hat"]),
        ("|κ err|",    inf["kappa_abs_err"],  out["kappa_abs_err"]),
        ("t-stat κ̂",  inf["kappa_tstat"],    out["kappa_tstat"]),
        ("sel rate",   inf["sel_rate"],       out["sel_rate"]),
        ("S1 IS R²",   inf["s1_is_r2"],       out["s1_is_r2"]),
        ("S1 OOS R²",  inf["s1_oos_r2"],      out["s1_oos_r2"]),
        ("S2 IS R²",   inf["s2_is_r2"],       out["s2_is_r2"]),
        ("S2 OOS R²",  inf["s2_oos_r2"],      out["s2_oos_r2"]),
    ]
    for label, iv, ov in rows:
        print(f"  {label:<20}  {iv:>12.4f}  {ov:>12.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "config": {
            "kappa_true": KAPPA_TRUE, "lambda_true": LAMBDA_TRUE,
            "lambda_base": LAMBDA_BASE, "sigma": SIGMA,
            "window_size": WINDOW_SIZE, "n_lags": N_LAGS, "n_topics": N_TOPICS,
            "n_trials": N_TRIALS, "lambda_low": LAMBDA_LOW, "lambda_high": LAMBDA_HIGH,
        },
        "infeasible_oracle": inf,
        "optuna_oracle": out,
    }
    out_path = ROOT / "Results" / "Figures" / "oracle_optuna_lambda_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
