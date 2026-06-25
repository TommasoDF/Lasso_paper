"""
Oracle test with kappa_true = 0.50 and window = 252.

Two exercises:
1. Infeasible oracle — simulate returns with known lambda_true, re-estimate kappa.
2. Stage-2 CV lambda selection — tune lambda by maximising stage-2 OOS R² on oracle data.

Lambda is calibrated from the stationary distribution of null OLS coefficients:
    sd_beta = sigma_d * sqrt((1-kappa)/2) / (sigma_x * sqrt(s))
    lambda_beta = z_{0.99} * sd_beta          [target false-sel prob = 2%]
    lambda_sklearn = lambda_beta * d_avg       [sklearn Lasso convention]

Outputs
-------
- Results/Figures/oracle_kappa05_infeasible_results.json
- Results/Figures/oracle_kappa05_stage2cv_results.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Lasso

ROOT = next(
    p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "Empirical").exists() and (p / "Data").exists()
)
SCRIPTS_DIR = ROOT / "Empirical" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage1 import to_ar1_innovations, calculate_r_squared
from stage2 import estimate_kappa_curve_fit, compute_alm_returns
from grid_search import estimate_single_config_fast

# ── Structural parameters ─────────────────────────────────────────────────────
KAPPA_TRUE   = 0.50
W_TRUE       = -np.log(KAPPA_TRUE) - np.exp(-10)   # ≈ 0.6931
SIGMA        = 0.001
WINDOW_SIZE  = 252
N_LAGS       = 1
RANDOM_SEED  = 12
N_TOPICS     = 187

TARGET_SEL_PROB = 0.02   # false-selection probability for lambda calibration
LAMBDA_MULT_TRUE = 0.75  # lambda_true = 0.75 * lambda_base (as in current oracle)

# Stage-2 CV grid multipliers
LAMBDA_MULTS = [0.35, 0.50, 0.75, 1.00, 1.25, 1.50]
CV_FOLDS     = 5
CV_MIN_TRAIN = 120
CV_VAL_FRAC  = 0.10


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


# ── Lambda calibration ────────────────────────────────────────────────────────

def calibrate_lambda(X: np.ndarray, kappa: float, sigma_d: float,
                     window: int) -> tuple[float, float]:
    """
    Returns (lambda_base, lambda_sklearn_true) where:
      lambda_base      = theoretically calibrated value (target 2% false-sel)
      lambda_sklearn_true = 0.75 * lambda_base  (DGP value used in simulation)
    """
    sigma_x  = float(np.sqrt(np.mean(X.var(axis=0))))
    sd_beta  = sigma_d * np.sqrt((1.0 - kappa) / 2.0) / (sigma_x * np.sqrt(window))
    z_crit   = norm.ppf(1 - TARGET_SEL_PROB / 2)
    lam_beta = z_crit * sd_beta
    d_avg    = float(np.mean(X ** 2))
    lam_base = lam_beta * d_avg
    lam_true = LAMBDA_MULT_TRUE * lam_base

    print(f"  sigma_x      = {sigma_x:.6f}")
    print(f"  sd_beta      = {sd_beta:.4e}")
    print(f"  lambda_base  = {lam_base:.6e}  (target 2% false-sel)")
    print(f"  lambda_true  = {lam_true:.6e}  ({LAMBDA_MULT_TRUE} × lambda_base)")
    return lam_base, lam_true


# ── DGP simulation ────────────────────────────────────────────────────────────

def simulate_returns(X: np.ndarray, kappa: float, w: float,
                     sigma: float, lambda_lasso: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate oracle returns using rolling offline LASSO DGP."""
    T, K = X.shape
    rng   = np.random.default_rng(RANDOM_SEED)
    beta  = np.zeros((T, K))
    r     = np.zeros(T)
    r[1 : WINDOW_SIZE + 1] = rng.normal(0, 1e-4, WINDOW_SIZE)

    for t in range(WINDOW_SIZE + 1, T):
        X_win = X[t - WINDOW_SIZE - 1 : t - 1]
        r_win = r[t - WINDOW_SIZE : t]
        mdl   = Lasso(alpha=lambda_lasso, fit_intercept=False,
                      random_state=42, max_iter=5000)
        mdl.fit(X_win, r_win)
        beta[t] = mdl.coef_

        f_tm1 = float(beta[t - 1] @ X[t - 1])
        f_t   = float(beta[t]     @ X[t])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r[t] = (np.log(1 - kappa * np.exp(f_tm1))
                        - np.log(1 - kappa * np.exp(f_t))
                        + rng.normal(0, sigma))
            except Exception:
                r[t] = rng.normal(0, sigma)

    return beta, r


# ── Infeasible oracle ─────────────────────────────────────────────────────────

def run_infeasible_oracle(X: np.ndarray, r_sim: np.ndarray,
                          lambda_val: float) -> dict:
    """Re-estimate model on simulated returns with known lambda."""
    X_df = pd.DataFrame(X)
    r_s  = pd.Series(r_sim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = estimate_single_config_fast(
            X=X_df, y=r_s,
            window_size=WINDOW_SIZE, n_lags=N_LAGS,
            lambda_val=lambda_val,
            standardize=False, verbose=False,
            return_details=False,
        )
    sm = res.get("summary", {})
    return {
        "true_kappa":        KAPPA_TRUE,
        "est_kappa":         float(sm.get("kappa",             np.nan)),
        "kappa_abs_err":     float(abs(sm.get("kappa", np.nan) - KAPPA_TRUE)),
        "kappa_tstat":       float(sm.get("kappa_tstat",       np.nan)),
        "stage1_is_r2":      float(sm.get("r2_insample_stage1",np.nan)),
        "stage1_oos_r2":     float(sm.get("r2_oos_stage1",     np.nan)),
        "stage2_is_r2":      float(sm.get("r2_insample_stage2",np.nan)),
        "stage2_oos_r2":     float(sm.get("r2_oos_stage2",     np.nan)),
        "sel_rate":          float(sm.get("avg_selection_rate", np.nan)),
    }


# ── Stage-2 CV scorer ─────────────────────────────────────────────────────────

def _stage2_cv_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    mask    = np.isfinite(preds) & np.isfinite(targets)
    preds   = preds[mask]; targets = targets[mask]
    n       = len(preds)
    val_sz  = max(40, int(CV_VAL_FRAC * n))
    last_te = n - val_sz
    if last_te <= CV_MIN_TRAIN:
        return np.nan

    ends    = np.unique(np.linspace(CV_MIN_TRAIN, last_te, CV_FOLDS, dtype=int))
    fold_r2 = []
    for te in ends:
        tr_df = pd.DataFrame({"vwretd": targets[:te], "predictions": preds[:te]})
        va_p  = preds[te : te + val_sz]
        va_t  = targets[te : te + val_sz]
        if len(va_p) < 2:
            continue
        try:
            popt, _ = estimate_kappa_curve_fit(tr_df)
            yhat    = compute_alm_returns(va_p, float(popt[0]), float(popt[1]))
            fold_r2.append(float(calculate_r_squared(va_t[1:], yhat)))
        except Exception:
            pass
    return float(np.nanmean(fold_r2)) if fold_r2 else np.nan


def score_lambda_on_oracle(X: np.ndarray, r_sim: np.ndarray,
                           lambda_val: float) -> dict:
    """Run stage-1 LASSO + stage-2 CV on oracle data at a given lambda."""
    X_df = pd.DataFrame(X)
    r_s  = pd.Series(r_sim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res  = estimate_single_config_fast(
                X=X_df, y=r_s,
                window_size=WINDOW_SIZE, n_lags=N_LAGS,
                lambda_val=lambda_val,
                standardize=False, verbose=False,
                return_details=False,
            )
            sm   = res.get("summary", {})
            preds   = np.asarray(res.get("preds",   getattr(res, "preds",   [])), dtype=float)
            targets = np.asarray(res.get("targets", getattr(res, "targets", [])), dtype=float)
        except Exception:
            return {"lambda": lambda_val, "cv_r2": np.nan, "kappa": np.nan,
                    "kappa_tstat": np.nan, "sel_rate": np.nan,
                    "s2_is_r2": np.nan, "s2_oos_r2": np.nan}

    return {
        "lambda":      lambda_val,
        "kappa":       float(sm.get("kappa",              np.nan)),
        "kappa_tstat": float(sm.get("kappa_tstat",        np.nan)),
        "sel_rate":    float(sm.get("avg_selection_rate", np.nan)),
        "s2_is_r2":    float(sm.get("r2_insample_stage2", np.nan)),
        "s2_oos_r2":   float(sm.get("r2_oos_stage2",      np.nan)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = ROOT / "Results" / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading topics …")
    X = load_topics()
    print(f"  X shape: {X.shape}")

    print(f"\nCalibrating lambda for kappa={KAPPA_TRUE}, window={WINDOW_SIZE} …")
    lam_base, lam_true = calibrate_lambda(X, KAPPA_TRUE, SIGMA, WINDOW_SIZE)

    print(f"\nSimulating oracle returns (kappa={KAPPA_TRUE}, lambda_true={lam_true:.4e}) …")
    beta_sim, r_sim = simulate_returns(X, KAPPA_TRUE, W_TRUE, SIGMA, lam_true)
    sel_sim = float(np.mean(beta_sim[WINDOW_SIZE + 1:] != 0))
    print(f"  Simulated T={len(r_sim)}, avg selection rate = {sel_sim:.4f}")

    # ── 1. Infeasible oracle ──────────────────────────────────────────────────
    print("\n--- Infeasible oracle (known lambda_true) ---")
    inf_res = run_infeasible_oracle(X, r_sim, lam_true)
    print(f"  True κ         = {inf_res['true_kappa']:.4f}")
    print(f"  Est  κ̂         = {inf_res['est_kappa']:.4f}  (|error| = {inf_res['kappa_abs_err']:.4f})")
    print(f"  κ̂ t-stat       = {inf_res['kappa_tstat']:.4f}")
    print(f"  S1 IS  R²      = {inf_res['stage1_is_r2']:.4f}")
    print(f"  S1 OOS R²      = {inf_res['stage1_oos_r2']:.4f}")
    print(f"  S2 IS  R²      = {inf_res['stage2_is_r2']:.4f}")
    print(f"  S2 OOS R²      = {inf_res['stage2_oos_r2']:.4f}")
    print(f"  Sel rate       = {inf_res['sel_rate']:.4f}")

    # ── 2. Stage-2 CV lambda selection ───────────────────────────────────────
    print("\n--- Stage-2 CV lambda grid ---")
    lambda_grid = [m * lam_base for m in LAMBDA_MULTS]
    print(f"  {'mult':>6}  {'lambda':>12}  {'S2 OOS R²':>12}  {'κ̂':>8}  {'sel':>6}")
    print("  " + "-" * 52)

    cv_rows = []
    best_lam, best_r2 = lambda_grid[0], -np.inf
    for mult, lam in zip(LAMBDA_MULTS, lambda_grid):
        row = score_lambda_on_oracle(X, r_sim, lam)
        r2  = row["s2_oos_r2"]
        marker = " ←" if (np.isfinite(r2) and r2 > best_r2) else ""
        print(f"  {mult:>6.2f}  {lam:>12.4e}  {r2:>12.6f}  "
              f"{row['kappa']:>8.4f}  {row['sel_rate']:>6.4f}{marker}")
        cv_rows.append({"lambda_mult": mult, **row})
        if np.isfinite(r2) and r2 > best_r2:
            best_r2, best_lam = r2, lam

    print(f"\n  CV-selected lambda = {best_lam:.6e}  (S2 OOS R² = {best_r2:.6f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "config": {
            "kappa_true": KAPPA_TRUE,
            "w_true": W_TRUE,
            "sigma": SIGMA,
            "window_size": WINDOW_SIZE,
            "n_lags": N_LAGS,
            "n_topics": N_TOPICS,
            "lambda_base": lam_base,
            "lambda_true": lam_true,
            "lambda_mult_true": LAMBDA_MULT_TRUE,
        },
        "infeasible_oracle": inf_res,
        "stage2_cv": {
            "best_lambda": best_lam,
            "best_s2_oos_r2": best_r2,
            "grid": cv_rows,
        },
    }

    out_inf = out_dir / "oracle_kappa05_infeasible_results.json"
    out_cv  = out_dir / "oracle_kappa05_stage2cv_results.json"
    with open(out_inf, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_cv, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved → {out_inf}")
    print("Done.")


if __name__ == "__main__":
    main()
