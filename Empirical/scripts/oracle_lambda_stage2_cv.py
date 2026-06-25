"""
Oracle experiment: tune stage-1 lambda by maximizing stage-2 OOS R^2.

Setup:
- Window size and number of lags are fixed/known.
- Data are generated once from the oracle DGP using a known lambda and kappa.
- Candidate lambdas are evaluated by:
  1) running stage 1 with that fixed lambda,
  2) building stage-2 input (returns + stage-1 predictions),
  3) scoring anchored time-series CV OOS R^2 in stage 2 only.

Outputs:
- Results/Figures/oracle_lambda_stage2_cv_results.csv
- Results/Figures/oracle_lambda_stage2_cv_results.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Lasso


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for p in [current, *current.parents]:
        if (p / "Empirical").exists() and (p / "Data").exists():
            return p
    raise FileNotFoundError("Could not locate repository root.")


ROOT = _repo_root()
SCRIPTS_DIR = ROOT / "Empirical" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage1 import lasso_rolling_window, to_ar1_innovations, calculate_r_squared
from stage2 import estimate_kappa_curve_fit, compute_alm_returns, compute_stage2_r_squared


def multivariate_cgl_offline_lasso_with_x(
    X: np.ndarray,
    sigma: float,
    kappa: float,
    w: float,
    rolling_window_size: int,
    lambda_lasso: float,
    random_seed: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate oracle returns using the same stage-1 operator in the DGP."""
    X = np.asarray(X, dtype=float)
    t_len, n_topics = X.shape
    if rolling_window_size >= t_len:
        raise ValueError("rolling_window_size must be < number of observations")

    rng = np.random.default_rng(random_seed)
    beta = np.zeros((t_len, n_topics), dtype=float)
    returns = np.zeros(t_len, dtype=float)

    # Small warm-up noise to initialize rolling windows.
    returns[1 : rolling_window_size + 1] = rng.normal(0.0, 1e-4, rolling_window_size)

    for t in range(rolling_window_size + 1, t_len):
        x_win = X[t - rolling_window_size - 1 : t - 1, :]
        y_win = returns[t - rolling_window_size : t]

        mdl = Lasso(alpha=lambda_lasso, fit_intercept=False, random_state=42, max_iter=5000)
        mdl.fit(x_win, y_win)
        beta[t, :] = mdl.coef_

        pred_tm1 = float(beta[t - 1, :] @ X[t - 1, :])
        pred_t = float(beta[t, :] @ X[t, :])
        returns[t] = (
            np.log(1.0 - kappa * np.exp(pred_tm1))
            - np.log(1.0 - kappa * np.exp(pred_t))
            + rng.normal(0.0, sigma)
        )

    return beta, returns


def build_stage2_input(X: np.ndarray, y: np.ndarray, window_size: int, n_lags: int, lambda_val: float) -> pd.DataFrame:
    """Run stage 1 and align outputs exactly as in estimate_single_config."""
    s1 = lasso_rolling_window(
        X=X,
        y=y,
        window_size=window_size,
        n_lags=n_lags,
        lambda_mode="fixed",
        fixed_lambda=lambda_val,
        standardize=False,
        verbose=False,
    )

    preds = np.asarray(s1["predictions"], dtype=float)
    preds = preds[:-2]
    y_valid = np.asarray(y[-len(preds) - 1 : -1], dtype=float)

    mask = np.isfinite(preds) & np.isfinite(y_valid)
    return pd.DataFrame({"vwretd": y_valid[mask], "predictions": preds[mask]})


def stage2_oos_cv_r2(
    stage2_df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 120,
    val_size: int | None = None,
) -> dict:
    """Anchored time-series CV for stage 2 only, scored by OOS R^2."""
    s2 = stage2_df[["vwretd", "predictions"]].dropna().copy()
    n = len(s2)
    if n < min_train_size + 30:
        return {"cv_mean_r2": np.nan, "fold_r2": [], "n_obs": n}

    if val_size is None:
        val_size = max(40, int(0.10 * n))

    last_train_end = n - val_size
    if last_train_end <= min_train_size:
        return {"cv_mean_r2": np.nan, "fold_r2": [], "n_obs": n}

    train_ends = np.linspace(min_train_size, last_train_end, num=n_splits, dtype=int)
    train_ends = np.unique(train_ends)

    fold_r2 = []
    for te in train_ends:
        tr = s2.iloc[:te].copy()
        va = s2.iloc[te : te + val_size].copy()
        if len(va) < 2:
            continue

        try:
            popt, _ = estimate_kappa_curve_fit(tr)
            kappa_hat, intercept_hat = float(popt[0]), float(popt[1])
            yhat = compute_alm_returns(va["predictions"].to_numpy(), kappa_hat, intercept_hat)
            ytrue = va["vwretd"].to_numpy()[1:]
            r2 = float(calculate_r_squared(ytrue, yhat))
        except Exception:
            r2 = np.nan

        fold_r2.append(r2)

    mean_r2 = float(np.nanmean(fold_r2)) if len(fold_r2) else np.nan
    return {"cv_mean_r2": mean_r2, "fold_r2": fold_r2, "n_obs": n, "val_size": int(val_size)}


def load_topic_innovations(n_topics: int = 187) -> pd.DataFrame:
    data_path = ROOT / "Data" / "clean_data" / "final_macro_topic_features.csv"
    topics = pd.read_csv(data_path)
    topics = topics[[c for c in topics.columns if not any(ch.isdigit() for ch in c)]]
    topics = topics.set_index("date")
    topics = to_ar1_innovations(topics)
    topics = topics.iloc[1:]
    topics = topics.apply(lambda col: winsorize(col, limits=[0.01, 0.01]))
    topics = topics.astype(float).dropna(axis=0, how="any")
    return topics.iloc[:, :n_topics].copy()


def main() -> None:
    # Oracle setup from the corrected K=187, s=252 run.
    window_size = 252
    n_lags = 1
    lambda_base = 8.833637112956498e-05
    lambda_true = 0.75 * lambda_base
    kappa_true = 0.30
    sigma = 0.001
    w = -np.log(kappa_true) - np.exp(-10)

    lambda_grid_mult = [0.35, 0.5, 0.75, 1.0, 1.25, 1.5]
    lambda_grid = [m * lambda_base for m in lambda_grid_mult]

    topics = load_topic_innovations(n_topics=187)
    X = topics.to_numpy()

    _, y_sim = multivariate_cgl_offline_lasso_with_x(
        X=X,
        sigma=sigma,
        kappa=kappa_true,
        w=w,
        rolling_window_size=window_size,
        lambda_lasso=lambda_true,
        random_seed=12,
    )

    rows = []
    for mult, lam in zip(lambda_grid_mult, lambda_grid):
        s2_input = build_stage2_input(X, y_sim, window_size=window_size, n_lags=n_lags, lambda_val=float(lam))
        cv = stage2_oos_cv_r2(s2_input, n_splits=5, min_train_size=120)
        holdout = compute_stage2_r_squared(s2_input, min_train_size=100)
        rows.append(
            {
                "lambda_mult": float(mult),
                "lambda": float(lam),
                "stage2_cv_oos_r2_mean": float(cv["cv_mean_r2"]),
                "stage2_cv_oos_r2_fold_values": [float(v) if np.isfinite(v) else np.nan for v in cv["fold_r2"]],
                "stage2_holdout_oos_r2": float(holdout.get("r2_oos", np.nan)),
                "stage2_insample_r2": float(holdout.get("r2_insample", np.nan)),
                "kappa_full_sample": float(holdout.get("kappa", np.nan)),
            }
        )

    res_df = pd.DataFrame(rows).sort_values("stage2_cv_oos_r2_mean", ascending=False)
    best = res_df.iloc[0].to_dict()

    out_dir = ROOT / "Results" / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "oracle_lambda_stage2_cv_results.csv"
    out_json = out_dir / "oracle_lambda_stage2_cv_results.json"

    res_df.to_csv(out_csv, index=False)

    payload = {
        "config": {
            "window_size": window_size,
            "n_lags": n_lags,
            "kappa_true_data_generating": kappa_true,
            "lambda_true_data_generating": lambda_true,
            "lambda_base": lambda_base,
            "objective": "maximize_stage2_cv_oos_r2",
            "cv_scheme": "anchored_time_series_cv_stage2_only",
            "n_folds": 5,
        },
        "best_by_stage2_cv": {
            "lambda_mult": float(best["lambda_mult"]),
            "lambda": float(best["lambda"]),
            "stage2_cv_oos_r2_mean": float(best["stage2_cv_oos_r2_mean"]),
            "stage2_holdout_oos_r2": float(best["stage2_holdout_oos_r2"]),
        },
        "grid_results": rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print("\nTop rows by stage2 CV OOS R^2:")
    print(res_df[["lambda_mult", "lambda", "stage2_cv_oos_r2_mean", "stage2_holdout_oos_r2"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
