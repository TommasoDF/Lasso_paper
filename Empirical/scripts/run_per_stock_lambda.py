"""
Per-stock lambda selection, then full cross-sectional estimation.

For each stock:
  1. Grid-search N_LAMBDA log-spaced values using the composite objective.
  2. Apply that stock's best lambda in the full rolling-window estimation.

Composite objective (identical to global baseline):
    0.7 * sigmoid(kappa_tstat - 1.96)
  + 0.2 * exp(-((sel_rate - 0.03) / 0.02)^2)
  + 0.1 * sigmoid((kappa - 0.3) / 0.2)
Hard guards: sel_rate < 0.3% or > 20% → 0.
Fallback: stocks with score = 0 at all grid points use GLOBAL_LAMBDA.

Set N_PILOT = 100 for a prototype, -1 for the full 1,296-stock run.

Outputs
-------
- Results/Estimation/Cross_Sectional_per_stock_lambda/
    betas.h5, summary.csv, per_stock_lambdas.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats.mstats import winsorize
from tqdm.auto import tqdm

ROOT = next(
    p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "Empirical").exists() and (p / "Data").exists()
)
SCRIPTS_DIR = ROOT / "Empirical" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage1 import to_ar1_innovations, calculate_r_squared, create_lagged_features_with_name
from stage2 import compute_alm_returns
from grid_search import estimate_single_config_fast
from cross_sectional_estimation import _process_one_stock, _date_str, save_outputs

# ── Settings ──────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 252
N_LAGS        = 1
SEED          = 42
TUNE_FRAC     = 0.20
WINDOW_HIGH   = 400
MIN_OBS       = WINDOW_HIGH + N_LAGS + 100

GLOBAL_LAMBDA = 3.223568e-3
N_LAMBDA      = 15
LAMBDA_LOW    = 0.20 * GLOBAL_LAMBDA
LAMBDA_HIGH   = 5.00 * GLOBAL_LAMBDA

N_PILOT       = -1    # -1 = all estimation stocks; set to 100 for prototype
N_JOBS        = -1


# ── Data ──────────────────────────────────────────────────────────────────────

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
    R_log   = np.log(1 + R_raw)
    R_log.index = pd.to_datetime(R_log.index)
    common2 = X_innov.index.intersection(R_log.index)
    return X_innov.loc[common2], R_log.loc[common2]


def make_stock_split(R: pd.DataFrame) -> tuple[list[str], list[str]]:
    eligible = [st for st in R.columns if R[st].dropna().shape[0] >= MIN_OBS]
    rng      = np.random.default_rng(SEED)
    perm     = list(rng.permutation(len(eligible)))
    eligible = [eligible[i] for i in perm]
    n_tune   = max(1, int(len(eligible) * TUNE_FRAC))
    return eligible[:n_tune], eligible[n_tune:]


# ── Composite objective ───────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def composite_score(sm: dict | None) -> float:
    if sm is None:
        return 0.0
    tstat    = float(sm.get("kappa_tstat",        np.nan))
    sel_rate = float(sm.get("avg_selection_rate", np.nan))
    kappa    = float(sm.get("kappa",              np.nan))
    if not all(np.isfinite([tstat, sel_rate, kappa])):
        return 0.0
    if sel_rate < 0.003 or sel_rate > 0.20:
        return 0.0
    return (0.7 * _sigmoid(tstat - 1.96)
          + 0.2 * np.exp(-((sel_rate - 0.03) / 0.02) ** 2)
          + 0.1 * _sigmoid((kappa - 0.3) / 0.2))


# ── Per-stock lambda selection ────────────────────────────────────────────────

def select_lambda_for_stock(stock: str, X: pd.DataFrame,
                             R: pd.DataFrame, lambda_grid: np.ndarray) -> dict:
    r       = R[stock].dropna()
    X_stock = X.loc[r.index].dropna(how="any")
    r       = r.loc[X_stock.index]

    if len(r) < WINDOW_SIZE + N_LAGS + 100:
        return {"stock": stock, "best_lambda": GLOBAL_LAMBDA,
                "best_score": np.nan, "best_sel": np.nan,
                "best_kappa": np.nan, "best_tstat": np.nan,
                "fallback": True}

    best_lam, best_score, best_meta = GLOBAL_LAMBDA, -np.inf, {}
    for lam in lambda_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = estimate_single_config_fast(
                    X=X_stock, y=r,
                    window_size=WINDOW_SIZE, n_lags=N_LAGS,
                    lambda_val=float(lam),
                    standardize=True, verbose=False, return_details=False,
                )
                sm = res.get("summary")
                sc = composite_score(sm)
            except Exception:
                sc, sm = 0.0, {}
        if sc > best_score:
            best_score, best_lam, best_meta = sc, float(lam), sm or {}

    return {
        "stock":       stock,
        "best_lambda": best_lam,
        "best_score":  best_score,
        "best_sel":    float(best_meta.get("avg_selection_rate", np.nan)),
        "best_kappa":  float(best_meta.get("kappa",              np.nan)),
        "best_tstat":  float(best_meta.get("kappa_tstat",        np.nan)),
        "fallback":    best_score <= 0.0,
    }


# ── Full rolling estimation (per-stock lambda) ────────────────────────────────

def _process_one_stock_star(args: tuple) -> tuple:
    return _process_one_stock(*args)


def run_per_stock_estimations(
    stocks: list[str],
    X: pd.DataFrame,
    R: pd.DataFrame,
    stock_lambdas: dict[str, float],
) -> tuple:
    """Like cross_sectional_estimation.run_estimations but with per-stock lambda."""
    args_list = [
        (st, R, X, WINDOW_SIZE, N_LAGS, stock_lambdas.get(st, GLOBAL_LAMBDA))
        for st in stocks
    ]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    raw: list[tuple] = []
    workers = max(1, mp.cpu_count() - 1) if N_JOBS == -1 else max(1, N_JOBS)
    with ctx.Pool(processes=workers) as pool:
        for res in tqdm(
            pool.imap_unordered(_process_one_stock_star, args_list, chunksize=1),
            total=len(stocks), desc="Estimating per stock",
        ):
            raw.append(res)

    topics      = None
    all_dates   = set()
    success, failed = 0, []
    for r in raw:
        t, d, sm = r[1], r[9], r[10]
        if t is not None:
            success += 1
            topics = topics or t
            for di in d:
                all_dates.add(_date_str(di))
        else:
            failed.append((r[0], sm.get("fail_reason", "?")))

    print(f"\n--- Estimation Report ---")
    print(f"Successful: {success} / {len(stocks)}  |  Failed: {len(failed)}")

    if topics is None:
        raise RuntimeError("All stocks failed.")

    date_strs   = sorted(all_dates)
    date_to_idx = {ds: i for i, ds in enumerate(date_strs)}
    n_t, n_d    = len(topics), len(date_strs)

    result_by_stock = {r[0]: r for r in raw}
    tensor_betas       = np.full((len(stocks), n_t, n_d), np.nan, dtype=np.float32)
    mat_r2_in          = np.full((len(stocks), n_d), np.nan, dtype=np.float32)
    mat_preds          = np.full((len(stocks), n_d), np.nan, dtype=np.float32)
    mat_targets        = np.full((len(stocks), n_d), np.nan, dtype=np.float32)
    mat_intercepts     = np.full((len(stocks), n_d), np.nan, dtype=np.float32)
    mat_stage2_preds   = np.full((len(stocks), n_d), np.nan, dtype=np.float32)
    mat_stage2_r2_in   = np.full((len(stocks), n_d), np.nan, dtype=np.float32)
    summary_rows = []

    for i, st in enumerate(stocks):
        if st not in result_by_stock:
            continue
        _, _, betas, r2_in, preds, targets, ints, s2p, s2r, dates, sm = result_by_stock[st]
        if betas is not None:
            for j, dj in enumerate(dates):
                k = date_to_idx.get(_date_str(dj))
                if k is not None:
                    tensor_betas[i, :, k]     = betas[j]
                    mat_r2_in[i, k]           = r2_in[j]
                    mat_preds[i, k]           = preds[j]
                    mat_targets[i, k]         = targets[j]
                    mat_intercepts[i, k]      = ints[j]
                    mat_stage2_preds[i, k]    = s2p[j]
                    mat_stage2_r2_in[i, k]    = s2r[j]
        if sm:
            summary_rows.append(sm)

    summary_df = (pd.DataFrame(summary_rows).set_index("stock")
                  if summary_rows else pd.DataFrame())

    return (tensor_betas, mat_r2_in, mat_preds, mat_targets,
            mat_intercepts, mat_stage2_preds, mat_stage2_r2_in,
            stocks, topics, date_strs, summary_df)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    X, R = load_data()
    print(f"  Features: {X.shape}  |  Returns: {R.shape}")

    _, estim_stocks = make_stock_split(R)

    if N_PILOT > 0:
        rng   = np.random.default_rng(SEED + 77)
        pilot = list(rng.choice(estim_stocks,
                                size=min(N_PILOT, len(estim_stocks)),
                                replace=False))
        out_label = "pilot"
    else:
        pilot     = estim_stocks
        out_label = "full"

    print(f"  Stocks: {len(pilot)} ({'pilot' if N_PILOT > 0 else 'full estimation set'})")

    lambda_grid = np.exp(
        np.linspace(np.log(LAMBDA_LOW), np.log(LAMBDA_HIGH), N_LAMBDA)
    )
    print(f"\nLambda grid: {N_LAMBDA} values in "
          f"[{LAMBDA_LOW:.3e}, {LAMBDA_HIGH:.3e}]  "
          f"({LAMBDA_LOW/GLOBAL_LAMBDA:.2f}×–{LAMBDA_HIGH/GLOBAL_LAMBDA:.2f}× global)")

    # ── Step 1: per-stock lambda selection ────────────────────────────────────
    print(f"\nStep 1 — per-stock lambda selection ({len(pilot)} stocks × {N_LAMBDA} pts) …")
    t0 = time.time()
    lam_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(select_lambda_for_stock)(st, X, R, lambda_grid) for st in pilot
    )
    t_sel = time.time() - t0

    lam_df     = pd.DataFrame(lam_results).set_index("stock")
    fallback_n = lam_df["fallback"].sum()
    print(f"  Done in {t_sel:.1f}s  ({t_sel/len(pilot):.1f}s/stock)")
    print(f"  Fallback to global λ: {fallback_n} ({100*fallback_n/len(lam_df):.1f}%)")
    print(f"  Selected λ — mean={lam_df.best_lambda.mean():.4e}  "
          f"median={lam_df.best_lambda.median():.4e}  "
          f"std={lam_df.best_lambda.std():.4e}")
    print(f"  Composite score — mean={lam_df.best_score.mean():.4f}  "
          f"median={lam_df.best_score.median():.4f}")
    print(f"  Selection rate  — mean={lam_df.best_sel.mean():.4f}  "
          f"median={lam_df.best_sel.median():.4f}")

    # ── Step 2: full rolling estimation with per-stock lambdas ────────────────
    out_dir = ROOT / "Results" / "Estimation" / f"Cross_Sectional_per_stock_lambda"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 2 — full rolling estimation with per-stock lambdas → {out_dir}")
    stock_lambdas = lam_df["best_lambda"].to_dict()
    t1 = time.time()
    (tensor_betas, mat_r2_in, mat_preds, mat_targets,
     mat_intercepts, mat_stage2_preds, mat_stage2_r2_in,
     stocks_out, topics_out, date_strs, summary_df) = run_per_stock_estimations(
        pilot, X, R, stock_lambdas
    )
    t_est = time.time() - t1
    print(f"  Done in {t_est:.1f}s  ({t_est/len(pilot):.1f}s/stock)")

    save_outputs(
        tensor_betas, mat_r2_in, mat_preds, mat_targets,
        mat_intercepts, mat_stage2_preds, mat_stage2_r2_in,
        stocks_out, topics_out, date_strs, summary_df, str(out_dir),
    )
    lam_df.to_csv(out_dir / "per_stock_lambdas.csv")

    # ── Timing summary ────────────────────────────────────────────────────────
    n = len(pilot)
    total = t_sel + t_est
    print(f"\nTiming:")
    print(f"  Lambda selection: {t_sel:.0f}s  → ~{t_sel/n*1296/60:.0f} min for 1296 stocks")
    print(f"  Estimation:       {t_est:.0f}s  → ~{t_est/n*1296/60:.0f} min for 1296 stocks")
    print(f"  Total:            {total:.0f}s  → ~{total/n*1296/60:.0f} min for 1296 stocks")
    print(f"\nSaved → {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
