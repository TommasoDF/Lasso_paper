# cross_sectional_estimation_v1.py
#
# Simplified version with no checkpointing.
# Runs all stocks in one pass and saves:
#   betas.h5    — HDF5 tensor (n_stocks, n_topics, n_dates) float32, plus r2_in, predictions, targets, and stage 2 data
#   summary.csv — per-stock scalar metrics (R², kappa, return variance, …)
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import multiprocessing as mp
import random
import warnings
import sys
import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

from get_repo_root import get_repo_root

REPO_ROOT = get_repo_root()
SCRIPTS_DIR = REPO_ROOT / "scripts" # Hier liegen grid_search, stage1, etc.

# Füge den Skript-Ordner zum Suchpfad hinzu
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from grid_search import estimate_single_config_fast
from stage1 import create_lagged_features_with_name
from stage2 import compute_alm_returns

# ============================================================
# Preprocessing helpers
# ============================================================


def read_best_hyperparameters(path: str) -> Dict[str, float]:
    best: Dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.strip().split(": ")
            best[key] = float(value)
    return best


# ============================================================
# Beta extraction from details DataFrame
# ============================================================

_LAG1_COL_RE = re.compile(r"^Lasso_(.+)_lag_1$")


def _extract_details(
    details: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts betas, r2_in, preds, targets, intercepts, and dates."""
    topic_col: Dict[str, str] = {}
    for c in details.columns:
        m = _LAG1_COL_RE.match(str(c))
        if m:
            topic_col[m.group(1)] = c

    topics = sorted(topic_col.keys())

    if "date" in details.columns:
        valid = details["date"].notna()
        sub = details.loc[valid]
        dates = pd.to_datetime(sub["date"]).to_numpy()
    else:
        sub = details
        dates = np.arange(len(sub), dtype=np.int64)

    betas = sub[[topic_col[t] for t in topics]].to_numpy(dtype=np.float32)
    
    r2_in = sub['lasso_r2_in'].to_numpy(dtype=np.float32)
    preds = sub['prediction'].to_numpy(dtype=np.float32)
    targets = sub['target'].to_numpy(dtype=np.float32)
    intercepts = sub['lasso_intercept'].to_numpy(dtype=np.float32)
    
    return topics, betas, r2_in, preds, targets, intercepts, dates


def _date_str(d: Any) -> str:
    """Normalise any date-like value to a sortable 'YYYY-MM-DD' string."""
    return str(pd.Timestamp(d))[:10]


# ============================================================
# Per-stock estimation
# ============================================================

def _process_one_stock(
    stock: str,
    df_returns: pd.DataFrame,
    X_features: pd.DataFrame,
    window_size: int,
    n_lags: int,
    lambda_val: float,
) -> Tuple[
    str,
    Optional[List[str]],
    Optional[np.ndarray],  # betas
    Optional[np.ndarray],  # r2_in
    Optional[np.ndarray],  # preds
    Optional[np.ndarray],  # targets
    Optional[np.ndarray],  # lasso_intercepts
    Optional[np.ndarray],  # stage2_preds (OOS)
    Optional[np.ndarray],  # stage2_r2_in
    Optional[np.ndarray],  # dates
    Dict[str, Any],
]:
    y = df_returns[stock]
    common_index = X_features.index.intersection(y.index)
    summary: Dict[str, Any] = {"stock": stock}
    
    # --- DEBUG CHECK 1: Empty Index ---
    if len(common_index) == 0:
        summary["failed"] = True
        summary["fail_reason"] = "Empty common index between features and stock returns."
        return stock, None, None, None, None, None, None, None, None, None, summary

    X_stock = X_features.loc[common_index]
    y_stock = y.loc[common_index]
    
    # --- DEBUG CHECK 2: Not enough valid data points ---
    valid_y = y_stock.dropna()
    required_obs = window_size + n_lags
    if len(valid_y) < required_obs:
        summary["failed"] = True
        summary["fail_reason"] = f"Insufficient data: {len(valid_y)} valid points, but {required_obs} required (window={window_size} + lags={n_lags})."
        return stock, None, None, None, None, None, None, None, None, None, summary

    return_var = float(np.nanvar(y_stock.to_numpy(dtype=np.float64), ddof=1))

    try:
        results = estimate_single_config_fast(X_stock, y_stock, window_size, n_lags, lambda_val)
    except Exception as e:
        # --- DEBUG CHECK 3: Exception inside estimator ---
        summary["failed"] = True
        summary["fail_reason"] = f"Estimator threw Exception: {str(e)}"
        return stock, None, None, None, None, None, None, None, None, None, summary

    details = results.get("details", None)

    summary.update(dict(results.get("summary") or {}))
    summary["return_variance"] = return_var

    # --- DEBUG CHECK 4: Estimator returned None or insufficient details ---
    if details is None or getattr(details, "shape", (0, 0))[0] < 2:
        summary["failed"] = True
        summary["fail_reason"] = f"Estimator returned None or shape < 2. Actual shape: {getattr(details, 'shape', 'None')}"
        return stock, None, None, None, None, None, None, None, None, None, summary

    topics, betas, r2_in, preds, targets, lasso_intercepts, dates = _extract_details(details)
    
    # --- Compute Stage 2 OOS Predictions and In-Sample R2 ---
    stage2_preds = np.full_like(preds, np.nan)
    stage2_r2_in = np.full_like(r2_in, np.nan)
    
    kappa = summary.get('kappa', np.nan)
    stg2_intercept = summary.get('intercept', np.nan)
    
    if not np.isnan(kappa) and not np.isnan(stg2_intercept):
        if len(preds) > 1:
            alm_oos = compute_alm_returns(preds, kappa, stg2_intercept)
            stage2_preds[1:] = alm_oos  # Shift by 1 to align t+1 pred with t+1 target
            
        try:
            X_lagged, _ = create_lagged_features_with_name(X_stock, n_lags)
            y_aligned = y_stock.to_numpy().flatten()[n_lags:]
            mean_y = np.nanmean(y_stock.to_numpy().flatten())
            y_aligned_centered = y_aligned - mean_y
            
            for i in range(len(r2_in)):
                start_idx, end_idx = max(0, i), max(0, i) + window_size
                if end_idx > len(y_aligned_centered):
                    break
                    
                X_window = X_lagged[start_idx:end_idx]
                y_window_actual = y_aligned[start_idx:end_idx]
                
                scaler = StandardScaler()
                X_window_scaled = scaler.fit_transform(X_window)
                
                coef_scaled = betas[i] * scaler.scale_
                stg1_pred_in = X_window_scaled @ coef_scaled + lasso_intercepts[i] + mean_y
                
                if len(stg1_pred_in) > 1:
                    alm_in = compute_alm_returns(stg1_pred_in, kappa, stg2_intercept)
                    y_target = y_window_actual[1:] 
                    
                    valid_mask = ~np.isnan(alm_in)
                    if np.sum(valid_mask) > 2:
                        ss_res = np.sum((y_target[valid_mask] - alm_in[valid_mask])**2)
                        y_mean_window = np.mean(y_target[valid_mask])
                        ss_tot = np.sum((y_target[valid_mask] - y_mean_window)**2)
                        if ss_tot > 1e-10:
                            stage2_r2_in[i] = 1.0 - (ss_res / ss_tot)
        except Exception:
            pass
            
    return stock, topics, betas, r2_in, preds, targets, lasso_intercepts, stage2_preds, stage2_r2_in, dates, summary


def _process_one_stock_star(args: tuple) -> tuple:
    return _process_one_stock(*args)


# ============================================================
# Tensor assembly
# ============================================================

def run_estimations(
    df_returns: pd.DataFrame,
    X_features: pd.DataFrame,
    window_size: int,
    n_lags: int,
    lambda_val: float,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str], pd.DataFrame]:
    stocks = list(df_returns.columns)
    workers = (
        max(1, mp.cpu_count() - 1)
        if (n_jobs is None or int(n_jobs) == -1)
        else max(1, int(n_jobs))
    )

    args_list = [
        (s, df_returns, X_features, window_size, n_lags, lambda_val) for s in stocks
    ]
    ctx = mp.get_context("spawn") 

    raw: List[tuple] = []
    with ctx.Pool(processes=workers) as pool:
        for res in tqdm(
            pool.imap_unordered(_process_one_stock_star, args_list, chunksize=1),
            total=len(stocks),
            desc="Estimating per stock",
        ):
            raw.append(res)

    topics: Optional[List[str]] = None
    all_date_strs: set = set()
    
    # --- DEBUG CHECK 5: Tally Success vs Failures ---
    success_count = 0
    failed_stocks = []
    
    for r in raw:
        t = r[1]
        d = r[9]
        summary_info = r[10]
        
        if t is not None:
            success_count += 1
            if topics is None:
                topics = t          
            for di in d:
                all_date_strs.add(_date_str(di))
        else:
            failed_stocks.append((r[0], summary_info.get("fail_reason", "Unknown error")))

    print(f"\n--- Estimation Report ---")
    print(f"Total stocks attempted: {len(stocks)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_stocks)}")
    
    if failed_stocks:
        print("\nFailure breakdown:")
        for stock_name, reason in failed_stocks:
            print(f" - {stock_name}: {reason}")
    print("-------------------------\n")

    if topics is None:
        raise RuntimeError(f"All {len(stocks)} stocks failed — cannot build tensor. See reasons above.")

    date_strs = sorted(all_date_strs)
    date_to_idx = {ds: i for i, ds in enumerate(date_strs)}
    n_topics = len(topics)
    n_dates = len(date_strs)

    result_by_stock = {r[0]: r for r in raw}
    tensor_betas = np.full((len(stocks), n_topics, n_dates), np.nan, dtype=np.float32)
    
    mat_r2_in = np.full((len(stocks), n_dates), np.nan, dtype=np.float32)
    mat_preds = np.full((len(stocks), n_dates), np.nan, dtype=np.float32)
    mat_targets = np.full((len(stocks), n_dates), np.nan, dtype=np.float32)
    
    mat_lasso_intercepts = np.full((len(stocks), n_dates), np.nan, dtype=np.float32)
    mat_stage2_preds = np.full((len(stocks), n_dates), np.nan, dtype=np.float32)
    mat_stage2_r2_in = np.full((len(stocks), n_dates), np.nan, dtype=np.float32)
    
    summary_rows: List[Dict[str, Any]] = []

    for i, s in enumerate(stocks):
        if s not in result_by_stock:
            continue
        _, _, betas, r2_in, preds, targets, intercepts, stg2_preds, stg2_r2_in, dates, summary = result_by_stock[s]
        
        if betas is not None and dates is not None:
            for j, dj in enumerate(dates):
                k = date_to_idx.get(_date_str(dj))
                if k is not None:
                    tensor_betas[i, :, k] = betas[j]
                    mat_r2_in[i, k] = r2_in[j]
                    mat_preds[i, k] = preds[j]
                    mat_targets[i, k] = targets[j]
                    mat_lasso_intercepts[i, k] = intercepts[j]
                    mat_stage2_preds[i, k] = stg2_preds[j]
                    mat_stage2_r2_in[i, k] = stg2_r2_in[j]
                    
        if summary is not None:
            summary_rows.append(summary)

    summary_df = (
        pd.DataFrame(summary_rows).set_index("stock")
        if summary_rows
        else pd.DataFrame()
    )

    return tensor_betas, mat_r2_in, mat_preds, mat_targets, mat_lasso_intercepts, mat_stage2_preds, mat_stage2_r2_in, stocks, topics, date_strs, summary_df

# ============================================================
# Output persistence
# ============================================================

def save_outputs(
    tensor_betas: np.ndarray,
    mat_r2_in: np.ndarray,
    mat_preds: np.ndarray,
    mat_targets: np.ndarray,
    mat_lasso_intercepts: np.ndarray,
    mat_stage2_preds: np.ndarray,
    mat_stage2_r2_in: np.ndarray,
    stocks: List[str],
    topics: List[str],
    date_strs: List[str],
    summary_df: pd.DataFrame,
    out_dir: str,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    h5_path = os.path.join(out_dir, "betas.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("betas", data=tensor_betas, dtype="float32", compression="gzip", compression_opts=4)
        
        f.create_dataset("r2_in", data=mat_r2_in, dtype="float32", compression="gzip", compression_opts=4)
        f.create_dataset("predictions", data=mat_preds, dtype="float32", compression="gzip", compression_opts=4)
        f.create_dataset("targets", data=mat_targets, dtype="float32", compression="gzip", compression_opts=4)
        
        f.create_dataset("lasso_intercepts", data=mat_lasso_intercepts, dtype="float32", compression="gzip", compression_opts=4)
        f.create_dataset("stage2_predictions", data=mat_stage2_preds, dtype="float32", compression="gzip", compression_opts=4)
        f.create_dataset("stage2_r2_in", data=mat_stage2_r2_in, dtype="float32", compression="gzip", compression_opts=4)
        
        f.create_dataset("stocks", data=np.array(stocks,     dtype=h5py.string_dtype()))
        f.create_dataset("topics", data=np.array(topics,     dtype=h5py.string_dtype()))
        f.create_dataset("dates",  data=np.array(date_strs,  dtype=h5py.string_dtype()))

    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv)

    print(f"Saved beta tensor  → {h5_path}  {tensor_betas.shape}  ({tensor_betas.nbytes / 1e9:.2f} GB uncompressed)")
    print(f"Saved summary      → {summary_csv}")

    return {"betas_h5": h5_path, "summary_csv": summary_csv}


# ============================================================
# Main entrypoint
# ============================================================

def main(
    returns_csv: str | Path = None,
    features_csv: str | Path = None,
    best_hyperparameters: str | Path = None,
    out_dir: str | Path = None,
    seed: int = 42,
    num_topics: int = -1,
    num_stocks_features: int = 0,
    n_jobs: int = -1,
    subset_start: int = 0,
    subset_end: int = -1,
    save: bool = True,
) -> Dict[str, Any]:
    random.seed(seed)

    # --- PORTABLE PFAD Logic ---
    if returns_csv is None:
        returns_csv = REPO_ROOT / "Data" / "data_raw" / "cross_sectional_returns.csv"
    else:
        returns_csv = Path(returns_csv)

    if features_csv is None:
        features_csv = REPO_ROOT / "Data" / "clean_data" / "final_macro_topic_features.csv"
    else:
        features_csv = Path(features_csv)

    if best_hyperparameters is None:
        best_hyperparameters = REPO_ROOT / "Data" / "clean_data" / "best_hyperparameters.txt"
    else:
        best_hyperparameters = Path(best_hyperparameters)

    if out_dir is None:
        out_dir = REPO_ROOT / "Results" / "Estimation"
    else:
        out_dir = Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    print(f"--- Debug Output ---")
    print(f"Load Returns from: {returns_csv}")
    df = pd.read_csv(returns_csv, index_col=0)
    print(f"Initial Returns Shape: {df.shape}")
    
    print(f"Load Features from: {features_csv}")
    feature_matrix = pd.read_csv(features_csv, index_col=0, parse_dates=True)
    print(f"Initial Features Shape: {feature_matrix.shape}")

    df = df[[col for col in df.columns if str(col).replace(".", "", 1).isdigit()]]
    df = np.log(1 + df)
    df.index = pd.to_datetime(df.index)

    X = feature_matrix.copy()
    X.columns = [str(col).replace(" ", "_") for col in X.columns]

    topic_cols = [col for col in X.columns if not str(col).isdigit()]
    stock_cols = [col for col in X.columns if str(col).isdigit()]

    selected_topics = (
        topic_cols if (num_topics is None or num_topics < 0)
        else random.sample(topic_cols, min(num_topics, len(topic_cols)))
    )
    selected_stocks = random.sample(stock_cols, min(num_stocks_features, len(stock_cols)))
    X = X[selected_topics + selected_stocks]

    if selected_stocks:
        X[selected_stocks] = np.log(X[selected_stocks] + 1)

    # --- DEBUG CHECK 6: Align Data ---
    common_index = X.index.intersection(df.index)
    print(f"\nData Alignment check:")
    print(f"Features Date Range: {X.index.min()} to {X.index.max()}")
    print(f"Returns Date Range:  {df.index.min()} to {df.index.max()}")
    print(f"Overlapping Dates Found: {len(common_index)}")
    if len(common_index) == 0:
        print("CRITICAL WARNING: 0 overlapping dates between Features and Returns! Tensor will fail.")

    X = X.loc[common_index].sort_index()
    df = df.loc[common_index].sort_index()

    best = read_best_hyperparameters(str(best_hyperparameters))
    window_size = int(best["window_size"])
    n_lags      = int(best["n_lags"])
    lambda_val  = float(best["lambda"])
    print(f"\nHyperparameters loaded: Window={window_size}, Lags={n_lags}, Lambda={lambda_val}")

    all_stocks = df.columns
    s0 = max(0, int(subset_start))
    s1 = len(all_stocks) if (subset_end is None or int(subset_end) < 0) else int(subset_end)
    df_sub = df.iloc[:, s0:s1]
    
    print(f"Selecting subset [{s0}:{s1}] -> Total {len(df_sub.columns)} stocks to evaluate.")
    print(f"--------------------\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (tensor_betas, mat_r2_in, mat_preds, mat_targets, 
         mat_lasso_intercepts, mat_stage2_preds, mat_stage2_r2_in, 
         stocks, topics, date_strs, summary_df) = run_estimations(
            df_sub, X, window_size, n_lags, lambda_val, n_jobs=n_jobs,
        )

    paths: Dict[str, str] = {}
    if save:
        paths = save_outputs(
            tensor_betas, mat_r2_in, mat_preds, mat_targets, 
            mat_lasso_intercepts, mat_stage2_preds, mat_stage2_r2_in,
            stocks, topics, date_strs, summary_df, str(out_dir)
        )

    return {
        "tensor_betas": tensor_betas,
        "mat_r2_in":    mat_r2_in,
        "mat_preds":    mat_preds,
        "mat_targets":  mat_targets,
        "mat_stage2_preds": mat_stage2_preds,
        "mat_stage2_r2_in": mat_stage2_r2_in,
        "stocks":       stocks,
        "topics":       topics,
        "date_strs":    date_strs,
        "summary_df":   summary_df,
        "paths":        paths,
        "config": {
            "returns_csv":          str(returns_csv),
            "features_csv":         str(features_csv),
            "best_hyperparameters": str(best_hyperparameters),
            "out_dir":              str(out_dir),
            "seed":                 seed,
            "num_topics":           num_topics,
            "window_size":          window_size,
            "n_lags":               n_lags,
            "lambda":               lambda_val,
        },
    }

if __name__ == "__main__":
    import argparse

    default_returns = REPO_ROOT / "Data" / "data_raw" / "cross_sectional_returns.csv"
    default_features = REPO_ROOT / "Data" / "clean_data" / "final_macro_topic_features.csv"
    default_hyperparams = REPO_ROOT / "Data" / "clean_data" / "best_hyperparameters.txt"
    default_output = REPO_ROOT / "Results" / "Estimation"

    p = argparse.ArgumentParser(
        description="Cross-sectional estimation with portable paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    
    p.add_argument("--returns_csv",          default=str(default_returns), 
                   help="Path to returns CSV")
    p.add_argument("--features_csv",         default=str(default_features), 
                   help="Path to features CSV (Macro + Topics)")
    p.add_argument("--best_hyperparameters", default=str(default_hyperparams), 
                   help="Path to best_hyperparameters.txt")
    p.add_argument("--out_dir",              default=str(default_output), 
                   help="Directory where results (betas.h5, summary.csv) will be saved")
    
    p.add_argument("--seed",                 type=int, default=42, 
                   help="Random seed for sampling")
    p.add_argument("--num_topics",           type=int, default=-1, 
                   help="Number of topics to use (-1 for all)")
    p.add_argument("--num_stocks_features",  type=int, default=0, 
                   help="Number of stocks to include as features")
    p.add_argument("--n_jobs",               type=int, default=-1, 
                   help="Number of parallel workers (-1 = use all but one core)")
    p.add_argument("--subset_start",         type=int, default=0, 
                   help="Start index for stock subset")
    p.add_argument("--subset_end",           type=int, default=-1, 
                   help="End index for stock subset (-1 for all)")
    
    args = p.parse_args()

    main(
        returns_csv=args.returns_csv,
        features_csv=args.features_csv,
        best_hyperparameters=args.best_hyperparameters,
        out_dir=args.out_dir,
        seed=args.seed,
        num_topics=args.num_topics,
        num_stocks_features=args.num_stocks_features,
        n_jobs=args.n_jobs,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
        save=True,
    )