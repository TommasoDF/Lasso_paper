# cross_sectional_estimation_v1.py
#
# Simplified version with no checkpointing.
# Runs all stocks in one pass and saves:
#   betas.h5    — HDF5 tensor (n_stocks, n_topics, n_dates) float32
#   summary.csv — per-stock scalar metrics (R², kappa, return variance, …)
#
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import re
import multiprocessing as mp
import random
import warnings

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm

from grid_search import estimate_single_config_fast


# ============================================================
# Preprocessing helpers
# ============================================================

def to_ar1_innovations(X: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Column-wise AR(1) innovations: residuals from x_t on [1, x_{t-1}]."""
    X_innov = pd.DataFrame(index=X.index, columns=X.columns, dtype="float64")
    for col in X.columns:
        s = pd.to_numeric(X[col], errors="coerce")
        tmp = pd.DataFrame({"x": s, "x_lag1": s.shift(1)}).dropna()
        if len(tmp) < min_obs or tmp["x"].nunique() < 3 or tmp["x_lag1"].nunique() < 3:
            continue
        res = sm.OLS(tmp["x"], sm.add_constant(tmp["x_lag1"])).fit()
        X_innov.loc[tmp.index, col] = res.resid
    return X_innov


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


def _extract_betas(
    details: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Extract the LASSO beta matrix from a details DataFrame.

    Uses only lag-1 columns — sufficient when n_lags=1, and the natural
    single-coefficient-per-topic representation otherwise.

    Returns
    -------
    topics : sorted list of topic names (length T)
    betas  : float32 array (n_valid_windows, T)
    dates  : datetime64 array (n_valid_windows,)
    """
    topic_col: Dict[str, str] = {}
    for c in details.columns:
        m = _LAG1_COL_RE.match(str(c))
        if m:
            topic_col[m.group(1)] = c

    topics = sorted(topic_col.keys())

    # Drop rows with missing dates (failed windows)
    if "date" in details.columns:
        valid = details["date"].notna()
        sub = details.loc[valid]
        dates = pd.to_datetime(sub["date"]).to_numpy()
    else:
        sub = details
        dates = np.arange(len(sub), dtype=np.int64)

    betas = sub[[topic_col[t] for t in topics]].to_numpy(dtype=np.float32)
    return topics, betas, dates


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
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, Any],
]:
    """
    Returns
    -------
    stock   : ticker string
    topics  : sorted topic names, or None if estimation failed
    betas   : float32 (n_windows, n_topics), or None
    dates   : datetime64 array (n_windows,), or None
    summary : dict of scalar metrics (always present; contains 'error' if failed)
    """
    y = df_returns[stock]
    common_index = X_features.index.intersection(y.index)
    X_stock = X_features.loc[common_index]
    y_stock = y.loc[common_index]

    return_var = float(np.nanvar(y_stock.to_numpy(dtype=np.float64), ddof=1))

    results = estimate_single_config_fast(X_stock, y_stock, window_size, n_lags, lambda_val)
    details = results.get("details", None)

    summary: Dict[str, Any] = dict(results.get("summary") or {})
    summary["stock"] = stock
    summary["return_variance"] = return_var

    if details is None or getattr(details, "shape", (0, 0))[0] < 2:
        summary["failed"] = True
        return stock, None, None, None, summary

    topics, betas, dates = _extract_betas(details)
    return stock, topics, betas, dates, summary


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
) -> Tuple[np.ndarray, List[str], List[str], List[str], pd.DataFrame]:
    """
    Run LASSO estimation for every stock and assemble the output tensor.

    Returns
    -------
    tensor     : float32 ndarray (n_stocks, n_topics, n_dates); NaN where missing
    stocks     : list of ticker strings  — axis 0 of tensor
    topics     : list of topic names     — axis 1 of tensor
    date_strs  : list of 'YYYY-MM-DD'   — axis 2 of tensor
    summary_df : DataFrame (index = stock) with per-stock scalar metrics
    """
    stocks = list(df_returns.columns)
    workers = (
        max(1, mp.cpu_count() - 1)
        if (n_jobs is None or int(n_jobs) == -1)
        else max(1, int(n_jobs))
    )

    args_list = [
        (s, df_returns, X_features, window_size, n_lags, lambda_val) for s in stocks
    ]
    ctx = mp.get_context("spawn")  # safest on macOS / Jupyter

    raw: List[tuple] = []
    with ctx.Pool(processes=workers) as pool:
        for res in tqdm(
            pool.imap_unordered(_process_one_stock_star, args_list, chunksize=1),
            total=len(stocks),
            desc="Estimating per stock",
        ):
            raw.append(res)

    # --- Establish canonical topic and date dimensions ---
    topics: Optional[List[str]] = None
    all_date_strs: set = set()

    for _, t, _, d, _ in raw:
        if t is not None:
            if topics is None:
                topics = t          # all stocks share the same X, so topics are identical
            for di in d:
                all_date_strs.add(_date_str(di))

    if topics is None:
        raise RuntimeError("All stocks failed — cannot build tensor.")

    # ISO strings sort correctly into chronological order
    date_strs = sorted(all_date_strs)
    date_to_idx = {ds: i for i, ds in enumerate(date_strs)}
    n_topics = len(topics)
    n_dates = len(date_strs)

    # --- Fill tensor in original stock order ---
    result_by_stock = {r[0]: r for r in raw}
    tensor = np.full((len(stocks), n_topics, n_dates), np.nan, dtype=np.float32)
    summary_rows: List[Dict[str, Any]] = []

    for i, s in enumerate(stocks):
        if s not in result_by_stock:
            continue
        _, _, betas, dates, summary = result_by_stock[s]
        if betas is not None and dates is not None:
            for j, dj in enumerate(dates):
                k = date_to_idx.get(_date_str(dj))
                if k is not None:
                    tensor[i, :, k] = betas[j]   # betas[j] shape: (n_topics,)
        if summary is not None:
            summary_rows.append(summary)

    summary_df = (
        pd.DataFrame(summary_rows).set_index("stock")
        if summary_rows
        else pd.DataFrame()
    )

    return tensor, stocks, topics, date_strs, summary_df


# ============================================================
# Output persistence
# ============================================================

def save_outputs(
    tensor: np.ndarray,
    stocks: List[str],
    topics: List[str],
    date_strs: List[str],
    summary_df: pd.DataFrame,
    out_dir: str,
) -> Dict[str, str]:
    """
    Persist final results.

    Files written
    -------------
    betas.h5    — HDF5 containing:
                    /betas   float32 (n_stocks, n_topics, n_dates)
                    /stocks  string  (n_stocks,)
                    /topics  string  (n_topics,)
                    /dates   string  (n_dates,)  'YYYY-MM-DD'
    summary.csv — per-stock scalar metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    h5_path = os.path.join(out_dir, "betas.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "betas", data=tensor,
            dtype="float32", compression="gzip", compression_opts=4,
        )
        f.create_dataset("stocks", data=np.array(stocks,     dtype=h5py.string_dtype()))
        f.create_dataset("topics", data=np.array(topics,     dtype=h5py.string_dtype()))
        f.create_dataset("dates",  data=np.array(date_strs,  dtype=h5py.string_dtype()))

    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv)

    print(f"Saved beta tensor  → {h5_path}  {tensor.shape}  ({tensor.nbytes / 1e9:.2f} GB uncompressed)")
    print(f"Saved summary      → {summary_csv}")

    return {"betas_h5": h5_path, "summary_csv": summary_csv}


# ============================================================
# Main entrypoint
# ============================================================

def main(
    returns_csv: str = "../../data/X.csv",
    features_csv: str = "../../data/merged_return_topic_data.csv",
    best_hyperparameters: str = "best_hyperparameters.txt",
    out_dir: str = ".",
    seed: int = 42,
    num_topics: int = -1,
    num_stocks_features: int = 0,
    n_jobs: int = -1,
    subset_start: int = 0,
    subset_end: int = -1,
    ar1_innovations: bool = False,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Cross-sectional LASSO estimation — no checkpointing, single-pass.

    Outputs in out_dir
    ------------------
    betas.h5    : float32 tensor (n_stocks, n_topics, n_dates)
    summary.csv : per-stock scalar metrics (R², kappa, return variance, …)
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(returns_csv, index_col=0)
    feature_matrix = pd.read_csv(features_csv, index_col=0, parse_dates=True)

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

    if ar1_innovations:
        X = to_ar1_innovations(X)
    if selected_stocks:
        X[selected_stocks] = np.log(X[selected_stocks] + 1)

    common_index = X.index.intersection(df.index)
    X = X.loc[common_index].sort_index()
    df = df.loc[common_index].sort_index()

    # --- Hyperparameters ---
    best = read_best_hyperparameters(best_hyperparameters)
    window_size = int(best["window_size"])
    n_lags      = int(best["n_lags"])
    lambda_val  = float(best["lambda"])

    # --- Stock subset ---
    all_stocks = df.columns
    s0 = max(0, int(subset_start))
    s1 = len(all_stocks) if (subset_end is None or int(subset_end) < 0) else int(subset_end)
    df_sub = df.iloc[:, s0:s1]

    # --- Run estimations ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor, stocks, topics, date_strs, summary_df = run_estimations(
            df_sub, X, window_size, n_lags, lambda_val, n_jobs=n_jobs,
        )

    # --- Persist ---
    paths: Dict[str, str] = {}
    if save:
        paths = save_outputs(tensor, stocks, topics, date_strs, summary_df, out_dir)

    return {
        "tensor":     tensor,
        "stocks":     stocks,
        "topics":     topics,
        "date_strs":  date_strs,
        "summary_df": summary_df,
        "paths":      paths,
        "config": {
            "returns_csv":          returns_csv,
            "features_csv":         features_csv,
            "best_hyperparameters": best_hyperparameters,
            "out_dir":              out_dir,
            "seed":                 seed,
            "num_topics":           num_topics,
            "num_stocks_features":  num_stocks_features,
            "n_jobs":               n_jobs,
            "subset_start":         subset_start,
            "subset_end":           subset_end,
            "ar1_innovations":      ar1_innovations,
            "window_size":          window_size,
            "n_lags":               n_lags,
            "lambda":               lambda_val,
        },
    }


# ============================================================
# CLI wrapper
# ============================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--returns_csv",          default="../../data/X.csv")
    p.add_argument("--features_csv",         default="../../data/merged_return_topic_data.csv")
    p.add_argument("--best_hyperparameters", default="best_hyperparameters.txt")
    p.add_argument("--out_dir",              default=".")
    p.add_argument("--seed",                 type=int, default=42)
    p.add_argument("--num_topics",           type=int, default=-1)
    p.add_argument("--num_stocks_features",  type=int, default=0)
    p.add_argument("--n_jobs",               type=int, default=-1)
    p.add_argument("--subset_start",         type=int, default=0)
    p.add_argument("--subset_end",           type=int, default=-1)
    p.add_argument("--ar1_innovations",      action="store_true")
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
        ar1_innovations=args.ar1_innovations,
        save=True,
    )
