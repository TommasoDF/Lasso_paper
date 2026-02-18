# cross_sectional_estimation.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import os
import re
import multiprocessing as mp
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm

# your project imports
from grid_search import estimate_single_config_fast


# ============================================================
# Preprocessing helpers
# ============================================================

def to_ar1_innovations(X: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """
    Column-wise AR(1) innovations: residuals from x_t on [1, x_{t-1}].
    """
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
# Checkpointing helpers
# ============================================================

def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _ensure_dirs(out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    compact_dir = os.path.join(out_dir, "compact")
    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(compact_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    return {"compact_dir": compact_dir, "summary_dir": summary_dir}


def _atomic_pickle_dump(obj: Any, path: str) -> None:
    """
    Atomic-ish pickle write: write tmp -> fsync -> os.replace.
    Prevents truncated pickles (EOFError) after interrupts/crashes.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _compact_path(out_dir: str, stock: str) -> str:
    dirs = _ensure_dirs(out_dir)
    return os.path.join(dirs["compact_dir"], f"{_safe_filename(stock)}.pkl")


def _summary_path(out_dir: str, stock: str) -> str:
    dirs = _ensure_dirs(out_dir)
    return os.path.join(dirs["summary_dir"], f"{_safe_filename(stock)}.pkl")


def checkpoint_write_compact(out_dir: str, stock: str, compact: Dict[str, Any]) -> None:
    _atomic_pickle_dump(compact, _compact_path(out_dir, stock))


def checkpoint_write_summary(out_dir: str, stock: str, summary: Any) -> None:
    _atomic_pickle_dump(summary, _summary_path(out_dir, stock))


def checkpoint_write_row(results_df: pd.DataFrame, out_dir: str, stock: str, row: Dict[str, Any]) -> None:
    """
    Update in-memory results_df and persist a full CSV snapshot.
    Overwrites the snapshot each time (simple, robust).
    """
    for k in row.keys():
        if k not in results_df.columns:
            results_df[k] = np.nan
    results_df.loc[stock, list(row.keys())] = list(row.values())
    results_csv = os.path.join(out_dir, "cross_sectional_lasso_results.csv")
    results_df.to_csv(results_csv)


def load_completed_stocks(out_dir: str) -> set[str]:
    """
    Completed stocks are those with compact/<stock>.pkl existing.
    Uses safe filenames.
    """
    dirs = _ensure_dirs(out_dir)
    compact_dir = dirs["compact_dir"]
    completed = set()
    for fn in os.listdir(compact_dir):
        if fn.endswith(".pkl"):
            completed.add(fn[:-4])
    return completed


# ============================================================
# Reduction: details -> compact objects needed for analysis
# ============================================================

_LASSO_COL_RE = re.compile(r"^Lasso_(.+)_lag_(\d+)$")


def _topic_from_lasso_col(col: str) -> Optional[str]:
    m = _LASSO_COL_RE.match(str(col))
    return m.group(1) if m else None


def _mean_run_length(active: np.ndarray) -> float:
    """
    active: 1D boolean/int array
    mean length of contiguous True-runs; 0.0 if never active
    """
    a = np.asarray(active, dtype=np.int8)
    if a.size == 0 or a.max() == 0:
        return 0.0
    padded = np.r_[0, a, 0]
    starts = np.where((padded[1:] == 1) & (padded[:-1] == 0))[0]
    ends = np.where((padded[1:] == 0) & (padded[:-1] == 1))[0]
    lengths = (ends - starts).astype(np.float64)
    return float(lengths.mean()) if lengths.size else 0.0


def reduce_details_to_compact(details: pd.DataFrame, return_variance: float) -> Dict[str, Any]:
    """
    Produces ONLY what you need.

    Stock level:
      - avg selection frequency per topic (aggregating over lags)
      - avg run length of being selected per topic (aggregating over lags)
      - mean # active coefficients
      - return variance

    Time level:
      - # active coefficients over time
      - stage 1 in-sample R^2 over time
      - stage 1 out-of-sample R^2 over time (cumulative definition)
    """
    out: Dict[str, Any] = {"return_variance": float(return_variance)}

    # ---- time series ----
    if "num_nonzero_coefficients" in details:
        ts_num_active = details["num_nonzero_coefficients"].to_numpy(dtype=np.int16, copy=True)
    else:
        lasso_cols_fallback = [c for c in details.columns if str(c).startswith("Lasso_")]
        ts_num_active = (details[lasso_cols_fallback].to_numpy() != 0).sum(axis=1).astype(np.int16) if lasso_cols_fallback else np.array([], dtype=np.int16)

    out["ts_num_active"] = ts_num_active
    out["mean_num_active"] = float(np.mean(ts_num_active)) if ts_num_active.size else np.nan

    out["ts_insample_r2"] = details["insample_r_squareds"].to_numpy(dtype=np.float32, copy=True) if "insample_r_squareds" in details else np.array([], dtype=np.float32)

    # cumulative OOS R^2 over time
    if ("predictions" in details) and ("targets" in details):
        p = details["predictions"].to_numpy(dtype=np.float64, copy=False)
        t = details["targets"].to_numpy(dtype=np.float64, copy=False)
        m = np.isfinite(p) & np.isfinite(t)

        if m.any():
            p2 = p[m]
            t2 = t[m]

            err2 = (t2 - p2) ** 2
            sse = np.cumsum(err2)

            csum = np.cumsum(t2)
            idx = np.arange(1, t2.size + 1, dtype=np.float64)
            mean_t = csum / idx
            sst = np.cumsum((t2 - mean_t) ** 2)

            oos_r2 = np.full_like(sse, np.nan, dtype=np.float64)
            ok = sst > 0
            oos_r2[ok] = 1.0 - (sse[ok] / sst[ok])
            out["ts_oos_r2"] = oos_r2.astype(np.float32)
        else:
            out["ts_oos_r2"] = np.array([], dtype=np.float32)
    else:
        out["ts_oos_r2"] = np.array([], dtype=np.float32)

    # ---- topic aggregation over lags ----
    lasso_cols = [c for c in details.columns if str(c).startswith("Lasso_")]
    topic_to_cols: Dict[str, List[str]] = {}
    for c in lasso_cols:
        topic = _topic_from_lasso_col(str(c))
        if topic is not None:
            topic_to_cols.setdefault(topic, []).append(c)

    avg_select: Dict[str, float] = {}
    avg_runlen: Dict[str, float] = {}
    topic_active_packed: Dict[str, bytes] = {} 

    for topic, cols in topic_to_cols.items():
        arr = details[cols].to_numpy()
        active = (arr != 0).any(axis=1).astype(np.uint8)  # aggregate across lags

        topic_active_packed[topic] = np.packbits(active).tobytes() 

        avg_select[topic] = float(active.mean()) if active.size else 0.0
        avg_runlen[topic] = _mean_run_length(active)

    out["avg_selection_by_topic"] = avg_select
    out["avg_runlen_by_topic"] = avg_runlen
    out["topic_active_packed"] = topic_active_packed   # <-- add after loop
    out["topic_active_len"] = int(details.shape[0])     # <-- add after loop
    out["n_windows"] = int(details.shape[0])



    return out


# ============================================================
# Estimation core
# ============================================================

def _process_one_stock(
    stock: str,
    df_returns: pd.DataFrame,
    X_features: pd.DataFrame,
    window_size: int,
    n_lags: int,
    lambda_val: float,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Any]:
    """
    Returns:
      - stock
      - row (for cross-sectional CSV)
      - compact dict (for notebook)
      - summary (kept)
    """
    y = df_returns[stock]

    common_index = X_features.index.intersection(y.index)
    X_stock = X_features.loc[common_index]
    y_stock = y.loc[common_index]

    return_var = float(np.nanvar(y_stock.to_numpy(dtype=np.float64), ddof=1))

    results = estimate_single_config_fast(X_stock, y_stock, window_size, n_lags, lambda_val)
    details = results.get("details", None)
    summary = results.get("summary", None)

    if details is None or getattr(details, "shape", (0, 0))[0] < 2:
        return stock, {"failed": 1}, {"failed": True, "return_variance": return_var}, summary


    compact = reduce_details_to_compact(details, return_variance=return_var)

    row: Dict[str, Any] = {
        "return_variance": compact["return_variance"],
        "mean_num_active": compact["mean_num_active"],
    }
    for topic, val in compact["avg_selection_by_topic"].items():
        row[f"avg_select_{topic}"] = val
    for topic, val in compact["avg_runlen_by_topic"].items():
        row[f"avg_runlen_{topic}"] = val

    return stock, row, compact, summary


def _process_one_stock_star(args):
    return _process_one_stock(*args)


def run_estimations_with_checkpointing(
    df_returns: pd.DataFrame,
    X_features: pd.DataFrame,
    results_df: pd.DataFrame,
    window_size: int,
    n_lags: int,
    lambda_val: float,
    out_dir: str,
    n_jobs: int = -1,
    batch_size: str | int = "auto",  # unused by mp.Pool, kept for API compatibility
    resume: bool = True,
) -> pd.DataFrame:
    """
    Streaming checkpointing:
      - For each completed stock: write compact/<stock>.pkl and summary/<stock>.pkl immediately
      - Update CSV snapshot after each completed stock
      - resume=True skips stocks with existing compact/<stock>.pkl
    """
    _ensure_dirs(out_dir)

    stocks = list(df_returns.columns)
    completed_safe_names = load_completed_stocks(out_dir) if resume else set()

    def _is_done(stock_name: str) -> bool:
        return _safe_filename(stock_name) in completed_safe_names

    todo_stocks = [s for s in stocks if not _is_done(s)]
    if resume and len(todo_stocks) < len(stocks):
        print(f"Resuming: {len(stocks) - len(todo_stocks)} already completed, {len(todo_stocks)} remaining.")

    if len(todo_stocks) == 0:
        return results_df

    if n_jobs is None or int(n_jobs) == -1:
        workers = max(1, mp.cpu_count() - 1)
    else:
        workers = max(1, int(n_jobs))

    args_iter = [(stock, df_returns, X_features, window_size, n_lags, lambda_val) for stock in todo_stocks]

    ctx = mp.get_context("spawn")  # safest on macOS/VS Code/Jupyter

    with ctx.Pool(processes=workers) as pool:
        for stock, row, compact, summary in tqdm(
            pool.imap_unordered(_process_one_stock_star, args_iter, chunksize=1),
            total=len(args_iter),
            desc="Estimating per stock",
        ):
            if compact is not None:
                checkpoint_write_compact(out_dir, stock, compact)
            if summary is not None:
                checkpoint_write_summary(out_dir, stock, summary)
            if row is not None:
                checkpoint_write_row(results_df, out_dir, stock, row)

    return results_df


# ============================================================
# Notebook-friendly entrypoint
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
    batch_size: str | int = "auto",
    subset_start: int = 0,
    subset_end: int = -1,
    ar1_innovations: bool = False,
    save_outputs: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Runs cross-sectional estimation with streaming checkpointing.
    Keeps summary per stock; never saves rolling details.

    Outputs in out_dir:
      - cross_sectional_lasso_results.csv
      - compact/<stock>.pkl
      - summary/<stock>.pkl
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    _ensure_dirs(out_dir)

    df = pd.read_csv(returns_csv, index_col=0)
    feature_matrix = pd.read_csv(features_csv, index_col=0, parse_dates=True)

    df = df[[col for col in df.columns if str(col).replace(".", "", 1).isdigit()]]
    df = np.log(1 + df)
    df.index = pd.to_datetime(df.index)

    X = feature_matrix.copy()
    X.columns = [str(col).replace(" ", "_") for col in X.columns]

    topic_cols = [col for col in X.columns if not str(col).isdigit()]
    stock_cols = [col for col in X.columns if str(col).isdigit()]

    if num_topics is None or num_topics < 0:
        selected_topics = topic_cols
    else:
        selected_topics = random.sample(topic_cols, min(num_topics, len(topic_cols)))

    selected_stocks = random.sample(stock_cols, min(num_stocks_features, len(stock_cols)))
    X = X[selected_topics + selected_stocks]

    if ar1_innovations:
        X = to_ar1_innovations(X)

    if selected_stocks:
        X[selected_stocks] = np.log(X[selected_stocks] + 1)

    common_index_all = X.index.intersection(df.index)
    X = X.loc[common_index_all].sort_index()
    df = df.loc[common_index_all].sort_index()

    best = read_best_hyperparameters(best_hyperparameters)
    window_size = int(best.get("window_size"))
    n_lags = int(best.get("n_lags"))
    lambda_val = float(best.get("lambda"))

    stocks = df.columns
    s0 = max(0, int(subset_start))
    s1 = len(stocks) if subset_end is None or int(subset_end) < 0 else int(subset_end)
    df_sub = df.iloc[:, s0:s1]

    results_csv = os.path.join(out_dir, "cross_sectional_lasso_results.csv")
    if resume and os.path.exists(results_csv):
        try:
            results_df = pd.read_csv(results_csv, index_col=0)
        except Exception:
            results_df = pd.DataFrame(index=df_sub.columns)
    else:
        results_df = pd.DataFrame(index=df_sub.columns)

    for s in df_sub.columns:
        if s not in results_df.index:
            results_df.loc[s, :] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_df = run_estimations_with_checkpointing(
            df_sub, X, results_df,
            window_size, n_lags, lambda_val,
            out_dir=out_dir, n_jobs=n_jobs, batch_size=batch_size, resume=resume
        )

    paths: Dict[str, str] = {}
    if save_outputs:
        paths = {
            "results_csv": results_csv,
            "compact_dir": os.path.join(out_dir, "compact"),
            "summary_dir": os.path.join(out_dir, "summary"),
        }

    return {
        "results_df": results_df,
        "paths": paths,
        "config": {
            "returns_csv": returns_csv,
            "features_csv": features_csv,
            "best_hyperparameters": best_hyperparameters,
            "out_dir": out_dir,
            "seed": seed,
            "num_topics": num_topics,
            "num_stocks_features": num_stocks_features,
            "n_jobs": n_jobs,
            "batch_size": batch_size,
            "subset_start": subset_start,
            "subset_end": subset_end,
            "ar1_innovations": ar1_innovations,
            "save_outputs": save_outputs,
            "resume": resume,
            "window_size": window_size,
            "n_lags": n_lags,
            "lambda": lambda_val,
        },
    }


# ============================================================
# CLI wrapper
# ============================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--returns_csv", type=str, default="../../data/X.csv")
    p.add_argument("--features_csv", type=str, default="../../data/merged_return_topic_data.csv")
    p.add_argument("--best_hyperparameters", type=str, default="best_hyperparameters.txt")
    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_topics", type=int, default=-1)
    p.add_argument("--num_stocks_features", type=int, default=0)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--subset_start", type=int, default=0)
    p.add_argument("--subset_end", type=int, default=-1)
    p.add_argument("--ar1_innovations", action="store_true")
    p.add_argument("--no_resume", action="store_true")
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
        batch_size=args.batch_size,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
        ar1_innovations=args.ar1_innovations,
        save_outputs=True,
        resume=not args.no_resume,
    )
