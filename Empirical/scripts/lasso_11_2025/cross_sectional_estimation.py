# cross_sectional_estimation.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Iterable
import os
import re

import pickle
import random
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# your project imports
from grid_search import estimate_single_config_fast


# ----------------------------
# Preprocessing helpers
# ----------------------------

def to_ar1_innovations(X: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
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


def build_results_template(stocks: pd.Index, feature_cols: pd.Index, n_lags: int) -> pd.DataFrame:
    results_df = pd.DataFrame(columns=["avg_num_selected_features"], index=stocks)

    lasso_cols = [
        f"Lasso_{feature}_lag_{lag}"
        for feature in feature_cols
        for lag in range(1, n_lags + 1)
    ]
    zeros = pd.DataFrame(0, index=stocks, columns=lasso_cols, dtype="int64")
    results_df = pd.concat([results_df, zeros], axis=1)
    return results_df


# ----------------------------
# Checkpointing helpers
# ----------------------------

def _safe_filename(name: str) -> str:
    # Keep it simple and filesystem-safe
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _ensure_dirs(out_dir: str) -> Dict[str, str]:
    details_dir = os.path.join(out_dir, "details")
    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    return {"details_dir": details_dir, "summary_dir": summary_dir}


def _details_path(details_dir: str, stock: str) -> str:
    return os.path.join(details_dir, f"{_safe_filename(stock)}.pkl")


def _summary_path(summary_dir: str, stock: str) -> str:
    return os.path.join(summary_dir, f"{_safe_filename(stock)}.pkl")


def load_completed_stocks(out_dir: str) -> set[str]:
    """
    Determine completed stocks based on existence of per-stock details files.
    This is robust against partial csv writes and lets you resume.
    """
    dirs = _ensure_dirs(out_dir)
    details_dir = dirs["details_dir"]
    completed = set()

    for fn in os.listdir(details_dir):
        if fn.endswith(".pkl"):
            completed.add(fn[:-4])  # safe filename (not original stock id)
    return completed


def checkpoint_write_row(
    results_df: pd.DataFrame,
    out_dir: str,
    stock: str,
    row: Dict[str, Any],
) -> None:
    """
    Update in-memory results_df and persist a full CSV snapshot.
    This is intentionally simple: overwrite the snapshot each time.
    """
    results_df.loc[stock, list(row.keys())] = list(row.values())
    results_csv = os.path.join(out_dir, "cross_sectional_lasso_results.csv")
    results_df.to_csv(results_csv)


def checkpoint_write_details_summary(
    out_dir: str,
    stock: str,
    details: Any,
    summary: Any,
) -> None:
    dirs = _ensure_dirs(out_dir)
    details_dir = dirs["details_dir"]
    summary_dir = dirs["summary_dir"]

    # per-stock pickles, atomic-ish write
    dpath = _details_path(details_dir, stock)
    spath = _summary_path(summary_dir, stock)

    with open(dpath, "wb") as f:
        pickle.dump(details, f)

    with open(spath, "wb") as f:
        pickle.dump(summary, f)


def load_checkpoints_into_memory(out_dir: str, stocks: Iterable[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load per-stock details/summary pickles for provided original stock names.
    If a pickle doesn't exist, value is None.
    """
    dirs = _ensure_dirs(out_dir)
    details_dir = dirs["details_dir"]
    summary_dir = dirs["summary_dir"]

    details_per_stock: Dict[str, Any] = {}
    summary_per_stock: Dict[str, Any] = {}

    for stock in stocks:
        dpath = _details_path(details_dir, stock)
        spath = _summary_path(summary_dir, stock)

        if os.path.exists(dpath):
            with open(dpath, "rb") as f:
                details_per_stock[stock] = pickle.load(f)
        else:
            details_per_stock[stock] = None

        if os.path.exists(spath):
            with open(spath, "rb") as f:
                summary_per_stock[stock] = pickle.load(f)
        else:
            summary_per_stock[stock] = None

    return details_per_stock, summary_per_stock


# ----------------------------
# Estimation core
# ----------------------------

def _process_one_stock(
    stock: str,
    df_returns: pd.DataFrame,
    X_features: pd.DataFrame,
    window_size: int,
    n_lags: int,
    lambda_val: float,
) -> Tuple[str, Optional[Dict[str, Any]], Any, Any]:
    y = df_returns[stock]

    common_index = X_features.index.intersection(y.index)
    X_stock = X_features.loc[common_index]
    y_stock = y.loc[common_index]

    results = estimate_single_config_fast(X_stock, y_stock, window_size, n_lags, lambda_val)
    details = results.get("details", None)
    summary = results.get("summary", None)

    if details is None or getattr(details, "shape", (0, 0))[0] < 2:
        return stock, None, None, summary

    avg_num_selected = float(details["num_nonzero_coefficients"].mean())
    row: Dict[str, Any] = {"avg_num_selected_features": avg_num_selected}

    lasso_cols = [c for c in details.columns if str(c).startswith("Lasso_")]
    if lasso_cols:
        nonzero_counts = (details[lasso_cols] != 0).sum()
        row.update(nonzero_counts.to_dict())

    for feature in X_features.columns:
        for lag_number in range(1, n_lags + 1):
            col_name = f"Lasso_{feature}_lag_{lag_number}"
            row.setdefault(col_name, 0)

    return stock, row, details, summary


def run_estimations_with_checkpointing(
    df_returns: pd.DataFrame,
    X_features: pd.DataFrame,
    results_df: pd.DataFrame,
    window_size: int,
    n_lags: int,
    lambda_val: float,
    out_dir: str,
    n_jobs: int = -1,
    batch_size: str | int = "auto",
    resume: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Minimal checkpoint strategy:
      - For each stock, write details/{stock}.pkl and summary/{stock}.pkl immediately
      - Update and overwrite cross_sectional_lasso_results.csv after each stock
      - If resume=True, skip stocks that already have details/{stock}.pkl

    Note: with joblib parallelism, "after each stock" becomes "after each completed task"
    in the main process as results are collected. This still protects you from kernel crashes.
    """
    _ensure_dirs(out_dir)

    stocks = list(df_returns.columns)
    completed_safe_names = load_completed_stocks(out_dir) if resume else set()

    # Determine which original stock names are done by checking safe filename existence
    def _is_done(stock_name: str) -> bool:
        return _safe_filename(stock_name) in completed_safe_names

    todo_stocks = [s for s in stocks if not _is_done(s)]
    if resume and len(todo_stocks) < len(stocks):
        print(f"Resuming: {len(stocks) - len(todo_stocks)} already completed, {len(todo_stocks)} remaining.")

    # If nothing to do, load existing per-stock outputs into memory and return
    if len(todo_stocks) == 0:
        details_per_stock, summary_per_stock = load_checkpoints_into_memory(out_dir, stocks)
        # Ensure results_df is loaded from disk if present
        results_csv = os.path.join(out_dir, "cross_sectional_lasso_results.csv")
        if os.path.exists(results_csv):
            results_df = pd.read_csv(results_csv, index_col=0)
        return results_df, details_per_stock, summary_per_stock

    # Parallel compute for remaining stocks
    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size=batch_size)(
        delayed(_process_one_stock)(stock, df_returns, X_features, window_size, n_lags, lambda_val)
        for stock in tqdm(todo_stocks, desc="Estimating per stock")
    )

    # Load existing checkpointed dicts (for those already completed earlier) if resume
    details_per_stock, summary_per_stock = load_checkpoints_into_memory(out_dir, stocks) if resume else ({}, {})

    # Collect and checkpoint new ones
    for stock, row, details, summary in out:
        checkpoint_write_details_summary(out_dir, stock, details, summary)
        details_per_stock[stock] = details
        summary_per_stock[stock] = summary

        if row is not None:
            checkpoint_write_row(results_df, out_dir, stock, row)

    return results_df, details_per_stock, summary_per_stock


# ----------------------------
# Notebook-friendly entrypoint
# ----------------------------

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
    Call from notebook or CLI. Implements minimal checkpointing.

    Outputs in out_dir:
      - cross_sectional_lasso_results.csv (updated as stocks finish)
      - details/{stock}.pkl (per stock)
      - summary/{stock}.pkl (per stock)

    resume=True skips stocks already having details/{stock}.pkl.
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(returns_csv, index_col=0)
    feature_matrix = pd.read_csv(features_csv, index_col=0, parse_dates=True)

    # Keep only numeric stock columns in returns
    df = df[[col for col in df.columns if str(col).replace(".", "", 1).isdigit()]]

    # Convert returns to log(1+r) and ensure datetime index
    df = np.log(1 + df)
    df.index = pd.to_datetime(df.index)

    # Features
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

    # Align
    common_index_all = X.index.intersection(df.index)
    X = X.loc[common_index_all].sort_index()
    df = df.loc[common_index_all].sort_index()

    # Hyperparameters
    best = read_best_hyperparameters(best_hyperparameters)
    window_size = int(best.get("window_size"))
    n_lags = int(best.get("n_lags"))
    lambda_val = float(best.get("lambda"))

    # Subset of stocks
    stocks = df.columns
    s0 = max(0, int(subset_start))
    s1 = len(stocks) if subset_end is None or int(subset_end) < 0 else int(subset_end)
    df_sub = df.iloc[:, s0:s1]

    # Results template
    results_df = build_results_template(df_sub.columns, X.columns, n_lags)

    # If resuming and a prior CSV exists, load it to keep previously-filled rows
    if resume:
        results_csv = os.path.join(out_dir, "cross_sectional_lasso_results.csv")
        if os.path.exists(results_csv):
            try:
                results_df = pd.read_csv(results_csv, index_col=0)
            except Exception:
                pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_df, details_per_stock, summary_per_stock = run_estimations_with_checkpointing(
            df_sub, X, results_df, window_size, n_lags, lambda_val,
            out_dir=out_dir, n_jobs=n_jobs, batch_size=batch_size, resume=resume
        )

    # Optional: also write consolidated pickles at end (small convenience)
    paths: Dict[str, str] = {}
    if save_outputs:
        consolidated_details = os.path.join(out_dir, "details_per_stock.pkl")
        consolidated_summary = os.path.join(out_dir, "summary_per_stock.pkl")

        with open(consolidated_details, "wb") as f:
            pickle.dump(details_per_stock, f)
        with open(consolidated_summary, "wb") as f:
            pickle.dump(summary_per_stock, f)

        paths = {
            "results_csv": os.path.join(out_dir, "cross_sectional_lasso_results.csv"),
            "details_dir": os.path.join(out_dir, "details"),
            "summary_dir": os.path.join(out_dir, "summary"),
            "details_pkl": consolidated_details,
            "summary_pkl": consolidated_summary,
        }

    return {
        "results_df": results_df,
        "details_per_stock": details_per_stock,
        "summary_per_stock": summary_per_stock,
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


# ----------------------------
# CLI wrapper
# ----------------------------
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
