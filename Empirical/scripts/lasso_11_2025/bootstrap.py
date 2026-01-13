import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from grid_search import estimate_single_config
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import random
from itertools import product
import numpy as np
from tqdm.auto import tqdm

from stage1 import lasso_rolling_window, calculate_r_squared
from stage2 import estimate_kappa_curve_fit, compute_alm_returns, compute_stage2_r_squared
from grid_search import grid_search, estimate_single_config








def circular_block_bootstrap_indices(T: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Function resamples blocks of length 'block_len' from a time series with length T with replacement.
    If block extends beyond T, it wraps around to the beginning (circular).
    """
    if block_len <= 0:
        raise ValueError("block_len must be a positive integer.")
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block_len)) % T for s in starts])[:T]
    return idx


def _take_bootstrap_sample(X, y, idx):
    """
    Function takes original data X, y and a set of indices idx generated earlier,
    and returns the bootstrap sample Xb, yb.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        Xb = X.iloc[idx].reset_index(drop=True)
    else:
        Xb = np.asarray(X)[idx]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        yb = y.iloc[idx].reset_index(drop=True)
        if isinstance(yb, pd.DataFrame) and yb.shape[1] == 1:
            yb = yb.iloc[:, 0]
    else:
        yb = np.asarray(y)[idx]

    return Xb, yb


def bootstrap_two_stage_block(
    X,
    y,
    window_size: int,
    n_lags: int,
    lambda_val: float,
    B: int = 500,
    block_len: int | None = None,
    standardize: bool = True,
    random_state: int = 0,
    show_progress: bool = True,
):
    """
    Block-bootstrap the *entire* two-stage procedure (Stage 1 rolling LASSO -> Stage 2 kappa fit).

    Returns
    -------
    dict with:
      - point_estimate: dict (the original-sample summary from estimate_single_config)
      - draws: DataFrame with bootstrap draws (kappa, intercept, r2s, etc.)
      - bootstrap_summary: dict with bootstrap SEs + percentile CIs + failure rate
    """
    # ---- basic checks ----
    T = len(y)
    if len(X) != T:
        raise ValueError(f"Length mismatch: len(X)={len(X)} vs len(y)={T}")

    # rule-of-thumb default for weekly data if user doesn't specify:
    # ~ T^(1/3), with a small floor to avoid tiny blocks
    if block_len is None:
        block_len = max(8, int(round(T ** (1 / 3))))

    rng = np.random.default_rng(random_state)

    # ---- point estimate on original sample ----
    pe = estimate_single_config(
        X=X,
        y=y,
        window_size=window_size,
        n_lags=n_lags,
        lambda_val=lambda_val,
        standardize=standardize,
        verbose=False,
        save_only_positve_r_squared=False,
        return_details=False,
    )["summary"]

    # ---- bootstrap draws ----
    rows = []
    failures = 0

    it = range(B)
    if show_progress:
        it = tqdm(it, total=B, desc="Block bootstrap (2-stage)")

    for b in it:
        idx = circular_block_bootstrap_indices(T, block_len, rng)
        Xb, yb = _take_bootstrap_sample(X, y, idx)

        try:
            out = estimate_single_config(
                X=Xb,
                y=yb,
                window_size=window_size,
                n_lags=n_lags,
                lambda_val=lambda_val,
                standardize=standardize,
                verbose=False,
                save_only_positve_r_squared=False,
                return_details=False,
            )["summary"]
            out = dict(out)  # ensure mutable
            out["boot_id"] = b
            out["block_len"] = block_len
            rows.append(out)
        except Exception:
            failures += 1
            continue

    draws = pd.DataFrame(rows)

    # ---- summarize bootstrap distribution ----
    def _pct_ci(x, lo=2.5, hi=97.5):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return (np.nan, np.nan)
        return (np.percentile(x, lo), np.percentile(x, hi))

    kappa_se = float(np.nanstd(draws["kappa"].values, ddof=1)) if len(draws) else np.nan
    intercept_se = float(np.nanstd(draws["intercept"].values, ddof=1)) if len(draws) else np.nan
    kappa_ci = _pct_ci(draws["kappa"].values) if len(draws) else (np.nan, np.nan)
    intercept_ci = _pct_ci(draws["intercept"].values) if len(draws) else (np.nan, np.nan)

    summary = {
        "B_requested": B,
        "B_success": int(len(draws)),
        "failure_rate": float(failures / B) if B > 0 else np.nan,
        "block_len": int(block_len),
        "kappa_point": pe.get("kappa", np.nan),
        "kappa_boot_se": kappa_se,
        "kappa_ci_2p5_97p5": kappa_ci,
        "intercept_point": pe.get("intercept", np.nan),
        "intercept_boot_se": intercept_se,
        "intercept_ci_2p5_97p5": intercept_ci,
    }

    return {
        "point_estimate": pe,
        "draws": draws,
        "bootstrap_summary": summary,
    }