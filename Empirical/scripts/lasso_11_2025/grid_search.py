# =======================
#        IMPORTS
# =======================

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm

# Update this line to match your project structure:
from stage1 import (
    lasso_rolling_window,
    calculate_r_squared
)

from stage2 import compute_stage2_r_squared

# =======================
#      FUNCTIONS
# =======================

def estimate_single_config(X, y, window_size, n_lags, lambda_val):
    """
    Estimate 1st and 2nd stage for a single configuration and return all metrics.
    
    Returns
    -------
    dict
        Dictionary containing all performance metrics, or None if estimation fails
    """
    try:
        # Stage 1: Rolling LASSO
        lasso_results = lasso_rolling_window(
            X=X, y=y, 
            window_size=window_size, 
            n_lags=n_lags,
            lambda_mode="fixed", 
            fixed_lambda=lambda_val, 
            verbose=True
        )
        
        # Extract predictions and align data
        preds = np.array(lasso_results["predictions"])
        y_valid = y[-len(preds):] 
        y_vals = y_valid.values 
        
        # Stage 1
        r2_oos_stage1 = calculate_r_squared(y_vals, preds)
        r2_insample_stage1 = np.mean(lasso_results['insample_r_squareds'])
        
        # Stage 2
        stage2_input = pd.DataFrame({"vwretd": y_vals, "predictions": preds})
        stage2_results = compute_stage2_r_squared(stage2_input, min_train_size=100)
        
        return {
            'window_size': window_size,
            'n_lags': n_lags,
            'lambda': lambda_val,
            'r2_insample_stage1': r2_insample_stage1,
            'r2_oos_stage1': r2_oos_stage1,
            'r2_insample_stage2': stage2_results['r2_insample'],
            'r2_oos_stage2': stage2_results['r2_oos'],
            'kappa': stage2_results['kappa'],
            'kappa_tstat': stage2_results['kappa_tstat'],
            'intercept': stage2_results['intercept'],
            'intercept_tstat': stage2_results['intercept_tstat'],
            'n_observations': len(y_vals),
            'n_windows': len(lasso_results['predictions']),
            'n_oos_predictions_stage2': stage2_results.get('n_oos_predictions', np.nan)
        }
        
    except Exception as e:
        return {
            'window_size': window_size,
            'n_lags': n_lags,
            'lambda': lambda_val,
            'r2_insample_stage1': np.nan,
            'r2_oos_stage1': np.nan,
            'r2_insample_stage2': np.nan,
            'r2_oos_stage2': np.nan,
            'kappa': np.nan,
            'kappa_tstat': np.nan,
            'intercept': np.nan,
            'intercept_tstat': np.nan,
            'n_observations': np.nan,
            'n_windows': np.nan,
            'n_oos_predictions_stage2': np.nan,
            'error': str(e)
        }




def grid_search(X, y, param_grid, verbose=True, n_jobs=-1, backend="loky", prefer=None):
    """
    Parallel grid search using joblib.
    - n_jobs: number of processes (-1 = all cores)
    - backend: 'loky' (processes, default), 'threading', or 'multiprocessing'
    - prefer: 'processes' or 'threads' (optional hint)
    """
    combos = list(product(param_grid['window_sizes'],
                          param_grid['n_lags'],
                          param_grid['lambdas']))
    if verbose:
        print(f"Testing {len(combos)} configurations...")
        print(f"Window sizes: {param_grid['window_sizes']}")
        print(f"N lags: {param_grid['n_lags']}")
        print(f"Lambdas: {param_grid['lambdas']}")

    # Wrapper to catch exceptions so one failure doesn't kill everything
    def _safe_run(args):
        w, L, lam = args
        try:
            return estimate_single_config(X, y, w, L, lam)
        except Exception as e:
            return {"window": w, "n_lags": L, "lambda": lam,
                    "error": str(e), "r2_oos_stage2": float("nan"),
                    "kappa": float("nan")}

    iterator = tqdm(combos, desc="Grid search") if verbose else combos

    results = Parallel(n_jobs=n_jobs, backend=backend, prefer=prefer)(
        delayed(_safe_run)(args) for args in iterator
    )

    results_df = pd.DataFrame(results).sort_values("r2_oos_stage2", ascending=False)

    if verbose:
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        n_failed = results_df["kappa"].isna().sum() if "kappa" in results_df else 0
        if n_failed > 0:
            print(f"⚠️  {n_failed}/{len(results_df)} configurations failed")

    return results_df
