# =======================
#        IMPORTS
# =======================

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm

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
    Estimate 1st and 2nd stage for a single configuration.
    
    Returns
    -------
    dict with keys:
      - 'summary': dict of scalar metrics
      - 'details': DataFrame of time-series coefficients (Lasso + OLS)
    """
    try:
        # Stage 1: Rolling LASSO + OLS
        stage1_results = lasso_rolling_window(
            X=X, y=y, 
            window_size=window_size, 
            n_lags=n_lags,
            lambda_mode="fixed", 
            fixed_lambda=lambda_val, 
            verbose=False # Keep false to avoid spamming console in parallel
        )
        
        # Extract predictions and align data
        preds = np.array(stage1_results["predictions"])
        y_valid = y[-len(preds):] if isinstance(y, (pd.Series, pd.DataFrame)) else y[-len(preds):]
        y_vals = y_valid.values if isinstance(y_valid, (pd.Series, pd.DataFrame)) else y_valid
        
        # Stage 1 Metrics
        r2_oos_stage1 = calculate_r_squared(y_vals, preds)
        r2_insample_stage1 = np.mean(stage1_results['insample_r_squareds'])
        
        # Stage 2 Estimation
        stage2_input = pd.DataFrame({"vwretd": y_vals, "predictions": preds})
        stage2_results = compute_stage2_r_squared(stage2_input, min_train_size=100)
        
        # --- Build Summary Dictionary ---
        summary = {
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
            'n_windows': len(stage1_results['predictions']),
            'n_oos_predictions_stage2': stage2_results.get('n_oos_predictions', np.nan)
        }
        
        # --- Build Detailed Coefficients DataFrame ---
        # We construct a dataframe where each row is a time window
        n_wins = len(stage1_results['lambdas'])
        dates = stage1_results.get('window_end_dates', np.arange(n_wins))
        features = stage1_results['feature_names']
        
        # We will collect rows as dicts for speed then dataframe them
        detail_rows = []
        
        # Retrieve coefficient arrays (Shape: n_windows x n_features)
        lasso_coefs_arr = stage1_results['coefficients']
        ols_coefs_arr = stage1_results['ols_coefficients']
        
        for i in range(n_wins):
            row = {
                'date': dates[i] if i < len(dates) else None,
                'window_size': window_size,
                'n_lags': n_lags,
                'lambda': lambda_val,
                'window_index': i,
                # Intercepts / R2
                'lasso_intercept': stage1_results['intercepts'][i],
                'ols_intercept': stage1_results['ols_intercepts'][i],
                'lasso_r2_in': stage1_results['insample_r_squareds'][i],
                'ols_r2_in': stage1_results['ols_insample_r2'][i],
            }
            
            # Add feature coefficients
            # Naming convention: Lasso_FeatureName, OLS_FeatureName
            for f_idx, f_name in enumerate(features):
                row[f"Lasso_{f_name}"] = lasso_coefs_arr[i, f_idx]
                row[f"OLS_{f_name}"] = ols_coefs_arr[i, f_idx]
                
            detail_rows.append(row)
            
        details_df = pd.DataFrame(detail_rows)
        
        return {'summary': summary, 'details': details_df}
        
    except Exception as e:
        # Return error dict for summary, empty df for details
        error_summary = {
            'window_size': window_size,
            'n_lags': n_lags,
            'lambda': lambda_val,
            'error': str(e),
            'kappa': np.nan, 
            'r2_oos_stage2': np.nan
        }
        return {'summary': error_summary, 'details': pd.DataFrame()}


def grid_search(X, y, param_grid, verbose=True, n_jobs=-1, backend="loky", prefer=None):
    """
    Parallel grid search.
    
    Returns
    -------
    results_df : pd.DataFrame
        Summary metrics for each configuration.
    coefficients_df : pd.DataFrame
        Detailed Lasso and OLS coefficients for every window of every configuration.
    """
    combos = list(product(param_grid['window_sizes'],
                          param_grid['n_lags'],
                          param_grid['lambdas']))
    if verbose:
        print(f"Testing {len(combos)} configurations...")

    # Wrapper to catch exceptions so one failure doesn't kill everything
    def _safe_run(args):
        w, L, lam = args
        return estimate_single_config(X, y, w, L, lam)

    iterator = tqdm(combos, desc="Grid search") if verbose else combos

    # Run Parallel
    results = Parallel(n_jobs=n_jobs, backend=backend, prefer=prefer)(
        delayed(_safe_run)(args) for args in iterator
    )

    # --- Aggregate Results ---
    summary_list = []
    details_list = []
    
    for res in results:
        if res is not None:
            summary_list.append(res['summary'])
            if not res['details'].empty:
                details_list.append(res['details'])

    # Create DataFrames
    results_df = pd.DataFrame(summary_list)
    if not results_df.empty and 'r2_oos_stage2' in results_df.columns:
        results_df = results_df.sort_values("r2_oos_stage2", ascending=False)
        
    coefficients_df = pd.DataFrame()
    if details_list:
        coefficients_df = pd.concat(details_list, ignore_index=True)

    if verbose:
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        n_failed = results_df["kappa"].isna().sum() if "kappa" in results_df else 0
        if n_failed > 0:
            print(f"⚠️  {n_failed}/{len(results_df)} configurations failed")

    return results_df, coefficients_df
