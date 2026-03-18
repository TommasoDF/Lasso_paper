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
    calculate_r_squared,
    lasso_rolling_window_fast
)

from stage2 import compute_stage2_r_squared

# =======================
#      FUNCTIONS
# =======================

def estimate_single_config(X, y, window_size, n_lags, lambda_val, standardize=True, verbose=False, save_only_positve_r_squared = True, return_details=True):
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
            verbose=False, # Keep false to avoid spamming console in parallel
            standardize=standardize
        )
       
        
        # Extract predictions and align data
        preds = np.array(stage1_results["predictions"]) # contains predictions up to r^e_{t+2} while y contains up to r_{t} so we need to align
     
        #shift preds to match y
        preds = preds[:-2] # remove last two prediction (nans) to align with y, now it contains up to r^e_{t}
        y_valid = y[-len(preds)-1:-1] #this contains up until r_{t-1} to align with preds
        y_vals = y_valid.values if isinstance(y_valid, (pd.Series, pd.DataFrame)) else y_valid
        
        #defining de-meaned y for comparing with stage 1 errors
        y_demeaned = y_vals - np.nanmean(y_vals)
        
        # Stage 1 Metrics
        r2_oos_stage1 = calculate_r_squared(y_demeaned, preds)
        r2_insample_stage1 = np.nanmean(stage1_results['insample_r_squareds'])

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

        if not return_details:
            return {"summary": summary, "details": pd.DataFrame()}

        
        # --- Build Detailed Coefficients DataFrame ---
        # We construct a dataframe where each row is a time window
        n_wins = len(stage1_results['lambdas'])
        dates = stage1_results.get('window_end_dates', np.arange(n_wins))
        features = stage1_results['feature_names']
        # We will collect rows as dicts for speed then dataframe them
        detail_rows = []
        
        if save_only_positve_r_squared and summary['r2_insample_stage2'] <=0:
            # If insample R2 is not positive, save only NaNs
            detail_rows.append({
                'date': None,
                'window_size': window_size,
                'n_lags': n_lags,
                'lambda': lambda_val,
                'window_index': None,
                'lasso_intercept': np.nan,
                # 'ols_intercept': np.nan,
                'lasso_r2_in': np.nan,
                'prediction_error': np.nan,
                # 'ols_r2_in': np.nan,
                'num_nonzero_coefficients': np.nan,
                'prediction': np.nan, # NEW
                'target': np.nan      # NEW
            })
        else:
            lasso_coefs_arr = stage1_results['coefficients']
            # ols_coefs_arr = stage1_results['ols_coefficients']
            
            # Pad the predictions and targets with NaNs so they match n_wins length
            pad_len = n_wins - len(stage1_results['predictions'])
            if pad_len > 0:
                padded_preds = np.pad(stage1_results['predictions'], (0, pad_len), constant_values=np.nan)
                padded_targets = np.pad(stage1_results['targets'], (0, pad_len), constant_values=np.nan)
            else:
                padded_preds = stage1_results['predictions']
                padded_targets = stage1_results['targets']
        
            for i in range(n_wins):
                row = {
                    'date': dates[i] if i < len(dates) else None,
                    'window_size': window_size,
                    'n_lags': n_lags,
                    'lambda': lambda_val,
                    'window_index': i,
                    # Intercepts / R2
                    'lasso_intercept': stage1_results['intercepts'][i],
                    # 'ols_intercept': stage1_results['ols_intercepts'][i],
                    'lasso_r2_in': stage1_results['insample_r_squareds'][i],
                    # 'ols_r2_in': stage1_results['ols_insample_r2'][i],
                    'num_nonzero_coefficients': stage1_results['num_nonzero_coefficients'][i],
                    'prediction': padded_preds[i], # NEW
                    'target': padded_targets[i]    # NEW
                }

                # Add feature coefficients
                # Naming convention: Lasso_FeatureName, OLS_FeatureName
                for f_idx, f_name in enumerate(features):
                    row[f"Lasso_{f_name}"] = lasso_coefs_arr[i, f_idx]
                    # row[f"OLS_{f_name}"] = ols_coefs_arr[i, f_idx]

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



def estimate_single_config_fast(X, y, window_size, n_lags, lambda_val, standardize=True, verbose=False, save_only_positve_r_squared=False, return_details=True):
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
        stage1_results = lasso_rolling_window_fast(
            X=X, y=y, 
            window_size=window_size, 
            n_lags=n_lags,
            lambda_mode="fixed", 
            fixed_lambda=lambda_val, 
            verbose=False,
            standardize=standardize
        )
       
        # Extract predictions and align data
        preds = np.array(stage1_results["predictions"])[:-2]
        y_valid = y[-len(preds)-1:-1]
        y_vals = y_valid.values if isinstance(y_valid, (pd.Series, pd.DataFrame)) else y_valid
        
        # Defining de-meaned y for comparing with stage 1 errors
        y_demeaned = y_vals - np.nanmean(y_vals)

        #drop first element form preds and y_demeaned (nan in preds)
        preds = preds[1:]
        y_demeaned = y_demeaned[1:]
        y_vals = y_vals[1:]
        # Stage 1 Metrics
        r2_oos_stage1 = calculate_r_squared(y_demeaned, preds)
        
        r2_insample_stage1 = np.nanmean(stage1_results['insample_r_squareds'])

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

        if not return_details:
            return {"summary": summary, "details": pd.DataFrame()}

        # --- Build Detailed Coefficients DataFrame ---
        n_wins = len(stage1_results['lambdas'])
        dates = stage1_results.get('window_end_dates', np.arange(n_wins))
        features = stage1_results['feature_names']
        
        if save_only_positve_r_squared and summary['r2_insample_stage2'] <= 0:
            # Return minimal DataFrame with NaNs
            detail_rows = [{
                'date': None,
                'window_size': window_size,
                'n_lags': n_lags,
                'lambda': lambda_val,
                'window_index': None,
                'lasso_intercept': np.nan,
                'lasso_r2_in': np.nan,
                'prediction_error': np.nan,
                'num_nonzero_coefficients': np.nan,
                'prediction': np.nan, # NEW
                'target': np.nan      # NEW
            }]
            details_df = pd.DataFrame(detail_rows)
            return {'summary': summary, 'details': details_df}
        else:
            lasso_coefs_arr = stage1_results['coefficients']
            
            # Align lengths by padding predictions/targets with NaNs
            pad_len = n_wins - len(stage1_results['predictions'])
            if pad_len > 0:
                padded_preds = np.pad(stage1_results['predictions'], (0, pad_len), constant_values=np.nan)
                padded_targets = np.pad(stage1_results['targets'], (0, pad_len), constant_values=np.nan)
            else:
                padded_preds = stage1_results['predictions']
                padded_targets = stage1_results['targets']
            
            # Build base DataFrame efficiently using numpy arrays
            base_data = {
                'date': dates if len(dates) == n_wins else list(dates) + [None] * (n_wins - len(dates)),
                'window_size': window_size,
                'n_lags': n_lags,
                'lambda': lambda_val,
                'window_index': np.arange(n_wins),
                'lasso_intercept': stage1_results['intercepts'],
                'lasso_r2_in': stage1_results['insample_r_squareds'],
                'num_nonzero_coefficients': stage1_results['num_nonzero_coefficients'],
                'prediction': padded_preds, # NEW
                'target': padded_targets    # NEW
            }
            
            # Add feature coefficients as columns directly from numpy array
            for f_idx, f_name in enumerate(features):
                base_data[f"Lasso_{f_name}"] = lasso_coefs_arr[:, f_idx]
            
            details_df = pd.DataFrame(base_data)
            return {'summary': summary, 'details': details_df}
            
    except Exception as e:
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