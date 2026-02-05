"""
Simplified Lasso Rolling Window Implementation (Updated with OLS)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Tuple, Optional, Dict, Any, List
from tqdm import tqdm


def to_ar1_innovations(X: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Return AR(1) innovations (residuals) for each column of X."""
    X_innov = pd.DataFrame(index=X.index, columns=X.columns, dtype="float64")

    for col in X.columns:
        s = pd.to_numeric(X[col], errors="coerce")
        tmp = pd.DataFrame({"x": s, "x_lag1": s.shift(1)}).dropna()
        if len(tmp) < min_obs or tmp["x"].nunique() < 3 or tmp["x_lag1"].nunique() < 3:
            continue

        res = sm.OLS(tmp["x"], sm.add_constant(tmp["x_lag1"])).fit()
        X_innov.loc[tmp.index, col] = res.resid

    return X_innov

def create_lagged_features(X: np.ndarray, n_lags: int = 3):
    if n_lags == 0:
        return X, [f'feature_{i}' for i in range(X.shape[1])]

    n_samples, n_features = X.shape

    # Create lagged blocks in the correct order: lag 1, lag 2, ...
    lagged_features = [
        X[n_lags - lag : n_samples - lag] 
        for lag in range(1, n_lags + 1)
    ]

    feature_names = [
        f'feature_{i}_lag_{lag}'
        for lag in range(1, n_lags + 1)
        for i in range(n_features)
    ]

    return np.hstack(lagged_features), feature_names

def create_lagged_features_with_name(X, n_lags: int = 3):
    """
    Create lagged feature matrix and names.
    Preserves original column names and appends _lag_{lag_number}.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        If DataFrame, uses X.columns as feature names.
        If ndarray, feature names default to feature_{i}.
    n_lags : int
        Number of lags.

    Returns
    -------
    X_lagged : np.ndarray
    feature_names : list[str]
    """
    # Infer names
    if hasattr(X, "columns"):
        base_names = list(X.columns)
        X_values = X.to_numpy()
    else:
        X_values = np.asarray(X)
        base_names = [f"feature_{i}" for i in range(X_values.shape[1])]

    if n_lags == 0:
        return X_values, base_names

    n_samples, n_features = X_values.shape

    # lag 1, lag 2, ...
    lagged_blocks = [
        X_values[n_lags - lag : n_samples - lag]
        for lag in range(1, n_lags + 1)
    ]
    X_lagged = np.hstack(lagged_blocks)

    feature_names = [
        f"{base}_lag_{lag_number}"
        for lag_number in range(1, n_lags + 1)
        for base in base_names
    ]

    return X_lagged, feature_names

def lasso_rolling_window(X: np.ndarray,
                         y: np.ndarray,
                         window_size: int = 30,
                         n_lags: int = 3,
                         burn_in: Optional[int] = None,
                         standardize: bool = True,
                         cv_folds: int = 5,
                         alphas: Optional[np.ndarray] = None,
                         verbose: bool = True,
                         lambda_mode: str = "cv",
                         fixed_lambda: Optional[float] = None) -> Dict[str, Any]:
    
    # Extract date index if pandas object
    date_index = y.index if isinstance(y, (pd.Series, pd.DataFrame)) else None
    y = np.asarray(y).flatten()
    #X = np.asarray(X)
    
    # Center returns
    mean_y = np.nanmean(y)
    y = y - mean_y
    
    # Create lagged features
    if verbose:
        print(f"Creating lagged features with {n_lags} lags...")
    
    X_lagged, feature_names = create_lagged_features_with_name(X, n_lags)
    y_aligned = y[n_lags:]

    if date_index is not None:
        date_index = date_index[n_lags:]
    
    # Safety Checks
    if X_lagged.shape[0] != y_aligned.shape[0]:
        raise ValueError("Mismatch between lagged X and aligned y observations.")
    
    burn_in = burn_in or window_size
    if len(y_aligned) < burn_in:
        raise ValueError("Insufficient data for window size and lags.")
    
    if lambda_mode not in {"cv", "fixed"}:
        raise ValueError("lambda_mode must be 'cv' or 'fixed'.")
    
    if lambda_mode == "fixed" and (fixed_lambda is None or fixed_lambda <= 0):
        raise ValueError("lambda_mode='fixed' requires positive fixed_lambda.")
    

    # Initialize storage for results
    results_data = {
        # LASSO Storage
        'lambdas': [], 'coefficients': [], 'intercepts': [], 'insample_r_squareds': [], 'num_nonzero_coefficients': [],
        'window_starts': [], 'window_ends': [], 'predictions': [], 'targets': []
    }
    
    
    if date_index is not None:
        results_data.update({
            'window_start_dates': [], 'window_end_dates': [], 'prediction_dates': []
        })
    

    # Rolling window loop
    n_windows = len(y_aligned) - window_size + 1
    if verbose:
        mode_str = f"cross-validation" if lambda_mode == "cv" else f"fixed (α={fixed_lambda})"
        print(f"Running {n_windows} rolling windows of size {window_size}...")
        print(f"Lambda selection: {mode_str}")
    
    for i in tqdm(range(burn_in - window_size, n_windows), desc="Rolling windows", disable=not verbose):
        start_idx, end_idx = max(0, i), max(0, i) + window_size # this way we selct data from 0 to window_size-1 in first iteration
        X_window = X_lagged[start_idx:end_idx]
        y_window = y_aligned[start_idx:end_idx]
        
        if standardize:
            scaler = StandardScaler()
            X_window_scaled = scaler.fit_transform(X_window)
        else:
            scaler = None
            X_window_scaled = X_window
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # --- 1. LASSO FIT ---
                if lambda_mode == "cv":
                    model = LassoCV(
                        cv=min(cv_folds, max(2, window_size // 2)),
                        max_iter=2000,
                        n_alphas=None if alphas is not None else 100,
                        alphas=alphas,
                        fit_intercept=False,
                        random_state=42
                    )
                else:
                    model = Lasso(
                        alpha=float(fixed_lambda),
                        max_iter=2000,
                        fit_intercept=False,
                        random_state=42
                    )
                
                model.fit(X_window_scaled, y_window)
                
                # Store Lasso results
                chosen_alpha = float(model.alpha_) if lambda_mode == "cv" else float(fixed_lambda)
                results_data['lambdas'].append(chosen_alpha)
                results_data['intercepts'].append(model.intercept_)
                results_data['insample_r_squareds'].append(model.score(X_window_scaled, y_window))

                #Store the number of non-zero coefficients
                results_data['num_nonzero_coefficients'].append(np.sum(model.coef_ != 0))
                
                # Transform coefficients back to original scale
                coef = model.coef_ / scaler.scale_ if standardize else model.coef_
                results_data['coefficients'].append(coef)

               
                if date_index is not None:
                    results_data['window_start_dates'].append(date_index[start_idx])
                    results_data['window_end_dates'].append(date_index[end_idx - 1])
                
                # Out-of-sample prediction (Using LASSO as primary predictor)
                if end_idx < len(y_aligned):
                    X_next = X_lagged[end_idx + 2].reshape(1, -1) # shift x by two step-ahead to compute beta_{t+1} x_{t+1}
                    X_next_scaled = scaler.transform(X_next) if standardize else X_next
                    pred = model.predict(X_next_scaled)[0]
                    results_data['predictions'].append(pred)
                    target = y_aligned[end_idx + 2]
                    results_data['targets'].append(target)
                    
                    if date_index is not None:
                        results_data['prediction_dates'].append(date_index[end_idx + 2])
        except Exception as e:
            if verbose:
                print(f"Warning: Fitting failed for window {i}: {e}")
            
            # Store NaNs on failure (for both Lasso and OLS)
            results_data['lambdas'].append(np.nan)
            results_data['coefficients'].append(np.full(X_lagged.shape[1], np.nan))
            results_data['intercepts'].append(np.nan)
            results_data['insample_r_squareds'].append(np.nan)
            results_data['num_nonzero_coefficients'].append(np.nan)
            results_data['window_starts'].append(start_idx + n_lags)
            results_data['window_ends'].append(end_idx + n_lags)
            
            if date_index is not None:
                results_data['window_start_dates'].append(date_index[start_idx])
                results_data['window_end_dates'].append(date_index[min(end_idx - 1, len(date_index) - 1)])
            
            if end_idx < len(y_aligned):
                results_data['predictions'].append(np.nan)
                results_data['targets'].append(np.nan)

                if date_index is not None:
                    results_data['prediction_dates'].append(date_index[min(end_idx, len(date_index) - 1)])
    
    if verbose:
        print("Rolling Estimation complete!")
    
    # Compile final results
    results = {
        'lambdas': np.array(results_data['lambdas']),
        'coefficients': np.array(results_data['coefficients']),
        'intercepts': np.array(results_data['intercepts']),
        'insample_r_squareds': np.array(results_data['insample_r_squareds']),
        'num_nonzero_coefficients': np.array(results_data['num_nonzero_coefficients']),
        'targets': np.array(results_data['targets']),
        'window_starts': np.array(results_data['window_starts']),
        'window_ends': np.array(results_data['window_ends']),
        'feature_names': feature_names,
        'predictions': np.array(results_data['predictions']) + mean_y,
        'n_lags': n_lags,
        'window_size': window_size,
        'lambda_mode': lambda_mode
    }
    
    
    if date_index is not None:
        results.update({
            'window_start_dates': np.array(results_data['window_start_dates']),
            'window_end_dates': np.array(results_data['window_end_dates']),
            'prediction_dates': np.array(results_data['prediction_dates'])
        })
    
    return results

def lasso_rolling_window_fast(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 30,
    n_lags: int = 3,
    burn_in: Optional[int] = None,
    standardize: bool = True,
    cv_folds: int = 5,                 # kept for signature compatibility; unused
    alphas: Optional[np.ndarray] = None, # kept for signature compatibility; unused
    verbose: bool = True,
    lambda_mode: str = "fixed",         # kept for signature compatibility; forced to fixed
    fixed_lambda: Optional[float] = None
) -> Dict[str, Any]:
    """
    Fast version of lasso_rolling_window:
      - No cross-validation
      - Fixed lambda (alpha) only
      - Uses warm_start by reusing the same Lasso estimator across windows
      - Otherwise mirrors lasso_rolling_window behaviour and outputs
    """

    # Extract date index if pandas object
    date_index = y.index if isinstance(y, (pd.Series, pd.DataFrame)) else None
    y = np.asarray(y).flatten()

    # Center returns
    mean_y = np.nanmean(y)
    y = y - mean_y

    # Create lagged features
    if verbose:
        print(f"Creating lagged features with {n_lags} lags...")

    X_lagged, feature_names = create_lagged_features_with_name(X, n_lags)
    y_aligned = y[n_lags:]

    if date_index is not None:
        date_index = date_index[n_lags:]

    # Safety checks
    if X_lagged.shape[0] != y_aligned.shape[0]:
        raise ValueError("Mismatch between lagged X and aligned y observations.")

    burn_in = burn_in or window_size
    if len(y_aligned) < burn_in:
        raise ValueError("Insufficient data for window size and lags.")

    if fixed_lambda is None or fixed_lambda <= 0:
        raise ValueError("lasso_rolling_window_fast requires positive fixed_lambda (no CV).")

    # Initialize storage for results
    results_data = {
        'lambdas': [],
        'coefficients': [],
        'intercepts': [],
        'insample_r_squareds': [],
        'num_nonzero_coefficients': [],
        'window_starts': [],
        'window_ends': [],
        'predictions': [],
        'targets': []
    }

    if date_index is not None:
        results_data.update({
            'window_start_dates': [],
            'window_end_dates': [],
            'prediction_dates': []
        })

    # Rolling window loop setup
    n_windows = len(y_aligned) - window_size + 1
    if verbose:
        print(f"Running {n_windows} rolling windows of size {window_size}...")
        print(f"Lambda selection: fixed (α={fixed_lambda})")

    # Reuse a single estimator for warm starts
    model = Lasso(
        alpha=float(fixed_lambda),
        max_iter=2000,
        fit_intercept=False,
        warm_start=True,
        random_state=42
    )

    for i in tqdm(range(burn_in - window_size, n_windows),
                  desc="Rolling windows", disable=not verbose):
        start_idx, end_idx = max(0, i), max(0, i) + window_size
        X_window = X_lagged[start_idx:end_idx]
        y_window = y_aligned[start_idx:end_idx]

        if standardize:
            scaler = StandardScaler()
            X_window_scaled = scaler.fit_transform(X_window)
        else:
            scaler = None
            X_window_scaled = X_window

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Fit with warm-start (same estimator object reused)
                model.alpha = float(fixed_lambda)
                model.fit(X_window_scaled, y_window)

                # Store results
                results_data['lambdas'].append(float(fixed_lambda))
                results_data['intercepts'].append(model.intercept_)
                results_data['insample_r_squareds'].append(model.score(X_window_scaled, y_window))
                results_data['num_nonzero_coefficients'].append(np.sum(model.coef_ != 0))

                # Transform coefficients back to original scale (if standardised)
                coef = model.coef_ / scaler.scale_ if standardize else model.coef_
                results_data['coefficients'].append(coef)

                # Window indices (kept consistent with your failure block intent)
                results_data['window_starts'].append(start_idx + n_lags)
                results_data['window_ends'].append(end_idx + n_lags)

                if date_index is not None:
                    results_data['window_start_dates'].append(date_index[start_idx])
                    results_data['window_end_dates'].append(date_index[end_idx - 1])

                # Out-of-sample prediction (mirrors your +2 indexing)
                if end_idx + 2 < len(y_aligned):
                    X_next = X_lagged[end_idx + 2].reshape(1, -1)
                    X_next_scaled = scaler.transform(X_next) if standardize else X_next
                    pred = model.predict(X_next_scaled)[0]
                    results_data['predictions'].append(pred)
                    target = y_aligned[end_idx + 2]
                    results_data['targets'].append(target)

                    if date_index is not None:
                        results_data['prediction_dates'].append(date_index[end_idx + 2])

        except Exception as e:
            if verbose:
                print(f"Warning: Fitting failed for window {i}: {e}")

            results_data['lambdas'].append(np.nan)
            results_data['coefficients'].append(np.full(X_lagged.shape[1], np.nan))
            results_data['intercepts'].append(np.nan)
            results_data['insample_r_squareds'].append(np.nan)
            results_data['num_nonzero_coefficients'].append(np.nan)
            results_data['window_starts'].append(start_idx + n_lags)
            results_data['window_ends'].append(end_idx + n_lags)

            if date_index is not None:
                results_data['window_start_dates'].append(date_index[start_idx])
                results_data['window_end_dates'].append(date_index[min(end_idx - 1, len(date_index) - 1)])

            if end_idx + 2 < len(y_aligned):
                results_data['predictions'].append(np.nan)
                results_data['targets'].append(np.nan)
                if date_index is not None:
                    results_data['prediction_dates'].append(date_index[end_idx + 2])

    if verbose:
        print("Rolling Estimation complete!")

    results = {
        'lambdas': np.array(results_data['lambdas']),
        'coefficients': np.array(results_data['coefficients']),
        'intercepts': np.array(results_data['intercepts']),
        'insample_r_squareds': np.array(results_data['insample_r_squareds']),
        'num_nonzero_coefficients': np.array(results_data['num_nonzero_coefficients']),
        'targets': np.array(results_data['targets']),
        'window_starts': np.array(results_data['window_starts']),
        'window_ends': np.array(results_data['window_ends']),
        'feature_names': feature_names,
        'predictions': np.array(results_data['predictions']) + mean_y,
        'n_lags': n_lags,
        'window_size': window_size,
        'lambda_mode': 'fixed'
    }

    if date_index is not None:
        results.update({
            'window_start_dates': np.array(results_data['window_start_dates']),
            'window_end_dates': np.array(results_data['window_end_dates']),
            'prediction_dates': np.array(results_data['prediction_dates'])
        })

    return results



def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared."""
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)