import os
from pyexpat import features
from pyexpat import features
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from fredapi import Fred

# Time series and preprocessing libraries
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.preprocessing import StandardScaler

def get_repo_root():
    """Finds the repository root containing 'Empirical' and 'Theory' folders."""
    current = Path.cwd().resolve()
    repo_root = next((p for p in [current, *current.parents] if (p / 'Empirical').exists() and (p / 'Theory').exists()), None)
    if repo_root is None:
        raise FileNotFoundError("Could not locate repository root containing 'Empirical' and 'Theory'.")
    return repo_root

def fetch_fred_data(api_key, start_date, end_date):
    """Downloads time-series data from FRED and forward-fills missing values."""
    fred = Fred(api_key=api_key)
    
    series_map = {
        'DFF': 'Fed Funds Effective Rate',
        'DGS10': '10-Year Treasury Yield',
        'VIXCLS': 'VIX Volatility Index',
        'STLFSI4': 'Financial Stress Index',
        'BAMLH0A0HYM2': 'High Yield Option-Adjusted Spread',
        'DTWEXBGS': 'Trade Weighted USD Index',
        'DGS3MO': '3-Month Treasury Yield',
        'DCOILWTICO': 'WTI Crude Oil',
        'T10YIE': '10-Year Breakeven Inflation Rate',
        'DEXJPUS': 'USD to JPY',
    }
    
    print("Connecting to FRED API to download macro data...")
    data = {}
    for series_id, name in series_map.items():
        print(f"  -> Pulling: {name}...")
        try:
            series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            data[name] = series
        except Exception as e:
            print(f"  -> Error pulling {name}: {e}")
            
    df = pd.DataFrame(data)
    
    # Forward-fill to handle weekends/weekly data gaps
    df.ffill(inplace=True)
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)
    
    return df

def extract_best_ar_innovations(features_df, max_lag=10, ic='aic'):
    """
    Finds the optimal AR(p) model for each time series and extracts the residuals.
    
    Parameters:
    - features_df: DataFrame of features including both topics and macro series.
    - max_lag: The maximum number of lags to test (1 to 10).
    - ic: Information criterion to use for selection ('aic' or 'bic').
    """
    print("\nExtracting AR innovations for all features...")
    innovations = pd.DataFrame(index=features_df.index)
    
    # Suppress warnings about index frequencies (common with statsmodels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for col in features_df.columns:
            series = features_df[col].dropna()
            
            # 1. Automatically find the best lag order (p)
            selector = ar_select_order(series, maxlag=max_lag, ic=ic, trend='c')
            best_lags = selector.ar_lags
            
            # Fallback to AR(1) if the algorithm finds no significant lags
            if best_lags is None or len(best_lags) == 0:
                best_lags = [1] 
                
            # 2. Fit the actual AR model using the best lags found
            best_model = AutoReg(series, lags=best_lags, trend='c').fit()
            
            # 3. Extract the residuals (innovations) and assign them back
            innovations[col] = best_model.resid
            print(f"  -> Feature '{col}': Best model is AR({max(best_lags)})")
            
    return innovations


def check_feature_correlation(df, threshold=0.8):
    """
    Calculates the correlation matrix and identifies highly correlated pairs.
    """
    print(f"\nChecking for correlations (Threshold: {threshold})...")
    corr_matrix = df.corr()
    
    # Extract the upper triangle to avoid duplicates (e.g., A-B and B-A) and self-correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs that exceed the threshold
    high_corr = sorted([
        (column, row, upper.loc[row, column]) 
        for column in upper.columns 
        for row in upper.index 
        if abs(upper.loc[row, column]) > threshold
    ], key=lambda x: abs(x[2]), reverse=True)

    if not high_corr:
        print(f"  -> Excellent! No features have a correlation above {threshold}.")
    else:
        print(f"  -> Warning: Found {len(high_corr)} highly correlated pairs:")
        for feat1, feat2, val in high_corr:
            print(f"     * {feat1} & {feat2}: {val:.4f}")
            
    # Print general stats
    avg_corr = upper.abs().mean().mean()
    print(f"  -> Average absolute off-diagonal correlation: {avg_corr:.4f}")
    return high_corr


def main():
    # Configuration
    FRED_API_KEY = 'db7bb3acaf0db1968dfd92d4fc89121d' # Replace/secure as needed
    START_DATE = '1984-01-01'
    END_DATE = '2017-12-31'
    STANDARDIZE_FEATURES = True
    
    repo_root = get_repo_root()
    
    # 1. Download Macro Data
    df_macro = fetch_fred_data(FRED_API_KEY, START_DATE, END_DATE)
    
    # 2. Load Topic Data
    print("\nLoading topic data...")
    topics_path = repo_root / 'Data' / 'data_raw' /'topics.csv'
    topics = pd.read_csv(topics_path)
    
    # Clean topic columns (removing digit-only/unnamed columns based on notebook logic)
    topics = topics[[col for col in topics.columns if not any(char.isdigit() for char in col)]]
    topics.set_index('date', inplace=True)
    topics.index = pd.to_datetime(topics.index)
    
    # 3. Combine Macro and Topic Data
    print("Merging macro and topic datasets...")
    combined_data = df_macro.merge(topics, left_index=True, right_index=True, how='inner')
    
    # 4. Apply AR Innovation Extraction
    features = extract_best_ar_innovations(combined_data)
    
    # Clean up innovations (drop early rows lost to lags, ensure float, drop NaNs)
    features = features.iloc[1:].astype(float).dropna(axis=0, how='any')
    
    # 5. Standardize Features
    if STANDARDIZE_FEATURES:
        print("\nStandardizing features...")
        scaler = StandardScaler()
        features = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
    print(f'Final feature matrix shape: {features.shape}')
    print(f'Average variance after preprocessing: {float(features.var().mean()):.6f}')

    # 5. Standardize Features
    if STANDARDIZE_FEATURES:
        print("\nStandardizing features...")
        scaler = StandardScaler()
        features = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
    
    # --- NEW: Correlation Check ---
    check_feature_correlation(features, threshold=0.2) 
    # ------------------------------

    print(f'\nFinal feature matrix shape: {features.shape}')
    
    output_path = repo_root / 'Data' / 'clean_data' / 'final_macro_topic_features.csv'
    
    # Create the directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    features.to_csv(output_path)
    print(f"\nSuccess! Final features saved to: {output_path}")

if __name__ == "__main__":
    main()