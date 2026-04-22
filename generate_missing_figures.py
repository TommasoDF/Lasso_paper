"""
Generates missing figures for the paper:
1. nonzero_coefficients_over_time.png (Predictability concentration over time)
2. avg_nonzero_coefficients_vs_volatility.png (Selection frequency vs. Return Variance)
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# ── Hardcoded Paths ──────────────────────────────────────────────────────────
RESULTS_DIR = r'C:\Users\jonat\Lasso_paper\Results\Estimation\Cross_Sectional'
H5_PATH = os.path.join(RESULTS_DIR, 'betas.h5')
SUMMARY_PATH = os.path.join(RESULTS_DIR, 'summary.csv')

# ── Style Settings (Matching provided snippet) ───────────────────────────────
# Primary blue: #1f77b4 | Primary red: #d62728
DPI = 150
FACE_COLOR = '#1f77b4'
LINE_COLOR = '#d62728'

def generate_paper_figures():
    # ── Load Data ────────────────────────────────────────────────────────────
    print(f"Loading tensor from {H5_PATH}...")
    if not os.path.exists(H5_PATH):
        print("Error: betas.h5 not found.")
        return

    with h5py.File(H5_PATH, 'r') as f:
        # betas shape: (n_stocks, n_topics, n_dates)
        betas = f['betas'][:] 
        stocks = f['stocks'][:].astype(str)
        dates = pd.to_datetime(f['dates'][:].astype(str))

    # Calculate selection indicator (nonzero coefficients)
    # nan_to_num handles padding/NaNs by treating them as zero (not selected)
    sel = (np.nan_to_num(betas) != 0).astype(np.float32)

    # Active predictors per (stock, day)
    active_per_day = sel.sum(axis=1)  # Resulting shape: (n_stocks, n_dates)

    # Mask dates where no estimation was performed (all NaNs in original betas)
    valid_dates_mask = np.isfinite(betas).any(axis=(0, 1))
    active_valid = active_per_day[:, valid_dates_mask]
    dates_valid = dates[valid_dates_mask]

    # ── FIGURE 1: Active Coefficients over Time ──────────────────────────────
    print("Generating: nonzero_coefficients_over_time.png")
    mean_active = np.mean(active_valid, axis=0)
    p5_active = np.percentile(active_valid, 5, axis=0)
    p95_active = np.percentile(active_valid, 95, axis=0)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(dates_valid, mean_active, color=FACE_COLOR, lw=1.8, label='Mean selected predictors')
    ax1.fill_between(dates_valid, p5_active, p95_active, color=FACE_COLOR, alpha=0.15, label='5th–95th percentile')
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Number of selected predictors', fontsize=11)
    ax1.set_title('Average Number of Active Coefficients over Time', fontsize=13)
    ax1.legend(loc='upper left', fontsize=9, frameon=True)
    ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, 'nonzero_coefficients_over_time.png')
    plt.savefig(out1, dpi=DPI, bbox_inches='tight')
    plt.close()

    # ── FIGURE 2: Selection vs. Volatility ───────────────────────────────────
    print("Generating: avg_nonzero_coefficients_vs_volatility.png")
    if not os.path.exists(SUMMARY_PATH):
        print("Error: summary.csv not found.")
        return

    summary_df = pd.read_csv(SUMMARY_PATH, index_col=0)
    summary_df.index = summary_df.index.astype(str)

    # Cross-sectional average of active coefficients per stock (mean over time)
    # active_valid shape is (stocks, dates)
    stock_avg_active = np.mean(active_valid, axis=1)
    
    # Map average activity to the summary dataframe via stock ticker
    activity_map = pd.Series(stock_avg_active, index=stocks)
    summary_df['avg_active'] = activity_map

    # Per Proposition, use return variance (sigma^2)
    # The notebook summary contains 'stock_vol' (daily return std)
    if 'stock_vol' in summary_df.columns:
        summary_df['return_variance'] = summary_df['stock_vol']**2
    
    # Clean data for plotting
    plot_df = summary_df[['return_variance', 'avg_active']].dropna()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Match style: blue scatter, red regression line
    sns.regplot(data=plot_df, x='return_variance', y='avg_active', 
                ax=ax2, 
                scatter_kws={'alpha': 0.4, 'color': FACE_COLOR, 's': 20},
                line_kws={'color': LINE_COLOR, 'lw': 2, 'label': 'Linear fit'})
    
    ax2.set_xlabel('Return Variance ($\sigma^2_i$)', fontsize=11)
    ax2.set_ylabel('Avg. Number of Non-Zero Coefficients', fontsize=11)
    ax2.set_title('Average Number of Non-Zero Coefficients vs. Return Variance', fontsize=13)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, 'avg_nonzero_coefficients_vs_volatility.png')
    plt.savefig(out2, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"Success. Figures saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    generate_paper_figures()