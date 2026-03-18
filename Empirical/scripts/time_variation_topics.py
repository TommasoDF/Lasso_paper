import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from get_repo_root import get_repo_root

# --- 1. Path & Environment Setup ---

# Setup Directories
REPO_ROOT = get_repo_root()

# Directory containing the betas.h5 and summary.csv files
DATA_DIR = REPO_ROOT / "Results" / "Estimation" / "Cross_Sectional" 
CODES_FILE = REPO_ROOT / "Data" / "data_raw" / "industry_codes.csv"
SAVE_DIR = Path(r"C:\Users\jonat\Lasso_paper\Results\Figures")

# Ensure the save directory exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Journal of Finance Style Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', # LaTeX-like math font
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 300
})
sns.set_style("white")

# --- 2. Load Data ---

h5_path = DATA_DIR / 'betas.h5'
csv_path = DATA_DIR / 'summary.csv'

with h5py.File(h5_path, 'r') as f:
    betas  = f['betas'][:]               # (n_stocks, n_topics, n_dates) float32
    stocks = f['stocks'][:].astype(str)
    topics = f['topics'][:].astype(str)
    dates  = pd.to_datetime(f['dates'][:].astype(str))

summary_df = pd.read_csv(csv_path, index_col=0)
summary_df.index = summary_df.index.astype(str)

# --- 3. Derived quantities & numerical values ---

# NaN entries mean the stock had no data at that date; treat as not selected
betas_nz = np.nan_to_num(betas, nan=0.0)
sel      = betas_nz != 0                 # (n_stocks, n_topics, n_dates) bool

has_data     = np.isfinite(betas).any(axis=1)                      # (n_stocks, n_dates)
n_valid_days = has_data.sum(axis=1, keepdims=True).astype(float)   # (n_stocks, 1)

# Selection frequency per (stock, topic): fraction of valid days topic was selected
sel_mat = sel.sum(axis=2) / np.maximum(n_valid_days, 1)  # (n_stocks, n_topics)

# Active topics per (stock, date) — NaN where stock has no data
active_per_day = sel.sum(axis=1).astype(float)   # (n_stocks, n_dates)
active_per_day[~has_data] = np.nan

# Per-stock mean active topics over time
with np.errstate(invalid='ignore'): # Ignore mean of empty slice warnings
    mean_num_active = np.nanmean(active_per_day, axis=1)  # (n_stocks,)

summary_df['mean_num_active'] = pd.Series(mean_num_active, index=stocks)

# Filtered selection matrix: only non-failed stocks (at least one selection)
valid_mask = mean_num_active > 0
n_non_failed = valid_mask.sum()

print("--- Data Summary ---")
print(f"Tensor shape : {betas.shape}")
print(f"Stocks       : {len(stocks)} ({n_non_failed} non-failed)")
print(f"Topics       : {len(topics)}")
print(f"Dates        : {len(dates)} ({dates[0].date()} -> {dates[-1].date()})")

n_significant = (summary_df['kappa_tstat'] > 1.96).sum()
print(f"\nNumber of stocks with kappa t-stat > 1.96: {n_significant} out of {len(summary_df)}")

# =========================================================
# "Is topic k on day t selected only by one stock or many?"
# Selection breadth across stocks for each (topic, day)
# =========================================================
n_stocks, n_topics, n_dates = sel.shape

breadth_counts = sel.sum(axis=0)               # shape: (n_topics, n_dates)
breadth_share = breadth_counts / n_non_failed  # fraction of stocks selecting topic k at date t

# Long table for inspection / filtering
breadth_df = (
    pd.DataFrame(
        {
            "topic": np.repeat(topics, n_dates),
            "date": np.tile(dates, n_topics),
            "n_selected_stocks": breadth_counts.reshape(-1),
            "share_selected_stocks": breadth_share.reshape(-1),
        }
    )
    .sort_values(["share_selected_stocks", "n_selected_stocks"], ascending=False)
    .reset_index(drop=True)
)

print("\n--- Top 20 most broadly selected (topic, date) pairs ---")
print(breadth_df.head(20).to_string())

# --- 4. Plot Generation & Saving ---
print(f"\nGenerating and saving plots to: {SAVE_DIR} ...")

# -----------------------------
# Plot 1: Days in Sample a Topic Was Selected (at least one stock)
# -----------------------------
topic_active_days = (breadth_counts > 0).sum(axis=1)

# Sorting for a cleaner bar plot (top 40 most active topics)
top_topics_idx = np.argsort(topic_active_days)[::-1][:40] 

plt.figure(figsize=(10, 10))
sns.barplot(x=topic_active_days[top_topics_idx], y=topics[top_topics_idx], orient="h", color="#4C72B0")
plt.title("Days in Sample a Predictor Was Selected (at least one stock) - Top 40 Predictors")
plt.xlabel("Number of days predictor is active")
plt.ylabel("Predictor")
plt.tight_layout()
plt.savefig(SAVE_DIR / "predictor_active_days.png")
plt.close()

# -----------------------------
# Plot 2: Conditional on topic being selected, for how many stocks is it active on a day?
# -----------------------------
active_stock_counts_cond = breadth_counts[breadth_counts > 0].ravel()

plt.figure(figsize=(10, 5))
sns.histplot(active_stock_counts_cond, bins=40, edgecolor="black", color="#4C72B0")
plt.axvline(active_stock_counts_cond.mean(), color="red", linestyle="--",
            label=f"Mean = {active_stock_counts_cond.mean():.1f} stocks")
plt.title("Active Stocks per Predictor-Day | Conditional on Predictor Being Selected")
plt.xlabel("Number of stocks where predictor is active on that day")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "active_stocks_cond_hist.png")
plt.close()

print("Script execution completed.")