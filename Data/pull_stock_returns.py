import wrds
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================
NOTEBOOK_DIR = Path('.').resolve()
REPO_ROOT    = NOTEBOOK_DIR.parent.parent
DATA_DIR     = REPO_ROOT / 'Data'
OUTPUT_FILE  = DATA_DIR / 'data_clean' / 'returns_wide.csv'

WRDS_USERNAME = 'jonathanseim'

START_DATE = '1984-01-01'
END_DATE   = '2017-12-31'
EXCHANGES  = ["NYSE"]
MIN_YEARS  = 5

# ============================================================
# 2. CONNECT & FETCH
# ============================================================
exchcd_map     = {'NYSE': 1, 'AMEX': 2, 'NASDAQ': 3}
exchcd_list    = [exchcd_map[ex.upper()] for ex in EXCHANGES]
exchcd_sql_str = ", ".join(map(str, exchcd_list))

print(f"Connecting to WRDS as {WRDS_USERNAME}...")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

sql_query = f"""
    SELECT
        a.DlyCalDt AS date,
        a.permno   AS permno,
        a.DlyRet   AS return
    FROM crsp.dsf_v2 AS a
    LEFT JOIN crsp.dsenames AS b
        ON  a.permno = b.permno
        AND a.DlyCalDt >= b.namedt
        AND a.DlyCalDt <= b.nameendt
    WHERE a.DlyCalDt >= '{START_DATE}'
      AND a.DlyCalDt <= '{END_DATE}'
      AND b.exchcd IN ({exchcd_sql_str})
"""

print("Executing query...")
stock_data = db.raw_sql(sql_query, date_cols=['date'])
db.close()
print(f"Downloaded {len(stock_data):,} rows.")

# ============================================================
# 3. CLEAN
# ============================================================
stock_data['return'] = pd.to_numeric(stock_data['return'], errors='coerce')
stock_data['permno'] = pd.to_numeric(stock_data['permno'],  errors='coerce').astype('Int64')
stock_data = (
    stock_data
    .dropna(subset=['return', 'permno'])
    .drop_duplicates(subset=['permno', 'date'])
    .sort_values(['permno', 'date'])
    .reset_index(drop=True)
)

# ============================================================
# 4. MIN-YEARS CONSECUTIVE FILTER
# ============================================================
print(f"Filtering for >= {MIN_YEARS} consecutive years...")

months_df = stock_data[['permno', 'date']].copy()
months_df['month_period'] = months_df['date'].dt.to_period('M')
months_df = months_df.drop_duplicates(subset=['permno', 'month_period']).sort_values(['permno', 'month_period'])

months_df['month_diff'] = (
    months_df.groupby('permno')['month_period']
    .diff()
    .apply(lambda x: x.n if pd.notnull(x) else 0)
)
months_df['streak_id'] = (months_df['month_diff'] != 1).cumsum()

max_streaks   = months_df.groupby(['permno', 'streak_id']).size().groupby('permno').max()
valid_permnos = max_streaks[max_streaks >= MIN_YEARS * 12].index

stock_data = stock_data[stock_data['permno'].isin(valid_permnos)].copy()
print(f"{len(valid_permnos):,} stocks passed the filter.")

# ============================================================
# 5. PIVOT TO WIDE  →  rows = date, columns = permno
# ============================================================
print("Pivoting to wide format...")

returns_wide = (
    stock_data
    .pivot(index='date', columns='permno', values='return')
)
returns_wide.index.name   = 'date'
returns_wide.columns.name = None
returns_wide.columns      = returns_wide.columns.astype(int)   # permno as plain int

print(f"Wide matrix shape: {returns_wide.shape}  (dates × permnos)")

# ============================================================
# 6. EXPORT
# ============================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
returns_wide.to_csv(OUTPUT_FILE)          # date as row index, permnos as header
print(f"Saved → {OUTPUT_FILE}")