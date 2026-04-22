"""
WRDS CRSP Stock Names Fetcher

Pulls the complete historical universe of PERMNO to Ticker mappings from CRSP
and formats it to replace the local stock_names.csv file.
"""

import os
import wrds
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
# Set this to wherever your pipeline expects the CSV to live
OUTPUT_DIR = 'Data/clean_data' 
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'stock_names.csv')

def fetch_crsp_stocknames():
    print("Establishing connection to WRDS...")
    # Note: On first run, this will prompt for your WRDS username and password.
    # It will offer to create a .pgpass file to store credentials for future automated runs.
    try:
        db = wrds.Connection()
    except Exception as e:
        print(f"Failed to connect to WRDS. Ensure your credentials are correct. Error: {e}")
        return

    print("Querying CRSP stocknames table...")
    # We pull 'nameendt' (Name End Date) and alias it as 'date'.
    # This ensures your downstream script's logic: .sort_values('date').groupby('PERMNO').last() 
    # continues to work perfectly to grab the most recent ticker.
    sql_query = """
        SELECT 
            permno, 
            ticker, 
            comnam, 
            nameenddt AS date
        FROM 
            crsp.stocknames
        WHERE 
            ticker IS NOT NULL
    """
    
    # Execute the query
    df = db.raw_sql(sql_query)
    
    print(f"Retrieved {len(df):,} historical ticker records.")

    # ── Formatting ────────────────────────────────────────────────────────────
    print("Formatting dataframe to match pipeline expectations...")
    
    # Capitalize columns to match your existing CSV (PERMNO, TICKER, COMNAM)
    df.columns = [col.upper() for col in df.columns]
    
    # Force the date column back to lowercase to match your pandas sort_values('date') call
    df.rename(columns={'DATE': 'date'}, inplace=True)
    
    # Clean up the PERMNOs (WRDS sometimes returns them as floats, e.g., 10001.0)
    df['PERMNO'] = df['PERMNO'].fillna(0).astype(int).astype(str)
    
    # Clean up Tickers (strip any trailing whitespace)
    df['TICKER'] = df['TICKER'].astype(str).str.strip()

    # ── Saving ────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Close the WRDS connection
    db.close()
    
    print("\nSuccess! Your stock_names.csv has been fully updated.")

if __name__ == "__main__":
    fetch_crsp_stocknames()