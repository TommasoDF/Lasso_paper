"""
WRDS CRSP Industry Codes (SIC) Fetcher

Pulls historical PERMNO -> header SIC code (HSICCD) mapping from CRSP,
needed to classify stocks into Fama-French 12 industries for the
firm-level uncertainty robustness analysis.
"""

import os
import wrds
import pandas as pd

OUTPUT_DIR = 'Data/data_raw'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'industry_codes.csv')


def fetch_crsp_industry_codes():
    print("Establishing connection to WRDS...")
    db = wrds.Connection(wrds_username='tommasodifrance')

    print("Querying CRSP stocknames table for HSICCD...")
    sql_query = """
        SELECT
            permno,
            siccd,
            nameenddt AS date
        FROM
            crsp.stocknames
        WHERE
            siccd IS NOT NULL
    """
    df = db.raw_sql(sql_query)
    print(f"Retrieved {len(df):,} historical SIC records.")

    df.columns = [col.upper() for col in df.columns]
    df.rename(columns={'DATE': 'date'}, inplace=True)
    df['PERMNO'] = df['PERMNO'].fillna(0).astype(int)
    df['SICCD'] = df['SICCD'].fillna(0).astype(int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)

    db.close()
    print("\nSuccess! industry_codes.csv has been created.")


if __name__ == "__main__":
    fetch_crsp_industry_codes()
