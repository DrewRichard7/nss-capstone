#!/usr/bin/env python3
"""
clean_mlb_data.py

Place this script in the same folder as your raw CSVs. It will:
 1. Find every .csv in the current directory
 2. Rename columns:
      - "H_caret-up" → "HR"
      - "P_caret-up" → "SO"
      - strip leading "H_" or "P_" from all other column names
 3. Overwrite each CSV in place with the cleaned version
"""

from pathlib import Path

import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns:
      - H_caret-up → HR
      - P_caret-up → SO
    """
    rename_map = {}
    for col in df.columns:
        if col == "H_caret-up":
            rename_map[col] = "H_HR"
        elif col == "P_caret-up":
            rename_map[col] = "P_SO"
    return df.rename(columns=rename_map)


def main():
    # Look for CSVs in the ../data/ directory relative to this script
    data_dir = Path(__file__).parent.parent / "data"
    for csv_path in data_dir.glob("*.csv"):
        print(f"Cleaning {csv_path.name} …")
        df = pd.read_csv(csv_path)
        df_clean = clean_columns(df)
        df_clean.to_csv(csv_path, index=False)
        print(f" ↳ Overwrote {csv_path.name}")


if __name__ == "__main__":
    main()
