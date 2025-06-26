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


def clean_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean team names by removing shortnames that are appended to the full name.
    Examples:
      - "Arizona DiamondbacksD-backs‌‌‌" → "Arizona Diamondbacks"
      - "Boston Red SoxRed Sox‌‌‌" → "Boston Red Sox"
    """
    if "TEAM" not in df.columns:
        return df

    # Define mapping of full names to clean names
    team_name_mapping = {
        "Arizona DiamondbacksD-backs‌‌‌": "Arizona Diamondbacks",
        "Atlanta BravesBraves‌‌‌": "Atlanta Braves",
        "Baltimore OriolesOrioles‌‌‌": "Baltimore Orioles",
        "Boston Red SoxRed Sox‌‌‌": "Boston Red Sox",
        "Chicago CubsCubs‌‌‌": "Chicago Cubs",
        "Chicago White SoxWhite Sox‌‌‌": "Chicago White Sox",
        "Cincinnati RedsReds‌‌‌": "Cincinnati Reds",
        "Cleveland GuardiansGuardians‌‌‌": "Cleveland Guardians",
        "Colorado RockiesRockies‌‌‌": "Colorado Rockies",
        "Detroit TigersTigers‌‌‌": "Detroit Tigers",
        "Houston AstrosAstros‌‌‌": "Houston Astros",
        "Kansas City RoyalsRoyals‌‌‌": "Kansas City Royals",
        "Los Angeles AngelsAngels‌‌‌": "Los Angeles Angels",
        "Los Angeles DodgersDodgers‌‌‌": "Los Angeles Dodgers",
        "Miami MarlinsMarlins‌‌‌": "Miami Marlins",
        "Milwaukee BrewersBrewers‌‌‌": "Milwaukee Brewers",
        "Minnesota TwinsTwins‌‌‌": "Minnesota Twins",
        "New York MetsMets‌‌‌": "New York Mets",
        "New York YankeesYankees‌‌‌": "New York Yankees",
        "Oakland AthleticsAthletics‌‌‌": "Oakland Athletics",
        "Philadelphia PhilliesPhillies‌‌‌": "Philadelphia Phillies",
        "Pittsburgh PiratesPirates‌‌‌": "Pittsburgh Pirates",
        "San Diego PadresPadres‌‌‌": "San Diego Padres",
        "San Francisco GiantsGiants‌‌‌": "San Francisco Giants",
        "Seattle MarinersMariners‌‌‌": "Seattle Mariners",
        "St. Louis CardinalsCardinals‌‌‌": "St. Louis Cardinals",
        "Tampa Bay RaysRays‌‌‌": "Tampa Bay Rays",
        "Texas RangersRangers‌‌‌": "Texas Rangers",
        "Toronto Blue JaysBlue Jays‌‌‌": "Toronto Blue Jays",
        "Washington NationalsNationals‌‌‌": "Washington Nationals",
    }

    # Clean team names
    df_clean = df.copy()
    df_clean["TEAM"] = df_clean["TEAM"].replace(team_name_mapping)

    return df_clean


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
        df_clean = clean_team_names(df_clean)
        df_clean.to_csv(csv_path, index=False)
        print(f" ↳ Overwrote {csv_path.name}")


if __name__ == "__main__":
    main()
