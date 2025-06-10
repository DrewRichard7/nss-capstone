from pybaseball import standings
import pandas as pd


def main():
    print("Getting MLB standings from the year 2016...")
    # The function returns a list of DataFrames, one for each division
    mlb_standings_list = standings(2016)

    if not mlb_standings_list:
        print("Could not retrieve standings for 2016. The list is empty.")
    else:
        print("Successfully retrieved standings. Displaying by division:")
        # Loop through the list and print each division's standings DataFrame
        for i, division_df in enumerate(mlb_standings_list):
            print(f"\n--- Division {i + 1} ---")
            print(division_df)

        # To combine all divisions into a single DataFrame, you can use pandas.concat
        print("\n--- Combined Standings ---")
        all_standings_df = pd.concat(mlb_standings_list)
        print(all_standings_df)


if __name__ == "__main__":
    main()
