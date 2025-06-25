import re
import subprocess
import sys
import time

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://www.mlb.com/stats/team"
SKIP_YEARS = {1994, 2020}
MIN_YEAR = 1990
MAX_YEAR = 2025
# We only have playoff data through 2024
MAX_PLAYOFF_YEAR = max(
    range(MIN_YEAR, MAX_YEAR)
)  # 2024 last year w/ playoff data

# World Series champions by year (1990–2024)
champions = {
    1990: "Cincinnati Reds",
    1991: "Minnesota Twins",
    1992: "Toronto Blue Jays",
    1993: "Toronto Blue Jays",
    1995: "Atlanta Braves",
    1996: "New York Yankees",
    1997: "Florida Marlins",
    1998: "New York Yankees",
    1999: "New York Yankees",
    2000: "New York Yankees",
    2001: "Arizona Diamondbacks",
    2002: "Anaheim Angels",
    2003: "Florida Marlins",
    2004: "Boston Red Sox",
    2005: "Chicago White Sox",
    2006: "St. Louis Cardinals",
    2007: "Boston Red Sox",
    2008: "Philadelphia Phillies",
    2009: "New York Yankees",
    2010: "San Francisco Giants",
    2011: "St. Louis Cardinals",
    2012: "San Francisco Giants",
    2013: "Boston Red Sox",
    2014: "San Francisco Giants",
    2015: "Kansas City Royals",
    2016: "Chicago Cubs",
    2017: "Houston Astros",
    2018: "Boston Red Sox",
    2019: "Washington Nationals",
    2020: "Los Angeles Dodgers",
    2021: "Atlanta Braves",
    2022: "Houston Astros",
    2023: "Texas Rangers",
    2024: "Los Angeles Dodgers",
}


def handle_cookies(driver):
    """Dismiss the GDPR/cookie banner if present."""
    try:
        print("looking for cookie banner…")
        btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "button.ot-close-icon")
            )
        )
        btn.click()
        time.sleep(2)
    except Exception:
        pass


def select_year(driver, year):
    """Open the year dropdown and pick `year`."""
    print(f"Selecting year: {year}")
    year_dd = WebDriverWait(driver, 35).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.bui-dropdown__control")
        )
    )
    year_dd.click()
    time.sleep(1)
    opt_xpath = (
        f"//div[contains(@class,'bui-dropdown__option')"
        f" and normalize-space(text())='{year}']"
    )
    year_opt = WebDriverWait(driver, 35).until(
        EC.presence_of_element_located((By.XPATH, opt_xpath))
    )
    driver.execute_script(
        "arguments[0].scrollIntoView({block:'center'});", year_opt
    )
    time.sleep(0.5)
    WebDriverWait(driver, 35).until(
        EC.element_to_be_clickable((By.XPATH, opt_xpath))
    ).click()
    time.sleep(2)


def select_split(driver, split_name="Pre All-Star"):
    """Select the split (e.g. Pre All-Star) from its dropdown."""
    print(f"Selecting split: {split_name}")
    dropdowns = WebDriverWait(driver, 35).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.bui-dropdown__control")
        )
    )
    split_dd = next(
        (dd for dd in dropdowns if "Select a Split" in dd.text), None
    )
    if split_dd is None:
        raise RuntimeError("Could not find the split dropdown")
    split_dd.click()
    time.sleep(1)
    opt = WebDriverWait(driver, 35).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                f"//div[contains(@class,'bui-dropdown__option')"
                f" and normalize-space(text())='{split_name}']",
            )
        )
    )
    opt.click()
    time.sleep(2)


def select_game_type(driver, game_type_name="Postseason"):
    """Select a game type (Regular Season / Postseason / etc.)."""
    print(f"Selecting game type: {game_type_name}")
    ctrl_xpath = (
        "//div[contains(@class,'stats-filter-gametype')]"
        "//div[contains(@class,'bui-dropdown__control')]"
    )
    game_dd = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, ctrl_xpath))
    )
    game_dd.click()
    time.sleep(1)
    opt_xpath = (
        "//div[contains(@class,'stats-filter-gametype')]"
        "//div[contains(@class,'bui-dropdown__option')"
        f" and normalize-space(text())='{game_type_name}']"
    )
    opt = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, opt_xpath))
    )
    opt.click()
    time.sleep(2)


def select_tab(driver, tab_name="Hitting"):
    """Switch between the Hitting/Pitching tabs."""
    print(f"Selecting tab: {tab_name}")
    btn = WebDriverWait(driver, 35).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, f'button[role="tab"][aria-label="{tab_name}"]')
        )
    )
    if btn.get_attribute("aria-selected") != "true":
        btn.click()
        time.sleep(2)


def scrape_table(driver):
    """Parse the currently visible team-stats table into a DataFrame."""
    print("Scraping table…")
    WebDriverWait(driver, 35).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.bui-table"))
    )
    driver.execute_script("""
        const t = document.querySelector('table.bui-table');
        if (t) { t.scrollLeft = t.scrollWidth; }
    """)
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", class_="bui-table")
    headers = [
        th.get_text(" ", strip=True).split()[0]
        for th in table.select("thead tr th:not([aria-hidden='true'])")
    ]
    rows = []
    for tr in table.select("tbody tr"):
        rows.append(
            [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
        )
    return pd.DataFrame(rows, columns=headers)


def add_world_series_column(df, year):
    """Add a boolean WON_WORLD_SERIES based on our champions dict."""
    print(f"Adding World Series column for {year}")
    champ = champions.get(year)
    df["WON_WORLD_SERIES"] = (
        df["TEAM"].str.contains(champ, case=False, na=False) if champ else False
    )
    return df


def clean_and_merge(h_df, p_df):
    """
    Prefix H_ and P_ to hitting/pitching stats, strip digits from TEAM,
    then merge on TEAM and LEAGUE.
    """
    print("Cleaning & merging Hitting/Pitching")
    h_rename = {
        c: f"H_{c}" for c in h_df.columns if c not in ("TEAM", "LEAGUE")
    }
    p_rename = {
        c: f"P_{c}" for c in p_df.columns if c not in ("TEAM", "LEAGUE")
    }
    h_df = h_df.rename(columns=h_rename)
    h_df["TEAM"] = h_df["TEAM"].str.replace(r"\d+", "", regex=True)
    p_df = p_df.rename(columns=p_rename)
    p_df["TEAM"] = p_df["TEAM"].str.replace(r"\d+", "", regex=True)
    return pd.merge(h_df, p_df, on=["TEAM", "LEAGUE"], how="outer")


def get_chrome_version():
    """Get the current Chrome version installed on the system."""
    try:
        result = subprocess.run(
            [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "--version",
            ],
            capture_output=True,
            text=True,
        )
        version_match = re.search(r"(\d+)\.", result.stdout)
        if version_match:
            return int(version_match.group(1))
    except Exception as e:
        print(f"Could not detect Chrome version: {e}")
    return None


def create_chrome_driver():
    """Create Chrome driver with proper version handling."""
    chrome_version = get_chrome_version()

    # Try undetected-chromedriver first with minimal options
    try:
        if chrome_version:
            print(f"Detected Chrome version: {chrome_version}")
            # Try to use specific version
            driver = uc.Chrome(version_main=chrome_version)
        else:
            # Fallback to auto-detection
            driver = uc.Chrome()
        print("Successfully created undetected Chrome driver")
        return driver
    except Exception as e:
        print(f"Error creating undetected Chrome driver: {e}")
        print("Trying with webdriver-manager...")

        # Fallback to webdriver-manager with minimal options
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("Successfully created Chrome driver with webdriver-manager")
            return driver
        except Exception as e2:
            print(
                f"Failed to create Chrome driver with webdriver-manager: {e2}"
            )

            # Last resort: try regular selenium without service
            try:
                driver = webdriver.Chrome(options=chrome_options)
                print(
                    "Successfully created Chrome driver with default selenium"
                )
                return driver
            except Exception as e3:
                print(f"All Chrome driver creation methods failed: {e3}")
                print(
                    "Please ensure Chrome is installed and update it to the latest version."
                )
                print("You can also try installing ChromeDriver manually.")
                raise


def get_user_input():
    """Get start and end year from user with validation."""
    print("MLB Data Scraper")
    print("================")
    print(f"Available years: {MIN_YEAR}-{MAX_YEAR}")
    print(
        f"Note: Years {sorted(SKIP_YEARS)} are automatically excluded (no data available)"
    )
    print()

    while True:
        try:
            start_year_input = input(
                f"Enter start year ({MIN_YEAR}-{MAX_YEAR}): "
            ).strip()
            if not start_year_input:
                print("Please enter a valid year.")
                continue
            start_year = int(start_year_input)

            if start_year < MIN_YEAR or start_year > MAX_YEAR:
                print(f"Start year must be between {MIN_YEAR} and {MAX_YEAR}")
                continue

            break
        except ValueError:
            print("Please enter a valid number.")
            continue

    while True:
        try:
            end_year_input = input(
                f"Enter end year ({start_year}-{MAX_YEAR}): "
            ).strip()
            if not end_year_input:
                print("Please enter a valid year.")
                continue
            end_year = int(end_year_input)

            if end_year < start_year:
                print("End year must be greater than or equal to start year.")
                continue
            if end_year > MAX_YEAR:
                print(f"End year must be {MAX_YEAR} or earlier.")
                continue

            break
        except ValueError:
            print("Please enter a valid number.")
            continue

    return start_year, end_year


def scrape_mlb_stats(start_year=None, end_year=None, interactive=True):
    """Main scraper: for each year, get pre‐All‐Star stats + playoff flag."""
    if start_year is None or end_year is None:
        start_year, end_year = get_user_input()

    # Generate years list excluding skip years
    years = [y for y in range(start_year, end_year + 1) if y not in SKIP_YEARS]

    if not years:
        print(f"No valid years to process between {start_year} and {end_year}")
        print(f"Excluded years: {sorted(SKIP_YEARS)}")
        return

    print(f"\nProcessing years: {years}")
    print(f"Total years to process: {len(years)}")

    # If excluded years exist in the range, show them
    excluded_in_range = [
        y for y in range(start_year, end_year + 1) if y in SKIP_YEARS
    ]
    if excluded_in_range:
        print(f"Excluded years in range: {excluded_in_range}")

    # Interactive confirmation for user input mode
    if interactive and len(sys.argv) == 1:
        print("\nThis will scrape MLB data for the specified years.")
        print(
            "The process may take several minutes depending on the number of years."
        )
        while True:
            confirm = (
                input("\nDo you want to continue? (y/n): ").strip().lower()
            )
            if confirm in ["y", "yes"]:
                break
            elif confirm in ["n", "no"]:
                print("Scraping cancelled.")
                return
            else:
                print("Please enter 'y' or 'n'")

    print()

    driver = create_chrome_driver()

    try:
        for year in years:
            print(f"\n=== Year {year} ===")
            driver.get(URL)
            handle_cookies(driver)

            # 1) Pre-All-Star Hitting & Pitching
            select_year(driver, year)
            select_split(driver, "Pre All-Star")
            select_tab(driver, "Hitting")
            h_df = scrape_table(driver)
            select_tab(driver, "Pitching")
            select_year(driver, year)
            p_df = scrape_table(driver)

            # 2) Merge & World Series flag
            merged = clean_and_merge(h_df, p_df)
            merged = add_world_series_column(merged, year)

            # 3) If year ≤ MAX_PLAYOFF_YEAR, scrape Postseason teams
            if year <= MAX_PLAYOFF_YEAR:
                # select_tab(driver, "Hitting")
                select_split(driver, "Post All-Star")
                select_year(driver, year)
                select_game_type(driver, "Postseason")
                po_df = scrape_table(driver)
                po_df["TEAM"] = po_df["TEAM"].str.replace(
                    r"\d+", "", regex=True
                )
                playoff_teams = set(po_df["TEAM"])
                merged["MADE_PLAYOFFS"] = merged["TEAM"].isin(playoff_teams)
            else:
                print(f"Skipping postseason for {year} (no data yet)")
                merged["MADE_PLAYOFFS"] = False

            # 4) Save to CSV
            fn = f"data/mlb_team_stats_{year}_pre_all_star.csv"
            merged.to_csv(fn, index=False)
            print("Saved", fn)
            time.sleep(2)
    finally:
        driver.quit()


if __name__ == "__main__":
    # Check if running non-interactively (e.g., from FirstTimeRun.sh)
    if len(sys.argv) > 1:
        try:
            start_year = int(sys.argv[1])
            end_year = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_YEAR

            # Validate arguments
            if start_year < MIN_YEAR or start_year > MAX_YEAR:
                print(
                    f"Error: Start year must be between {MIN_YEAR} and {MAX_YEAR}"
                )
                sys.exit(1)
            if end_year < start_year or end_year > MAX_YEAR:
                print(
                    f"Error: End year must be between {start_year} and {MAX_YEAR}"
                )
                sys.exit(1)

            print(
                f"Running with command line arguments: {start_year}-{end_year}"
            )
            scrape_mlb_stats(start_year, end_year, interactive=False)
        except ValueError:
            print("Error: Please provide valid year arguments")
            print("Usage: python baseball.py [start_year] [end_year]")
            sys.exit(1)
    else:
        # Interactive mode
        scrape_mlb_stats(interactive=True)
