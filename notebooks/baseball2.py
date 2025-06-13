import time
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

URL = "https://www.mlb.com/stats/team"

# World Series champions by year (1990-2024, 1994=None, 2020 included for completeness)
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
    try:
        print("looking for cookie banner...")
        cookie_close_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ot-close-icon"))
        )
        cookie_close_btn.click()
        time.sleep(2)
    except Exception:
        pass


def select_year(driver, year):
    print(f"Selecting year: {year}")
    dropdowns = WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.bui-dropdown__control")
        )
    )
    year_dropdown = dropdowns[0]
    year_dropdown.click()
    time.sleep(1)
    year_option = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                f"//div[contains(@class, 'bui-dropdown__option') and text()='{year}']",
            )
        )
    )
    year_option.click()
    time.sleep(2)


def select_split(driver, split_name="Pre All-Star"):
    print(f"Selecting split: {split_name}")
    dropdowns = WebDriverWait(driver, 30).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.bui-dropdown__control")
        )
    )
    split_dropdown = None
    for dd in dropdowns:
        try:
            if "Select a Split" in dd.text:
                split_dropdown = dd
                break
        except Exception:
            continue
    if split_dropdown is None:
        for dd in dropdowns:
            dd.click()
            time.sleep(1)
            options = driver.find_elements(By.CSS_SELECTOR, "div.bui-dropdown__option")
            if any("None" in opt.text for opt in options):
                split_dropdown = dd
                break
            dd.click()
            time.sleep(1)
    if split_dropdown is None:
        raise Exception("Could not find the split dropdown!")
    split_dropdown.click()
    time.sleep(1)
    split_option = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                f"//div[contains(@class, 'bui-dropdown__option') and text()='{split_name}']",
            )
        )
    )
    split_option.click()
    time.sleep(2)


def select_tab(driver, tab_name="Hitting"):
    print(f"Selecting tab: {tab_name}")
    # Use aria-label for robust selection
    tab_btn = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, f'button[role="tab"][aria-label="{tab_name}"]')
        )
    )
    # Only click if not already selected
    if tab_btn.get_attribute("aria-selected") != "true":
        tab_btn.click()
        time.sleep(2)


def scrape_table(driver):
    print("Scraping table...")
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.bui-table")))

    driver.execute_script("""
        const table = document.querySelector('table.bui-table');
        if (table) {
            table.scrollLeft = table.scrollWidth;
        }
    """)
    time.sleep(1)

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, "html.parser")
    stats_table = soup.find("table", class_="bui-table")
    if not stats_table:
        raise Exception("Table element not found")

    headers = []
    for th in stats_table.select("thead tr th:not([aria-hidden='true'])"):
        header_text = th.get_text(" ", strip=True)
        clean_header = header_text.split()[0]
        headers.append(clean_header)

    all_rows_data = []
    for row in stats_table.select("tbody tr"):
        row_data = []
        for cell in row.find_all(["th", "td"]):
            row_data.append(cell.get_text(strip=True))
        all_rows_data.append(row_data)

    df = pd.DataFrame(all_rows_data, columns=headers)
    return df


def add_world_series_column(df, year):
    print(f"Adding World Series column for year: {year}")
    champion = champions.get(year)
    if champion:
        df["WON_WORLD_SERIES"] = df["TEAM"].str.contains(champion, case=False, na=False)
    else:
        df["WON_WORLD_SERIES"] = False
    return df


def clean_and_merge(hitting_df, pitching_df):
    print("Cleaning and merging dataframes...")
    # Only keep TEAM and LEAGUE once, prefix all other columns
    hitting_cols = hitting_df.columns.tolist()
    pitching_cols = pitching_df.columns.tolist()

    hitting_df = hitting_df.rename(
        columns={
            col: f"H_{col}" for col in hitting_cols if col not in ["TEAM", "LEAGUE"]
        }
    )
    pitching_df = pitching_df.rename(
        columns={
            col: f"P_{col}" for col in pitching_cols if col not in ["TEAM", "LEAGUE"]
        }
    )

    merged = pd.merge(
        hitting_df, pitching_df, on=["TEAM", "LEAGUE"], how="outer", suffixes=("", "_P")
    )
    return merged


def scrape_mlb_stats():
    # 1990-2025, skipping 1994 and 2020
    years = [y for y in range(1990, 2026) if y not in (1994, 2020)]
    driver = uc.Chrome()
    try:
        for year in years:
            print(f"\nScraping year: {year}")
            driver.get(URL)
            handle_cookies(driver)
            select_year(driver, year)
            select_split(driver, "Pre All-Star")

            # Hitting
            select_tab(driver, "Hitting")
            hitting_df = scrape_table(driver)

            # Pitching
            select_tab(driver, "Pitching")
            pitching_df = scrape_table(driver)

            # Merge
            merged_df = clean_and_merge(hitting_df, pitching_df)
            merged_df = add_world_series_column(merged_df, year)

            filename = f"../data/mlb_team_stats_{year}_pre_all_star.csv"
            merged_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
            time.sleep(2)
    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_mlb_stats()
