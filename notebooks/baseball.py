import time
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

URL = "https://www.mlb.com/stats/team"

# World Series champions by year (2010-2024, skipping 2020)
champions = {
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
    2021: "Atlanta Braves",
    2022: "Houston Astros",
    2023: "Texas Rangers",
    2024: "Los Angeles Dodgers",
}


def handle_cookies(driver):
    try:
        cookie_close_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ot-close-icon"))
        )
        cookie_close_btn.click()
        time.sleep(3)
    except Exception as e:
        print(f"No cookie banner found: {e}")


def select_year(driver, year):
    dropdowns = WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.bui-dropdown__control")
        )
    )
    # The first dropdown is the year selector
    year_dropdown = dropdowns[0]
    year_dropdown.click()
    time.sleep(3)
    year_option = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                f"//div[contains(@class, 'bui-dropdown__option') and text()='{year}']",
            )
        )
    )
    year_option.click()
    time.sleep(3)


def select_split(driver, split_name="Pre All-Star"):
    # Find all dropdowns
    dropdowns = WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.bui-dropdown__control")
        )
    )
    # Find the one with the label "Select a Split"
    split_dropdown = None
    for dd in dropdowns:
        try:
            if "Select a Split" in dd.text:
                split_dropdown = dd
                break
        except Exception:
            continue
    if split_dropdown is None:
        # Fallback: try to find the one with "None" as the first option
        for dd in dropdowns:
            dd.click()
            time.sleep(2)
            options = driver.find_elements(By.CSS_SELECTOR, "div.bui-dropdown__option")
            if any("None" in opt.text for opt in options):
                split_dropdown = dd
                break
            dd.click()  # close it
            time.sleep(2)
    if split_dropdown is None:
        raise Exception("Could not find the split dropdown!")
    split_dropdown.click()
    time.sleep(3)
    split_option = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                f"//div[contains(@class, 'bui-dropdown__option') and text()='{split_name}']",
            )
        )
    )
    split_option.click()
    time.sleep(3)


def scrape_table(driver):
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.bui-table")))

    driver.execute_script("""
        const table = document.querySelector('table.bui-table');
        if (table) {
            table.scrollLeft = table.scrollWidth;
        }
    """)
    time.sleep(2)

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, "html.parser")
    stats_table = soup.find("table", class_="bui-table")
    if not stats_table:
        raise Exception("Table element not found")

    headers = ["TEAM"]
    for th in stats_table.select("thead tr th:not([aria-hidden='true'])"):
        if th.get_text(strip=True).startswith("TEAM"):
            continue
        header_text = th.get_text(" ", strip=True)
        clean_header = header_text.split()[0]
        headers.append(clean_header)

    all_rows_data = []
    for row in stats_table.select("tbody tr"):
        team_name = row.find("th").get_text(strip=True)
        cells = row.find_all("td")
        row_data = [team_name] + [cell.get_text(strip=True) for cell in cells]
        all_rows_data.append(row_data)

    df = pd.DataFrame(all_rows_data, columns=headers)
    df = df.rename(columns={"caret-up": "HR"})
    return df


def add_world_series_column(df, year):
    champion = champions.get(year)
    if champion:
        df["WON_WORLD_SERIES"] = df["TEAM"].str.contains(champion, case=False, na=False)
    else:
        df["WON_WORLD_SERIES"] = False
    return df


def scrape_mlb_stats():
    current_year = datetime.now().year
    years = [y for y in range(current_year, current_year - 16, -1) if y != 2020]

    driver = uc.Chrome()
    try:
        for year in years:
            print(f"\nScraping year: {year}")
            driver.get(URL)
            handle_cookies(driver)
            select_year(driver, year)
            select_split(driver, "Pre All-Star")
            df = scrape_table(driver)
            df = add_world_series_column(df, year)
            filename = f"../data/mlb_team_stats_{year}_pre_all_star.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
            time.sleep(2)
    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_mlb_stats()
