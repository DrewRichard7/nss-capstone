import time
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://www.mlb.com/stats/team"


def scrape_mlb_stats():
    driver = uc.Chrome()
    print(f"Navigating to: {URL}")
    driver.get(URL)

    try:
        # Handle cookie banner
        try:
            print("Checking for cookie banner...")
            cookie_close_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ot-close-icon"))
            )
            cookie_close_btn.click()
            print("Closed cookie banner")
            time.sleep(1)
        except:
            print("No cookie banner found")

        # Wait for stats table
        print("Waiting for stats table...")
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.bui-table")))
        print("Table found!")

        # Scroll horizontally
        print("Scrolling to load all columns...")
        driver.execute_script("""
            const table = document.querySelector('table.bui-table');
            if (table) {
                table.scrollLeft = table.scrollWidth;
            }
        """)
        time.sleep(2)

        # Get the final HTML
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")
        stats_table = soup.find("table", class_="bui-table")

        if not stats_table:
            print("Table element not found")
            return

        # Extract headers - including the team header
        headers = ["TEAM"]  # Manually add team header
        for th in stats_table.select("thead tr th:not([aria-hidden='true'])"):
            # Skip the first header since we added "TEAM" manually
            if th.get_text(strip=True).startswith("TEAM"):
                continue
            header_text = th.get_text(" ", strip=True)
            clean_header = header_text.split()[0]
            headers.append(clean_header)

        print(f"Final Headers ({len(headers)}): {headers}")

        # Extract rows - including team names
        all_rows_data = []
        for row in stats_table.select("tbody tr"):
            # Get team name from <th> element
            team_name = row.find("th").get_text(strip=True)

            # Get all other stats from <td> elements
            cells = row.find_all("td")
            row_data = [team_name] + [cell.get_text(strip=True) for cell in cells]
            all_rows_data.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(all_rows_data, columns=headers)
        print("\nFirst 5 rows:")
        print(df.head())

        # Save to CSV
        df.to_csv("mlb_team_stats.csv", index=False)
        print("\nData saved to mlb_team_stats.csv")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_mlb_stats()
