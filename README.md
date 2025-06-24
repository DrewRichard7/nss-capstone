# NSS DS8 Capstone Proposal - Andrew Richard

## Executive Summary

    - This project examines historical MLB data and builds a machine learning model to predict the World Series winner based on previous years' team and player performance. Baseball is a sport rich in data, with decades of detailed statistics available. By leveraging this wealth of information, I aim to determine whether it's possible to accurately predict the next World Series champion, and to identify which factors are most influential in making that prediction.

## Motivation

    - Previous NSS projects have used machine learning to analyze football and baseball data, but I want to take a deeper dive into baseball, a sport with a long history of statistical analysis and a vast amount of publicly available data.
    - The World Series is the ultimate prize in Major League Baseball, awarded to the team that wins the postseason tournament after a long regular season. Predicting the winner is a classic challenge in sports analytics.
    - Data will be collected from [Baseball-Reference](https://www.baseball-reference.com/leagues/), which provides comprehensive team and player statistics for every MLB season.
    - As a baseball fan, I'm motivated by the opportunity to combine my interest in the sport with my passion for data science and predictive modeling.

## Data Questions

    - Which team is most likely to win the World Series based on regular season and historical performance data?
    - What statistics or features are most important in making accurate predictions about postseason success?

## Minimum Viable Product

    - The minimum viable product will be a machine learning model that takes in previous years' MLB season data and outputs probabilities for each team's chances of winning the World Series.

## Schedule

    - Get the Data (finish date)
    - Clean & Explore the Data (finish date)
    - Create Presentation (finish date)
    - Internal Demos (6/28/2025)
    - Graduation (7/3/2025)
    - Demo Day (7/10/2025)

## Data Sources

    - [Baseball-Reference MLB Data](https://www.baseball-reference.com/leagues/)

## Known Issues and Challenges

    - Web scraping can be challenging, but there are many resources available to help manage this process.
    - Not every team makes the postseason, and roster changes can impact team performance from year to year. I may need to account for these factors in my modeling.
    - Some teams may have limited postseason data, especially if they haven't made the playoffs often. I may need to focus on regular season performance and engineer features that best capture a team's championship potential.

## How to Run This Project

### 1. Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python dependency management.  
If you don't have `uv` installed, you can get it with:

```bash
pip install uv
```

Then, to create and sync your environment (Python 3.12+ required):

```bash
uv venv
uv pip sync
```

Or, if you prefer, use the provided `pyproject.toml`:

```bash
uv pip install -r pyproject.toml
```

Activate your environment:

```bash
source .venv/bin/activate
```

### 2. Data Collection (Web Scraping)

To collect the raw MLB team stats data (1990–2025), run the scraper:

```bash
uv run notebooks/baseball.py
```

- This will launch a Chrome browser (using Selenium and undetected-chromedriver) and automatically scrape team stats from MLB.com for each year.
- The script saves one CSV per year in the `data/` directory.

**Note:** You may need Chrome installed, and the first run may take several minutes.

### 3. Data Cleaning

To clean the raw CSVs (fix column names, etc.), run:

```bash
uv run data/clean_data.py
```

- This script finds all `.csv` files in `data/`, renames columns for consistency, and overwrites the files in place.

### 4. Model Training (Optional)

To train the XGBoost model and generate the model file for the app, run:

```bash
uv run main.py
```

- This script loads all cleaned CSVs, trains a model to predict playoff teams, evaluates performance, and saves the model to `assets/xgb_playoffs.pkl`.

### 5. Launch the Streamlit App

To start the interactive dashboard:

```bash
streamlit run streamlit_app.py
```

- The app will load the latest data and model, show validation results, and let you upload new season data for predictions.

### Troubleshooting

- If you encounter issues with ChromeDriver, make sure your Chrome browser is up to date.
- For web scraping, a stable internet connection is required.
- If you add new dependencies, use `uv pip install <package>` and re-sync your environment.
