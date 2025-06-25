# MLB Playoff Prediction - Capstone Project

Predicts MLB World Series winners using machine learning analysis of historical team statistics (1990-2025).

## Quick Start

### Automated Setup (Recommended)

```bash
./FirstTimeRun.sh                    # Full dataset (1990-2025, ~40 minutes)
./FirstTimeRun.sh 2018 2023         # Recent years (6 years, ~15 minutes)
./FirstTimeRun.sh 2020              # From 2020 to 2025 (6 years, ~15 minutes)
./FirstTimeRun.sh --help            # Show all options
```

This automatically:
- Sets up Python environment with `uv`
- Collects MLB data for specified years
- Cleans and processes the data
- Trains XGBoost model
- Launches Streamlit dashboard

### Manual Setup

```bash
# 1. Environment
uv venv && source .venv/bin/activate && uv sync

# 2. Data Collection
python notebooks/baseball.py 2018 2023    # Command line
python notebooks/baseball.py              # Interactive mode

# 3. Data Cleaning
python notebooks/clean_data.py

# 4. Model Training
python notebooks/xgb_model.py

# 5. Launch App
streamlit run streamlit_app.py
```

## Year Range Selection

Choose your data scope based on needs:

| Range | Years | Time | Use Case |
|-------|-------|------|----------|
| `2023 2023` | 1 | ~6 min | Quick testing |
| `2020 2023` | 3 | ~10 min | Recent analysis |
| `2018 2023` | 5 | ~15 min | Medium dataset |
| `2000 2010` | 11 | ~25 min | Decade study |
| `1990 2025` | 34 | ~40 min | Full historical |

**Available years**: 1990-2025 (excludes 1994 and 2020 - no data available)

## Usage Examples

### Development & Testing
```bash
./FirstTimeRun.sh 2023 2023          # Single year test
python notebooks/baseball.py 2022 2023  # Recent data only
```

### Analysis & Research
```bash
./FirstTimeRun.sh                    # Complete historical dataset
./FirstTimeRun.sh 2000 2009         # Specific decade
```

### Interactive Mode
```bash
python notebooks/baseball.py
# Prompts for start/end years with validation
```

## Model Performance

- **Training Data**: 940 team records (1990-2023)
- **Validation**: 30 team records (2024-2025)
- **Accuracy**: 90% on 2024 validation data
- **ROC AUC**: 96%
- **Top Features**: Pitching losses, wins, strikeouts

## Data Pipeline

1. **Web Scraping** (`baseball.py`): Collects team stats from MLB.com
2. **Data Cleaning** (`clean_data.py`): Standardizes column names and formats
3. **Model Training** (`xgb_model.py`): Trains XGBoost classifier for playoff prediction
4. **Streamlit App** (`streamlit_app.py`): Interactive dashboard for predictions

## Input Validation

All scripts validate inputs with helpful error messages:

- Years must be between 1990-2025
- End year must be ≥ start year
- Automatically excludes problematic years (1994, 2020)
- Shows preview of years to process before starting

## Troubleshooting

### Chrome Driver Issues
The script automatically tries multiple Chrome driver methods:
1. Update Chrome browser to latest version
2. Restart the script (it will try different drivers)

### No Data Generated
- Ensure stable internet connection
- Check that `data/` directory exists
- Verify Chrome browser is installed

### FirstTimeRun.sh Issues
```bash
chmod +x FirstTimeRun.sh             # Make executable
./FirstTimeRun.sh --help            # Check usage
```

## Project Structure

```
capstone/
├── notebooks/
│   ├── baseball.py          # Data collection with year input
│   ├── clean_data.py        # Data preprocessing
│   └── xgb_model.py         # Model training
├── data/                    # CSV files (created by scripts)
├── assets/                  # Trained models (created by scripts)
├── FirstTimeRun.sh          # Automated setup script
└── streamlit_app.py         # Dashboard application
```

## Technical Details

- **Web Scraping**: Selenium with undetected-chromedriver
- **ML Model**: XGBoost binary classifier
- **Data**: Team hitting/pitching stats, playoff results, World Series winners
- **Features**: 37 statistical features per team per year
- **Target**: Binary classification (made playoffs: yes/no)

## Dependencies

Managed with `uv` (fast Python package manager):
- pandas, numpy: Data manipulation
- scikit-learn, xgboost: Machine learning
- selenium, beautifulsoup4: Web scraping
- streamlit: Dashboard
- See `pyproject.toml` for complete list

---

**Author**: Andrew Richard  
**Program**: NSS Data Science Cohort 8  
**Data Source**: MLB.com team statistics