# MLB Playoff Prediction - Capstone Project

Predicts MLB playoff outcomes using XGBoost and Logistic Regression models trained on historical team statistics (1990-2025). Features interactive Streamlit dashboard with model comparison and 2025 season predictions.

## Quick Start

### Automated Setup (Recommended)

```bash
# In your terminal, run:
./FirstTimeRun.sh                    # Full dataset (1990-2025, ~30 minutes)
./FirstTimeRun.sh 2018 2023         # Recent years (6 years, ~15 minutes)
./FirstTimeRun.sh --help            # Show all options
```

This automatically:
- Sets up Python environment
- Collects and processes MLB data
- Trains both ML models
- Launches interactive streamlit app

### Manual Setup

```bash
# Environment setup
uv venv && source .venv/bin/activate && uv sync

# Data pipeline
python defs/baseball.py 2018 2023    # Collect data
python defs/clean_data.py            # Clean data

# Train models (recommended: cross-validated)
python models/run_cv_training.py     # Train optimized CV models
# OR train original models
python models/xgb_model.py           # Train XGBoost
python models/logistic_model.py      # Train Logistic Regression

# Launch dashboard
streamlit run streamlit_app.py
```

## Features

### Cross-Validated Models (Enhanced)
- **5-Fold Stratified Cross-Validation**: Ensures robust model evaluation
- **Hyperparameter Optimization**: GridSearchCV (Logistic) + RandomizedSearchCV (XGBoost)
- **Performance**: >97% ROC AUC on cross-validation
- **Automatic Model Selection**: Uses best available models (CV preferred)

### Multi-Model Comparison
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **Logistic Regression**: Linear model with cross-validated regularization
- **Ensemble**: Combines both models for robust predictions
- **Model Agreement**: 96.7% prediction agreement between models

### Interactive Dashboard
- **Current Season**: Live 2025 playoff predictions with CV model indicators
- **Model Analysis**: Cross-validation methodology, hyperparameter details, stability analysis
- **Historical Data**: Explore team statistics and playoff outcomes
- **Visualizations**: Charts and metrics with data citations

### Model Training Options
```bash
# Recommended: Cross-validated models with optimization
python models/run_cv_training.py        # ~25 seconds, optimized hyperparameters

# Check model status and performance
python models/upgrade_to_cv_models.py   # Status check and recommendations

# Individual model demonstration
python models/demo_cv_models.py         # See CV models in action
```

### Model Performance

**Cross-Validated Models (Recommended):**
- **Logistic Regression**: 97.75% ROC AUC (5-fold CV), 86.7% validation accuracy
- **XGBoost**: 97.25% ROC AUC (5-fold CV), 90.0% validation accuracy
- **Model Agreement**: 96.7% consensus on predictions
- **Training Time**: ~25 seconds for complete hyperparameter optimization

**Original Models:**
- **87% accuracy** on 2024 validation data
- **94.4% ROC AUC** score

**Key Features**: Pitching losses (P_L), team wins (P_W), ERA (P_ERA), saves (P_SV)

## Data Pipeline

1. **Web Scraping**: Collect team stats from MLB.com (1990-2025)
2. **Data Cleaning**: Standardize formats and handle missing data
3. **Model Training**: Train XGBoost and Logistic Regression classifiers
4. **Validation**: Test on held-out 2024-2025 data

## Year Range Options

| Range | Years | Time | Best For |
|-------|-------|------|----------|
| `2023 2023` | 1 | ~5 min | Quick testing |
| `2018 2023` | 6 | ~15 min | Recent analysis |
| `1990 2025` | 34 | ~30 min | Full historical data |

**Note**: Years 1994 and 2020 excluded (no playoff data available)

## Project Structure

```
capstone/
├── defs/                    # Data collection and cleaning
├── models/                  # ML model training and utilities
├── pages/                   # Streamlit dashboard pages
├── data/                    # Generated CSV files
├── assets/                  # Trained model files
├── FirstTimeRun.sh         # Automated setup script
└── streamlit_app.py        # Main dashboard application
```

## Requirements

- Python 3.12+
- Chrome browser (for web scraping)
- Dependencies managed via `uv`

## Data Source

Team statistics scraped from [MLB.com](https://www.mlb.com/stats/team) with proper rate limiting and error handling.

---

**Author**: Andrew Richard
**NSS Data Science Cohort 8**
