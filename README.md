# MLB Playoff Prediction - Capstone Project

## NSS DS8 Capstone Proposal - Andrew Richard

### Executive Summary
This project examines historical MLB data and builds machine learning models to predict playoff outcomes based on team performance statistics. Baseball is a sport rich in data, with decades of detailed statistics available. By leveraging this wealth of information, the project successfully determines which teams are most likely to make the playoffs and identifies the most influential factors in making accurate predictions. The final product includes XGBoost and Logistic Regression models with >97% ROC AUC performance and an interactive Streamlit dashboard.

### Motivation
- Previous NSS projects have used machine learning to analyze sports data, but this project takes a comprehensive approach to baseball playoff prediction using advanced cross-validation techniques and model optimization
- MLB playoffs represent the ultimate achievement after a long regular season, and predicting playoff success is a classic challenge in sports analytics
- Data is collected from [MLB.com](https://www.mlb.com/stats/team), which provides comprehensive team statistics for every MLB season
- As a baseball fan, I was motivated by the opportunity to combine my interest in the sport with advanced data science and predictive modeling techniques

### Data Questions
- Which teams are most likely to make the playoffs based on regular season performance data?
- What statistics or features are most important in making accurate predictions about playoff qualification?
- How do different machine learning models compare in their ability to predict playoff outcomes?

### Minimum Viable Product
The minimum viable product is a complete machine learning pipeline that takes MLB season data and outputs playoff probabilities for each team, featuring:
- Cross-validated XGBoost and Logistic Regression models with >97% ROC AUC
- Interactive Streamlit dashboard with live 2025 predictions
- Model comparison and ensemble predictions
- Historical data exploration and visualization

## Overview

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
- **Logistic Regression**: 97.75% ROC AUC (5-fold CV), 86.7% validation accuracy on 2024 data
- **XGBoost**: 97.25% ROC AUC (5-fold CV), 90.0% validation accuracy on 2024 data
- **Model Agreement**: 96.7% consensus on predictions (exceptional reliability)
- **Training Time**: ~25 seconds for complete hyperparameter optimization
- **Context**: >97% ROC AUC represents exceptional performance for MLB playoff prediction

**Original Models (Legacy):**
- **87% accuracy** on 2024 validation data
- **94.4% ROC AUC** score

**Most Predictive Features**: Pitching losses (P_L), team wins (P_W), ERA (P_ERA), saves (P_SV)
**Note**: Models trained on pre-All-Star break data to predict full-season playoff outcomes

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

## Schedule (Completed)
- ✅ Get the Data - Web scraping pipeline from MLB.com (1990-2025)
- ✅ Clean & Explore the Data - Data preprocessing and feature engineering
- ✅ Model Development - XGBoost and Logistic Regression with cross-validation
- ✅ Model Optimization - Hyperparameter tuning and performance validation
- ✅ Dashboard Creation - Interactive Streamlit application
- ✅ Project Completion - Full pipeline with automated setup

## Data Sources
- [MLB.com Team Statistics](https://www.mlb.com/stats/team) - Comprehensive team performance data (1990-2025)
- Historical playoff data and team records
- Pre-All-Star break statistics for current season predictions

## Known Issues and Challenges (Resolved)
- ✅ **Web Scraping Complexity**: Implemented robust scraping with rate limiting and error handling
- ✅ **Playoff Qualification Variability**: Focused on regular season statistics that best predict playoff success
- ✅ **Limited Postseason Data**: Used comprehensive regular season features and cross-validation for robust modeling
- ✅ **Model Performance**: Achieved >97% ROC AUC through hyperparameter optimization and ensemble methods
- ✅ **Data Quality**: Handled missing data and standardized formats across different seasons
- ✅ **Years with No Playoffs**: Excluded 1994 and 2020 seasons from training data

## Data Source

Team statistics scraped from [MLB.com](https://www.mlb.com/stats/team) with proper rate limiting and error handling.

---

**Author**: Andrew Richard  
**NSS Data Science Cohort 8**
