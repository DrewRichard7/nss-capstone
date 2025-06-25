#!/bin/bash

# FirstTimeRun.sh - Setup script for MLB Playoff Prediction Capstone Project
# This script sets up the environment and runs the complete pipeline
# Usage: ./FirstTimeRun.sh [start_year] [end_year]
# Example: ./FirstTimeRun.sh 2018 2023

set -e  # Exit on any error

# Function to show help
show_help() {
    echo "MLB Playoff Prediction - First Time Setup Script"
    echo ""
    echo "Usage: ./FirstTimeRun.sh [start_year] [end_year]"
    echo ""
    echo "Arguments:"
    echo "  start_year    Starting year for data collection (1990-2025)"
    echo "  end_year      Ending year for data collection (start_year-2025)"
    echo ""
    echo "Examples:"
    echo "  ./FirstTimeRun.sh                # Use default range (1990-2025)"
    echo "  ./FirstTimeRun.sh 2018 2023      # Collect data for 2018-2023"
    echo "  ./FirstTimeRun.sh 2020           # Start from 2020, end at 2025"
    echo ""
    echo "Notes:"
    echo "  - Years 1994 and 2020 are automatically excluded (no data available)"
    echo "  - Full range (1990-2025) takes 20-30 minutes"
    echo "  - Smaller ranges complete much faster"
    echo ""
}

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Default years
DEFAULT_START_YEAR=1990
DEFAULT_END_YEAR=2025

# Parse command line arguments
START_YEAR=${1:-$DEFAULT_START_YEAR}
END_YEAR=${2:-$DEFAULT_END_YEAR}

# Validate year arguments
if ! [[ "$START_YEAR" =~ ^[0-9]+$ ]] || ! [[ "$END_YEAR" =~ ^[0-9]+$ ]]; then
    echo "Error: Years must be numeric values"
    echo "Usage: ./FirstTimeRun.sh [start_year] [end_year]"
    echo "Example: ./FirstTimeRun.sh 2018 2023"
    exit 1
fi

if [ "$START_YEAR" -lt 1990 ] || [ "$START_YEAR" -gt 2025 ]; then
    echo "Error: Start year must be between 1990 and 2025"
    exit 1
fi

if [ "$END_YEAR" -lt "$START_YEAR" ] || [ "$END_YEAR" -gt 2025 ]; then
    echo "Error: End year must be between $START_YEAR and 2025"
    exit 1
fi

echo "========================================"
echo "MLB Playoff Prediction - First Time Setup"
echo "Data Range: $START_YEAR - $END_YEAR"
echo "========================================"

# Show which years will be processed (excluding skip years)
echo "Calculating years to process..."
YEARS_TO_PROCESS=()
EXCLUDED_YEARS=()

for year in $(seq $START_YEAR $END_YEAR); do
    if [ "$year" -eq 1994 ] || [ "$year" -eq 2020 ]; then
        EXCLUDED_YEARS+=($year)
    else
        YEARS_TO_PROCESS+=($year)
    fi
done

echo ""
echo "Years to process: ${YEARS_TO_PROCESS[*]}"
if [ ${#EXCLUDED_YEARS[@]} -gt 0 ]; then
    echo "Excluded years in range: ${EXCLUDED_YEARS[*]}"
fi
echo "Total years to process: ${#YEARS_TO_PROCESS[@]}"
echo ""

# Confirmation prompt
if [ ${#YEARS_TO_PROCESS[@]} -eq 0 ]; then
    echo "Error: No valid years to process in the specified range."
    echo "All years in range $START_YEAR-$END_YEAR are excluded."
    exit 1
fi

echo "This will set up the complete MLB prediction pipeline including:"
echo "  - Environment setup"
echo "  - Data collection for ${#YEARS_TO_PROCESS[@]} years"
echo "  - Data cleaning and preprocessing"
echo "  - Model training"
echo "  - Streamlit app launch"
echo ""
echo "Estimated time: $(( ${#YEARS_TO_PROCESS[@]} * 1 + 5 )) minutes"
echo ""

read -p "Do you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the capstone directory."
    exit 1
fi

# Step 1: Install uv if not already installed
echo "Step 1: Installing uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv with pip..."
    pip install uv
else
    echo "uv is already installed."
fi

# Step 2: Create virtual environment with uv
echo ""
echo "Step 2: Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Removing old environment..."
    rm -rf .venv
fi
uv venv
source .venv/bin/activate
echo "Virtual environment active"

# Step 3: Sync dependencies
echo ""
echo "Step 3: Syncing dependencies..."
uv sync

# Step 4: Create data directory if it doesn't exist
echo ""
echo "Step 4: Setting up data directory..."
mkdir -p data

# Step 5: Run data collection (baseball.py)
echo ""
echo "Step 5: Collecting MLB data for years $START_YEAR-$END_YEAR..."
echo "This may take several minutes as it scrapes data from multiple years..."
echo "Note: Years 1994 and 2020 will be automatically excluded if in range"
uv run defs/baseball.py "$START_YEAR" "$END_YEAR"

# Step 6: Clean the collected data
echo ""
echo "Step 6: Cleaning collected data..."
uv run defs/clean_data.py

# Step 7: Train the model
echo ""
echo "Step 7: Training the XGBoost model..."
echo "This will create the trained model in the assets directory..."
mkdir -p assets
uv run models/xgb_model.py

# Step 8: Launch the Streamlit app
echo ""
echo "Step 8: Launching Streamlit application..."
echo "The app will open in your default browser."
echo "Press Ctrl+C to stop the application when you're done."
echo ""
echo "Starting Streamlit in 3 seconds..."
sleep 3

uv run streamlit run streamlit_app.py

echo ""
echo "========================================"
echo "Setup complete! The Streamlit app should now be running."
echo "Data collected for years: $START_YEAR-$END_YEAR"
echo "If the browser didn't open automatically, go to: http://localhost:8501"
echo "========================================"
