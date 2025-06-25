#!/bin/bash

# FirstTimeRun.sh - Setup script for MLB Playoff Prediction Capstone Project
# This script sets up the environment and runs the complete pipeline

set -e  # Exit on any error

echo "========================================"
echo "MLB Playoff Prediction - First Time Setup"
echo "========================================"

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
echo "Step 5: Collecting MLB data..."
echo "This may take several minutes as it scrapes data from multiple years..."
uv run notebooks/baseball.py

# Step 6: Clean the collected data
echo ""
echo "Step 6: Cleaning collected data..."
uv run notebooks/clean_data.py

# Step 7: Train the model
echo ""
echo "Step 7: Training the XGBoost model..."
echo "This will create the trained model in the assets directory..."
mkdir -p assets
uv run notebooks/xgboost.py

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
echo "If the browser didn't open automatically, go to: http://localhost:8501"
echo "========================================"
