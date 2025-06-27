# ======== Import necessary libraries ========
import glob  # for finding file paths via pattern matching
import os
import pickle  # for serializing the trained model
import re  # for extracting year from filename

import pandas as pd  # for CSV I/O and DataFrame operations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ======== Locate all yearly CSVs ========
print("======== Locating CSV files ========")
pattern = "data/mlb_team_stats_*_pre_all_star.csv"
all_files = sorted(glob.glob(pattern))  # get list of all matching CSV files
print(f"======== Found {len(all_files)} files ========")

# Check if any files were found
if len(all_files) == 0:
    print("ERROR: No CSV files found!")
    print(f"Looking for pattern: {pattern}")
    print("Please run the data collection script first:")
    print("  python defs/baseball.py")
    print("or use FirstTimeRun.sh for complete setup")
    exit(1)

# ======== Split into train (<=2023) and val (2024) ========
train_dfs = []
val_dfs = []
for path in all_files:
    m = re.search(r"_(\d{4})_", path)  # extract the four-digit year
    if not m:
        continue
    year = int(m.group(1))
    print(f"======== Reading {os.path.basename(path)} (year={year}) ========")
    df = pd.read_csv(path)
    if year < 2024:
        train_dfs.append(df)  # append to training list
    elif year == 2024:
        val_dfs.append(df)  # append to validation list
    # any files for year >2024 are ignored for now

# Check if we have training data
if len(train_dfs) == 0:
    print("ERROR: No training data found (no files with years < 2024)!")
    print("Please ensure you have collected data for years before 2024.")
    exit(1)

# concatenate all yearly DataFrames into one large train/val set
df_train = pd.concat(train_dfs, ignore_index=True)

# Handle validation data (may be empty if no 2024 data)
if len(val_dfs) > 0:
    df_val = pd.concat(val_dfs, ignore_index=True)
    print(f"======== Combined validation shape: {df_val.shape} ========")
else:
    df_val = None
    print("======== No 2024 validation data found ========")

print(f"======== Combined train shape: {df_train.shape} ========")


# ======== Preprocessing function ========
def preprocess(df_in):
    """
    Clean and prepare one season's DataFrame:
      - Drop unused ID columns
      - Encode categorical league
      - Convert target to numeric
      - Clean up any columns parsed as object
      - Split into feature matrix X and target vector y
    """
    df = df_in.copy()

    # Drop identifier and the alternate target we won't use now
    df.drop(columns=["TEAM", "WON_WORLD_SERIES"], inplace=True)

    # Map AL/NL to binary values for modeling
    df["LEAGUE"] = df["LEAGUE"].map({"AL": 0, "NL": 1})

    # Our binary target (made playoffs) → integer 0/1
    df["MADE_PLAYOFFS"] = df["MADE_PLAYOFFS"].astype(int)

    # Find any remaining object-typed columns (e.g. numeric data read as strings)
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        # strip non-numeric chars then cast to float
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )

    # Replace any NaNs introduced in conversion
    df.fillna(0, inplace=True)

    # All columns except the target become features
    feature_cols = [c for c in df.columns if c != "MADE_PLAYOFFS"]
    X = df[feature_cols]
    y = df["MADE_PLAYOFFS"]
    return X, y


# ======== Preprocess train & validation ========
print("======== Preprocessing data ========")
X_train, y_train = preprocess(df_train)

if df_val is not None:
    X_val, y_val = preprocess(df_val)
else:
    X_val, y_val = None, None

# ======== Feature scaling for Logistic Regression ========
print("======== Scaling features ========")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

if X_val is not None:
    X_val_scaled = scaler.transform(X_val)
else:
    X_val_scaled = None

# ======== Train Logistic Regression model ========
print("======== Training Logistic Regression model ========")
# Using L2 regularization with balanced class weights
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced",  # Handle class imbalance
    C=1.0,  # Regularization strength
    penalty="l2",  # L2 regularization
)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

print("======== Model training completed ========")

# ======== Evaluate on 2024 validation set ========
if X_val_scaled is not None and y_val is not None:
    print("======== Evaluating on 2024 data ========")
    # get predicted probabilities and convert to binary labels
    y_proba = logistic_model.predict_proba(X_val_scaled)[
        :, 1
    ]  # probability of class 1
    y_pred = (y_proba > 0.5).astype(int)

    print(f"ROC AUC   : {roc_auc_score(y_val, y_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
else:
    print("======== Skipping validation evaluation (no 2024 data) ========")

# ======== Feature importances (coefficients) ========
print("======== Top 10 feature coefficients (absolute values) ========")
feature_names = X_train.columns
coefficients = logistic_model.coef_[0]
feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients),
    }
).sort_values("abs_coefficient", ascending=False)

print("Top 10 most important features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']}: {row['coefficient']:.4f}")

# ======== Save the trained model and scaler ========
print("======== Saving model and scaler to pickle ========")
# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

# Save both the model and scaler together
model_data = {
    "model": logistic_model,
    "scaler": scaler,
    "feature_names": list(X_train.columns),
}

with open("assets/logistic_playoffs.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(
    "======== Model and scaler pickled to assets/logistic_playoffs.pkl ========"
)
