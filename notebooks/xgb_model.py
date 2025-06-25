# ======== Import necessary libraries ========
import glob  # for finding file paths via pattern matching
import os
import pickle  # for serializing the trained model
import re  # for extracting year from filename

import pandas as pd  # for CSV I/O and DataFrame operations
import xgboost as xgb  # for training the Gradient Boosted Trees
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

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
    print("  python notebooks/baseball.py")
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

# ======== Create XGBoost DMatrix objects ========
print("======== Creating DMatrix ========")
# DMatrix is the optimized internal data structure for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)

# Only create validation DMatrix if we have validation data
if df_val is not None:
    X_val, y_val = preprocess(df_val)
    dval = xgb.DMatrix(X_val, label=y_val)
else:
    dval = None
    X_val, y_val = None, None

# ======== Set XGBoost parameters ========
params = {
    "objective": "binary:logistic",  # for binary classification
    "eval_metric": "logloss",  # training/validation loss
    "tree_method": "hist",  # fast histogram-based splits
    "eta": 0.1,  # learning rate
    "max_depth": 6,  # maximum tree depth
    "subsample": 0.8,  # row subsample ratio
    "colsample_bytree": 0.8,  # feature subsample ratio
    "seed": 42,  # for reproducibility
}

# ======== Train with early stopping ========
if dval is not None:
    print("======== Training with early stopping ========")
    # provide both train and validation DMatrix for monitoring
    evals = [(dtrain, "train"), (dval, "validation")]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # maximum number of boosting rounds
        evals=evals,
        early_stopping_rounds=10,  # stop if no improvement on validation
        verbose_eval=True,  # print progress every round
    )
else:
    print("======== Training without validation (no 2024 data) ========")
    # train with just the training set
    evals = [(dtrain, "train")]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=100,  # fewer rounds without validation
        evals=evals,
        verbose_eval=True,  # print progress every round
    )

# ======== Evaluate on 2024 validation set ========
if dval is not None and y_val is not None:
    print("======== Evaluating on 2024 data ========")
    # get predicted probabilities and convert to binary labels
    y_proba = bst.predict(dval)
    y_pred = (y_proba > 0.5).astype(int)

    # print(f"Accuracy  : {accuracy_score(y_val, y_pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_val, y_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
else:
    print("======== Skipping validation evaluation (no 2024 data) ========")

# ======== Feature importances ========
print("======== Top 10 feature importances ========")
# get_score returns a dict: feature_name → importance weight
imps = bst.get_score(importance_type="weight")
# sort and list top 10 by weight
top10 = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, score in top10:
    print(f"{feat}: {score}")

# ======== Save the trained Booster as a pickle ========
print("======== Saving model to pickle ========")
# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)
# save_raw() returns the internal model bytes
raw = bst.save_raw()
with open("assets/xgb_playoffs.pkl", "wb") as f:
    pickle.dump(raw, f)
print("======== Model pickled to assets/xgb_playoffs.pkl ========")
