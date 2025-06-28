# ======== Import necessary libraries ========
import glob  # for finding file paths via pattern matching
import os
import pickle  # for serializing the trained model
import re  # for extracting year from filename
import time

import numpy as np
import pandas as pd  # for CSV I/O and DataFrame operations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
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

# ======== Define hyperparameter grid for GridSearchCV ========
print("======== Setting up GridSearchCV hyperparameter grid ========")

# Define parameter grid - focusing on key hyperparameters
param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
    "penalty": ["l1", "l2", "elasticnet"],  # Regularization type
    "solver": ["liblinear", "saga"],  # Solvers that support all penalties
    "class_weight": [None, "balanced"],  # Handle class imbalance
    "max_iter": [1000, 2000],  # Maximum iterations
}

# For elasticnet, we need to add l1_ratio parameter
param_grid_elasticnet = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["elasticnet"],
    "solver": ["saga"],  # Only saga supports elasticnet
    "class_weight": [None, "balanced"],
    "max_iter": [1000, 2000],
    "l1_ratio": [0.1, 0.5, 0.9],  # L1 ratio for elasticnet
}

print(
    f"Total parameter combinations (main grid): {np.prod([len(v) for v in param_grid.values()])}"
)
print(
    f"Total parameter combinations (elasticnet grid): {np.prod([len(v) for v in param_grid_elasticnet.values()])}"
)

# ======== Setup cross-validation strategy ========
print("======== Setting up cross-validation strategy ========")

# Use StratifiedKFold to maintain class distribution across folds
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom scoring - we'll use ROC AUC as primary metric
scoring = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

# ======== Perform GridSearchCV ========
print("======== Starting GridSearchCV (this may take several minutes) ========")
start_time = time.time()

# Base logistic regression
base_logistic = LogisticRegression(random_state=42)

# First grid search (main parameters)
print("======== Running main parameter grid search ========")
grid_search_main = GridSearchCV(
    estimator=base_logistic,
    param_grid=param_grid,
    scoring="roc_auc",  # Primary metric for selection
    cv=cv_strategy,
    n_jobs=-1,  # Use all available cores
    verbose=1,
    return_train_score=True,
)

grid_search_main.fit(X_train_scaled, y_train)

# Second grid search (elasticnet parameters)
print("======== Running elasticnet parameter grid search ========")
grid_search_elasticnet = GridSearchCV(
    estimator=base_logistic,
    param_grid=param_grid_elasticnet,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

grid_search_elasticnet.fit(X_train_scaled, y_train)

# Compare best scores and select overall best
if grid_search_main.best_score_ >= grid_search_elasticnet.best_score_:
    best_grid_search = grid_search_main
    print("======== Main parameter grid achieved best score ========")
else:
    best_grid_search = grid_search_elasticnet
    print("======== Elasticnet parameter grid achieved best score ========")

end_time = time.time()
print(
    f"======== GridSearchCV completed in {end_time - start_time:.2f} seconds ========"
)

# ======== Extract best model and results ========
best_model = best_grid_search.best_estimator_
best_params = best_grid_search.best_params_
best_score = best_grid_search.best_score_

print("======== Best Model Results ========")
print(f"Best CV ROC AUC Score: {best_score:.4f}")
print("Best Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# ======== Cross-validation with multiple metrics ========
print(
    "======== Performing detailed cross-validation with multiple metrics ========"
)

cv_scores = {}
for metric_name, metric_scorer in scoring.items():
    scores = cross_val_score(
        best_model,
        X_train_scaled,
        y_train,
        cv=cv_strategy,
        scoring=metric_scorer,
        n_jobs=-1,
    )
    cv_scores[metric_name] = scores
    print(
        f"{metric_name.upper():>12}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
    )

# ======== Analyze CV results in detail ========
print("======== Detailed Cross-Validation Analysis ========")
cv_results_df = pd.DataFrame(best_grid_search.cv_results_)

# Top 10 parameter combinations
top_10_results = cv_results_df.nlargest(10, "mean_test_score")[
    ["mean_test_score", "std_test_score", "params"]
]
print("\nTop 10 parameter combinations:")
for idx, row in top_10_results.iterrows():
    print(
        f"Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score'] * 2:.4f}), Params: {row['params']}"
    )

# ======== Train final model on all training data ========
print("======== Training final model on complete training set ========")
final_model = LogisticRegression(**best_params, random_state=42)
final_model.fit(X_train_scaled, y_train)

# ======== Evaluate on validation set if available ========
if X_val_scaled is not None and y_val is not None:
    print("======== Evaluating on 2024 validation set ========")

    # Predictions with best model
    y_proba = final_model.predict_proba(X_val_scaled)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    val_roc_auc = roc_auc_score(y_val, y_proba)
    print(f"Validation ROC AUC: {val_roc_auc:.4f}")
    print(f"CV ROC AUC:         {best_score:.4f}")
    print(f"Difference:         {val_roc_auc - best_score:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred))
    print("\nValidation Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Store validation results
    validation_results = {
        "roc_auc": val_roc_auc,
        "classification_report": classification_report(
            y_val, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }
else:
    print("======== Skipping validation evaluation (no 2024 data) ========")
    validation_results = None

# ======== Feature importance analysis ========
print("======== Feature Importance Analysis ========")
feature_names = X_train.columns
coefficients = final_model.coef_[0]
feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }
).sort_values("abs_coefficient", ascending=False)

print("Top 15 most important features:")
for idx, row in feature_importance.head(15).iterrows():
    direction = "positive" if row["coefficient"] > 0 else "negative"
    print(
        f"{row['feature']:>25}: {row['coefficient']:>8.4f} ({direction} impact)"
    )

# ======== Model Performance Summary ========
print("======== Final Model Performance Summary ========")
summary = {
    "model_type": "Logistic Regression with GridSearchCV",
    "best_cv_score": best_score,
    "best_parameters": best_params,
    "cv_scores": {
        metric: {"mean": scores.mean(), "std": scores.std()}
        for metric, scores in cv_scores.items()
    },
    "training_time_seconds": end_time - start_time,
    "total_parameter_combinations_tested": len(cv_results_df),
    "validation_results": validation_results,
    "feature_importance": feature_importance.head(10).to_dict("records"),
}

print(f"Model Type: {summary['model_type']}")
print(f"Best CV ROC AUC: {summary['best_cv_score']:.4f}")
print(f"Training Time: {summary['training_time_seconds']:.2f} seconds")
print(
    f"Parameter Combinations Tested: {summary['total_parameter_combinations_tested']}"
)

# ======== Save enhanced model and metadata ========
print("======== Saving enhanced model with cross-validation results ========")
os.makedirs("assets", exist_ok=True)

# Enhanced model data with CV results
enhanced_model_data = {
    "model": final_model,
    "scaler": scaler,
    "feature_names": list(X_train.columns),
    "best_params": best_params,
    "cv_results": cv_results_df.to_dict("records"),
    "best_cv_score": best_score,
    "cv_scores_by_metric": cv_scores,
    "feature_importance": feature_importance.to_dict("records"),
    "model_summary": summary,
    "grid_search_results": {
        "main_grid_best_score": grid_search_main.best_score_,
        "main_grid_best_params": grid_search_main.best_params_,
        "elasticnet_grid_best_score": grid_search_elasticnet.best_score_,
        "elasticnet_grid_best_params": grid_search_elasticnet.best_params_,
    },
}

# Save to pickle
with open("assets/logistic_playoffs_cv.pkl", "wb") as f:
    pickle.dump(enhanced_model_data, f)

# Also save a JSON summary for easy inspection
import json

json_summary = {
    "model_type": summary["model_type"],
    "best_cv_score": float(summary["best_cv_score"]),
    "best_parameters": best_params,
    "training_time_seconds": float(summary["training_time_seconds"]),
    "total_parameter_combinations_tested": int(
        summary["total_parameter_combinations_tested"]
    ),
    "top_10_features": [
        {
            "feature": row["feature"],
            "coefficient": float(row["coefficient"]),
            "abs_coefficient": float(row["abs_coefficient"]),
        }
        for _, row in feature_importance.head(10).iterrows()
    ],
}

with open("assets/logistic_playoffs_cv_summary.json", "w") as f:
    json.dump(json_summary, f, indent=2)

print(
    "======== Enhanced logistic model saved to assets/logistic_playoffs_cv.pkl ========"
)
print(
    "======== Model summary saved to assets/logistic_playoffs_cv_summary.json ========"
)
print(
    "======== Cross-validation enhanced Logistic Regression training completed! ========"
)
