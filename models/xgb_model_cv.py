# ======== Import necessary libraries ========
import glob  # for finding file paths via pattern matching
import os
import pickle  # for serializing the trained model
import re  # for extracting year from filename
import time

import numpy as np
import pandas as pd  # for CSV I/O and DataFrame operations
import xgboost as xgb  # for training the Gradient Boosted Trees
from scipy.stats import randint, uniform
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
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

# ======== Define hyperparameter distributions for RandomizedSearchCV ========
print(
    "======== Setting up RandomizedSearchCV hyperparameter distributions ========"
)

# Define parameter distributions for random search
# Using scipy.stats distributions for continuous parameters
param_distributions = {
    # Learning rate - between 0.01 and 0.31
    "learning_rate": uniform(0.01, 0.3),
    # Tree structure parameters
    "max_depth": randint(3, 10),  # Between 3 and 9
    "min_child_weight": randint(1, 6),  # Between 1 and 5
    # Regularization parameters
    "reg_alpha": uniform(0, 1),  # L1 regularization
    "reg_lambda": uniform(1, 4),  # L2 regularization between 1 and 5
    # Sampling parameters - fix the ranges
    "subsample": uniform(0.6, 0.4),  # Between 0.6 and 1.0
    "colsample_bytree": uniform(0.6, 0.4),  # Between 0.6 and 1.0
    "colsample_bylevel": uniform(0.6, 0.4),  # Between 0.6 and 1.0
    # Number of estimators
    "n_estimators": randint(100, 900),  # Between 100 and 999
    # Gamma (minimum split loss)
    "gamma": uniform(0, 0.5),  # Between 0 and 0.5
}

print("Parameter distributions defined for RandomizedSearchCV")
print("Distributions:")
for param, dist in param_distributions.items():
    if hasattr(dist, "kwds"):
        if "loc" in dist.kwds and "scale" in dist.kwds:
            print(
                f"  {param}: uniform({dist.kwds['loc']}, {dist.kwds['loc'] + dist.kwds['scale']})"
            )
        elif "low" in dist.kwds and "high" in dist.kwds:
            print(
                f"  {param}: randint({dist.kwds['low']}, {dist.kwds['high']})"
            )
        else:
            print(f"  {param}: {type(dist).__name__}")
    else:
        print(f"  {param}: {type(dist).__name__}")

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

# ======== Perform RandomizedSearchCV ========
print(
    "======== Starting RandomizedSearchCV (this may take several minutes) ========"
)
start_time = time.time()

# Base XGBoost classifier
base_xgb = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,  # Use all available cores
    verbosity=0,  # Reduce XGBoost output
)

# RandomizedSearchCV with 100 iterations (good balance of thoroughness vs speed)
n_iter = 100
print(f"======== Running {n_iter} random parameter combinations ========")

random_search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_distributions,
    n_iter=n_iter,  # Number of parameter combinations to try
    scoring="roc_auc",  # Primary metric for selection
    cv=cv_strategy,
    n_jobs=-1,  # Use all available cores for parallel processing
    verbose=1,
    random_state=42,
    return_train_score=True,
)

random_search.fit(X_train, y_train)

end_time = time.time()
print(
    f"======== RandomizedSearchCV completed in {end_time - start_time:.2f} seconds ========"
)

# ======== Extract best model and results ========
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = random_search.best_score_

print("======== Best Model Results ========")
print(f"Best CV ROC AUC Score: {best_score:.4f}")
print("Best Parameters:")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"  {param}: {value:.4f}")
    else:
        print(f"  {param}: {value}")

# ======== Cross-validation with multiple metrics ========
print(
    "======== Performing detailed cross-validation with multiple metrics ========"
)

cv_scores = {}
for metric_name, metric_scorer in scoring.items():
    scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring=metric_scorer,
        n_jobs=-1,
    )
    cv_scores[metric_name] = scores
    print(
        f"{metric_name.upper():>12}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
    )

# ======== Analyze RandomizedSearchCV results in detail ========
print("======== Detailed RandomizedSearchCV Analysis ========")
cv_results_df = pd.DataFrame(random_search.cv_results_)

# Top 10 parameter combinations
top_10_results = cv_results_df.nlargest(10, "mean_test_score")[
    ["mean_test_score", "std_test_score", "params"]
]
print("\nTop 10 parameter combinations:")
for idx, row in top_10_results.iterrows():
    print(
        f"Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score'] * 2:.4f})"
    )
    params_str = ", ".join(
        [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row["params"].items()
        ]
    )
    print(f"       Params: {params_str}")

# ======== Train final model with early stopping ========
print(
    "======== Training final model with early stopping on complete training set ========"
)

# For final training, we'll use a portion of training data for early stopping
from sklearn.model_selection import train_test_split

# Split training data for early stopping
X_train_fit, X_train_val, y_train_fit, y_train_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Create final model with best parameters
final_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)

# Train final model on all training data
final_model.fit(X_train, y_train)

print("Final model trained on complete training set")

# ======== Evaluate on validation set if available ========
if X_val is not None and y_val is not None:
    print("======== Evaluating on 2024 validation set ========")

    # Predictions with best model
    y_proba = final_model.predict_proba(X_val)[:, 1]
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
feature_importance_values = final_model.feature_importances_

feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": feature_importance_values,
    }
).sort_values("importance", ascending=False)

print("Top 15 most important features:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:>25}: {row['importance']:>8.4f}")

# ======== Model Performance Summary ========
print("======== Final Model Performance Summary ========")
summary = {
    "model_type": "XGBoost with RandomizedSearchCV",
    "best_cv_score": best_score,
    "best_parameters": best_params,
    "cv_scores": {
        metric: {"mean": scores.mean(), "std": scores.std()}
        for metric, scores in cv_scores.items()
    },
    "training_time_seconds": end_time - start_time,
    "total_parameter_combinations_tested": n_iter,
    "best_iteration": getattr(final_model, "best_iteration", None),
    "validation_results": validation_results,
    "feature_importance": feature_importance.head(10).to_dict("records"),
}

print(f"Model Type: {summary['model_type']}")
print(f"Best CV ROC AUC: {summary['best_cv_score']:.4f}")
print(f"Training Time: {summary['training_time_seconds']:.2f} seconds")
print(
    f"Parameter Combinations Tested: {summary['total_parameter_combinations_tested']}"
)
if summary["best_iteration"]:
    print(f"Best Iteration (Early Stopping): {summary['best_iteration']}")

# ======== Save enhanced model and metadata ========
print("======== Saving enhanced model with cross-validation results ========")
os.makedirs("assets", exist_ok=True)

# Enhanced model data with CV results
enhanced_model_data = {
    "model": final_model,
    "feature_names": list(X_train.columns),
    "best_params": best_params,
    "cv_results": cv_results_df.to_dict("records"),
    "best_cv_score": best_score,
    "cv_scores_by_metric": cv_scores,
    "feature_importance": feature_importance.to_dict("records"),
    "model_summary": summary,
    "random_search_results": {
        "best_score": random_search.best_score_,
        "best_params": random_search.best_params_,
        "best_index": random_search.best_index_,
    },
}

# Save to pickle
with open("assets/xgb_playoffs_cv.pkl", "wb") as f:
    pickle.dump(enhanced_model_data, f)

# Also save a JSON summary for easy inspection
import json

json_summary = {
    "model_type": summary["model_type"],
    "best_cv_score": float(summary["best_cv_score"]),
    "best_parameters": {
        k: float(v)
        if isinstance(v, (np.floating, float))
        else int(v)
        if isinstance(v, (np.integer, int))
        else v
        for k, v in best_params.items()
    },
    "training_time_seconds": float(summary["training_time_seconds"]),
    "total_parameter_combinations_tested": int(
        summary["total_parameter_combinations_tested"]
    ),
    "best_iteration": summary["best_iteration"],
    "top_10_features": [
        {
            "feature": row["feature"],
            "importance": float(row["importance"]),
        }
        for _, row in feature_importance.head(10).iterrows()
    ],
}

with open("assets/xgb_playoffs_cv_summary.json", "w") as f:
    json.dump(json_summary, f, indent=2)

print(
    "======== Enhanced XGBoost model saved to assets/xgb_playoffs_cv.pkl ========"
)
print(
    "======== Model summary saved to assets/xgb_playoffs_cv_summary.json ========"
)
print("======== Cross-validation enhanced XGBoost training completed! ========")
