# ======== Enhanced Model Utilities for Cross-Validated Models ========
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_xgboost_cv_model(path="assets/xgb_playoffs_cv.pkl"):
    """Load XGBoost cross-validated model from pickle file"""
    with open(path, "rb") as f:
        model_data = pickle.load(f)
    return (
        model_data["model"],
        model_data["feature_names"],
        model_data["model_summary"],
        model_data.get("cv_results", []),
    )


def load_logistic_cv_model(path="assets/logistic_playoffs_cv.pkl"):
    """Load Logistic Regression cross-validated model and scaler from pickle file"""
    with open(path, "rb") as f:
        model_data = pickle.load(f)
    return (
        model_data["model"],
        model_data["scaler"],
        model_data["feature_names"],
        model_data["model_summary"],
        model_data.get("cv_results", []),
    )


def load_both_cv_models():
    """
    Load both cross-validated models
    Returns tuple: (xgb_data, logistic_data)
    """
    try:
        xgb_data = load_xgboost_cv_model()
    except FileNotFoundError:
        print("Warning: XGBoost CV model not found")
        xgb_data = None

    try:
        logistic_data = load_logistic_cv_model()
    except FileNotFoundError:
        print("Warning: Logistic CV model not found")
        logistic_data = None

    return xgb_data, logistic_data


def preprocess_for_cv_models(df):
    """
    Preprocess data for both cross-validated models
    Returns X (features), y (target if available), team_info
    """
    df = df.copy()

    # Store team info before dropping
    team_info = df[["TEAM", "LEAGUE"]].copy() if "TEAM" in df.columns else None

    # Drop unused columns - handle missing columns gracefully
    cols_to_drop = ["TEAM"]
    if "WON_WORLD_SERIES" in df.columns:
        cols_to_drop.append("WON_WORLD_SERIES")
    df = df.drop(columns=cols_to_drop)

    # Encode league as binary
    df["LEAGUE"] = df["LEAGUE"].map({"AL": 0, "NL": 1})

    # Target → int
    if "MADE_PLAYOFFS" in df.columns:
        df["MADE_PLAYOFFS"] = df["MADE_PLAYOFFS"].astype(int)

    # Convert any object‐typed columns to float
    for c in df.select_dtypes("object").columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )

    # Fill missing
    df = df.fillna(0)

    # Split features/target
    if "MADE_PLAYOFFS" in df.columns:
        X = df.drop(columns=["MADE_PLAYOFFS"])
        y = df["MADE_PLAYOFFS"]
        return X, y, team_info
    else:
        return df, None, team_info


def get_cv_model_predictions(X, xgb_data=None, logistic_data=None):
    """
    Get predictions from both cross-validated models
    Returns dictionary with probabilities and binary predictions
    """
    predictions = {}

    # XGBoost predictions
    if xgb_data is not None:
        xgb_model, feature_names, _, _ = xgb_data
        # Ensure feature order matches training
        X_xgb = X[feature_names] if isinstance(X, pd.DataFrame) else X
        xgb_proba = xgb_model.predict_proba(X_xgb)[:, 1]
        xgb_pred = (xgb_proba > 0.5).astype(int)
        predictions.update({"xgb_proba": xgb_proba, "xgb_pred": xgb_pred})

    # Logistic Regression predictions
    if logistic_data is not None:
        logistic_model, scaler, feature_names, _, _ = logistic_data
        # Ensure feature order matches training
        X_logistic = X[feature_names] if isinstance(X, pd.DataFrame) else X
        X_scaled = scaler.transform(X_logistic)
        logistic_proba = logistic_model.predict_proba(X_scaled)[:, 1]
        logistic_pred = (logistic_proba > 0.5).astype(int)
        predictions.update(
            {
                "logistic_proba": logistic_proba,
                "logistic_pred": logistic_pred,
            }
        )

    return predictions


def compare_cv_model_metrics(y_true, predictions):
    """
    Compare metrics between cross-validated models
    Returns dictionary with metrics for both models
    """
    metrics = {}

    # Check if we have valid data for metrics calculation
    if y_true is None or len(np.unique(y_true)) < 2:
        # Return placeholder metrics for invalid data
        for model_name in ["xgb", "logistic"]:
            metrics[model_name] = {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1_score": float("nan"),
                "roc_auc": float("nan"),
                "confusion_matrix": np.array([[0, 0], [0, 0]]),
            }
        return metrics

    for model_name in ["xgb", "logistic"]:
        proba_key = f"{model_name}_proba"
        pred_key = f"{model_name}_pred"

        if proba_key in predictions and pred_key in predictions:
            y_proba = predictions[proba_key]
            y_pred = predictions[pred_key]

            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                roc_auc = float("nan")

            metrics[model_name] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc,
                "confusion_matrix": confusion_matrix(y_true, y_pred),
            }

    return metrics


def get_cv_feature_importance_comparison(xgb_data=None, logistic_data=None):
    """
    Compare feature importance between cross-validated models
    Returns DataFrame with importance scores from both models
    """
    importance_data = []

    # Get feature names (use XGBoost if available, otherwise logistic)
    if xgb_data is not None:
        feature_names = xgb_data[1]
    elif logistic_data is not None:
        feature_names = logistic_data[2]
    else:
        return pd.DataFrame()

    # Initialize DataFrame
    importance_df = pd.DataFrame({"feature": feature_names})

    # XGBoost importance
    if xgb_data is not None:
        xgb_model = xgb_data[0]
        xgb_importance = xgb_model.feature_importances_
        importance_df["xgb_importance"] = xgb_importance
    else:
        importance_df["xgb_importance"] = 0

    # Logistic Regression coefficients (absolute values)
    if logistic_data is not None:
        logistic_model = logistic_data[0]
        logistic_coef = np.abs(logistic_model.coef_[0])
        importance_df["logistic_importance"] = logistic_coef
    else:
        importance_df["logistic_importance"] = 0

    # Normalize importance scores to 0-1 scale for comparison
    if importance_df["xgb_importance"].max() > 0:
        importance_df["xgb_importance_norm"] = (
            importance_df["xgb_importance"]
            / importance_df["xgb_importance"].max()
        )
    else:
        importance_df["xgb_importance_norm"] = 0

    if importance_df["logistic_importance"].max() > 0:
        importance_df["logistic_importance_norm"] = (
            importance_df["logistic_importance"]
            / importance_df["logistic_importance"].max()
        )
    else:
        importance_df["logistic_importance_norm"] = 0

    # Calculate average normalized importance
    importance_df["avg_importance"] = (
        importance_df["xgb_importance_norm"]
        + importance_df["logistic_importance_norm"]
    ) / 2

    return importance_df.sort_values("avg_importance", ascending=False)


def create_cv_prediction_comparison_df(team_info, predictions, y_true=None):
    """
    Create a DataFrame comparing predictions from both cross-validated models
    """
    # Determine number of predictions
    n_predictions = len(next(iter(predictions.values())))

    df = pd.DataFrame(
        {
            "team": (
                team_info["TEAM"]
                if team_info is not None and "TEAM" in team_info.columns
                else range(n_predictions)
            ),
            "league": (
                team_info["LEAGUE"]
                if team_info is not None and "LEAGUE" in team_info.columns
                else "Unknown"
            ),
        }
    )

    # Add predictions for each model
    for key in [
        "xgb_probability",
        "logistic_probability",
        "xgb_prediction",
        "logistic_prediction",
    ]:
        model_key = key.replace("_probability", "_proba").replace(
            "_prediction", "_pred"
        )
        if model_key in predictions:
            df[key] = predictions[model_key]

    # Add comparisons if both models available
    if "xgb_probability" in df.columns and "logistic_probability" in df.columns:
        df["prob_difference"] = (
            df["xgb_probability"] - df["logistic_probability"]
        )

    if "xgb_prediction" in df.columns and "logistic_prediction" in df.columns:
        df["models_agree"] = df["xgb_prediction"] == df["logistic_prediction"]

    # Add actual results if available
    if y_true is not None:
        df["actual"] = y_true
        if "xgb_prediction" in df.columns:
            df["xgb_correct"] = df["xgb_prediction"] == df["actual"]
        if "logistic_prediction" in df.columns:
            df["logistic_correct"] = df["logistic_prediction"] == df["actual"]
        if "xgb_correct" in df.columns and "logistic_correct" in df.columns:
            df["both_correct"] = df["xgb_correct"] & df["logistic_correct"]

    return df


def get_cv_model_summaries():
    """
    Get performance summaries for both cross-validated models
    """
    summaries = {}

    # Try to load both models
    xgb_data, logistic_data = load_both_cv_models()

    if xgb_data is not None:
        summaries["XGBoost"] = xgb_data[2]  # model_summary

    if logistic_data is not None:
        summaries["Logistic"] = logistic_data[3]  # model_summary

    return summaries


def compare_cv_training_results():
    """
    Compare cross-validation training results between models
    """
    summaries = get_cv_model_summaries()
    comparison = {}

    if "XGBoost" in summaries and "Logistic" in summaries:
        xgb_summary = summaries["XGBoost"]
        logistic_summary = summaries["Logistic"]

        comparison = {
            "cv_scores": {
                "XGBoost": xgb_summary.get("best_cv_score", float("nan")),
                "Logistic": logistic_summary.get("best_cv_score", float("nan")),
            },
            "training_time": {
                "XGBoost": xgb_summary.get(
                    "training_time_seconds", float("nan")
                ),
                "Logistic": logistic_summary.get(
                    "training_time_seconds", float("nan")
                ),
            },
            "parameter_combinations_tested": {
                "XGBoost": xgb_summary.get(
                    "total_parameter_combinations_tested", 0
                ),
                "Logistic": logistic_summary.get(
                    "total_parameter_combinations_tested", 0
                ),
            },
            "best_parameters": {
                "XGBoost": xgb_summary.get("best_parameters", {}),
                "Logistic": logistic_summary.get("best_parameters", {}),
            },
        }

        # Determine winner
        xgb_score = comparison["cv_scores"]["XGBoost"]
        logistic_score = comparison["cv_scores"]["Logistic"]

        if not np.isnan(xgb_score) and not np.isnan(logistic_score):
            if xgb_score > logistic_score:
                comparison["better_model"] = "XGBoost"
                comparison["score_difference"] = xgb_score - logistic_score
            elif logistic_score > xgb_score:
                comparison["better_model"] = "Logistic"
                comparison["score_difference"] = logistic_score - xgb_score
            else:
                comparison["better_model"] = "Tie"
                comparison["score_difference"] = 0.0

    return comparison


def get_ensemble_cv_prediction(predictions, method="average", weights=None):
    """
    Create ensemble predictions from both cross-validated models
    """
    available_models = []
    probabilities = []

    if "xgb_proba" in predictions:
        available_models.append("xgb")
        probabilities.append(predictions["xgb_proba"])

    if "logistic_proba" in predictions:
        available_models.append("logistic")
        probabilities.append(predictions["logistic_proba"])

    if len(probabilities) == 0:
        raise ValueError("No model predictions available for ensemble")

    if len(probabilities) == 1:
        # Only one model available
        ensemble_proba = probabilities[0]
    else:
        # Multiple models available
        if method == "average":
            ensemble_proba = np.mean(probabilities, axis=0)
        elif method == "weighted":
            if weights is None:
                weights = [0.6, 0.4]  # Default: slightly favor first model
            ensemble_proba = np.average(probabilities, axis=0, weights=weights)
        else:
            raise ValueError("Method must be 'average' or 'weighted'")

    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    return {
        "ensemble_proba": ensemble_proba,
        "ensemble_pred": ensemble_pred,
        "models_used": available_models,
    }


def analyze_cv_model_stability(cv_results_list, model_name):
    """
    Analyze cross-validation stability for a model
    """
    if not cv_results_list:
        return {"error": "No CV results available"}

    cv_df = pd.DataFrame(cv_results_list)

    if "mean_test_score" not in cv_df.columns:
        return {"error": "No test scores in CV results"}

    stability_analysis = {
        "model_name": model_name,
        "mean_cv_score": cv_df["mean_test_score"].mean(),
        "std_cv_score": cv_df["mean_test_score"].std(),
        "min_cv_score": cv_df["mean_test_score"].min(),
        "max_cv_score": cv_df["mean_test_score"].max(),
        "cv_score_range": cv_df["mean_test_score"].max()
        - cv_df["mean_test_score"].min(),
        "coefficient_of_variation": cv_df["mean_test_score"].std()
        / cv_df["mean_test_score"].mean(),
        "top_10_percent_threshold": cv_df["mean_test_score"].quantile(0.9),
        "stable_high_performers": len(
            cv_df[
                cv_df["mean_test_score"]
                >= cv_df["mean_test_score"].quantile(0.9)
            ]
        ),
    }

    return stability_analysis


def get_roc_curve_data(y_true, predictions):
    """
    Get ROC curve data for plotting
    """
    roc_data = {}

    for model_name in ["xgb", "logistic"]:
        proba_key = f"{model_name}_proba"
        if proba_key in predictions and y_true is not None:
            try:
                fpr, tpr, thresholds = roc_curve(y_true, predictions[proba_key])
                auc_score = roc_auc_score(y_true, predictions[proba_key])
                roc_data[model_name] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                    "auc": auc_score,
                }
            except ValueError:
                roc_data[model_name] = None

    return roc_data


def save_cv_model_comparison_report(
    output_path="assets/cv_model_comparison_report.json",
):
    """
    Generate and save comprehensive comparison report
    """
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_summaries": get_cv_model_summaries(),
        "training_comparison": compare_cv_training_results(),
    }

    # Add stability analysis
    xgb_data, logistic_data = load_both_cv_models()

    if xgb_data is not None:
        cv_results = xgb_data[3]
        report["xgb_stability"] = analyze_cv_model_stability(
            cv_results, "XGBoost"
        )

    if logistic_data is not None:
        cv_results = logistic_data[4]
        report["logistic_stability"] = analyze_cv_model_stability(
            cv_results, "Logistic"
        )

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def load_cv_model_comparison_report(
    file_path="assets/cv_model_comparison_report.json",
):
    """
    Load saved comparison report
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"Report not found at {file_path}"}


# Import os for makedirs
import os
