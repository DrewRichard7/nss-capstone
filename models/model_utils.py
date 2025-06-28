# ======== Model Utilities for Comparison ========
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_xgboost_model(path="assets/xgb_playoffs_cv.pkl"):
    """Load XGBoost model from pickle file (defaults to cross-validated model)"""
    try:
        # Try to load CV model first
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        # Check if it's the new CV format
        if isinstance(model_data, dict) and "model" in model_data:
            return model_data["model"]
        else:
            # Old format - raw model
            bst = xgb.Booster()
            bst.load_model(model_data)
            return bst
    except FileNotFoundError:
        # Fallback to original model if CV model not found
        fallback_path = "assets/xgb_playoffs.pkl"
        with open(fallback_path, "rb") as f:
            raw = pickle.load(f)
        bst = xgb.Booster()
        bst.load_model(raw)
        return bst


def load_logistic_model(path="assets/logistic_playoffs_cv.pkl"):
    """Load Logistic Regression model and scaler from pickle file (defaults to cross-validated model)"""
    try:
        # Try to load CV model first
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        return (
            model_data["model"],
            model_data["scaler"],
            model_data["feature_names"],
        )
    except FileNotFoundError:
        # Fallback to original model if CV model not found
        fallback_path = "assets/logistic_playoffs.pkl"
        with open(fallback_path, "rb") as f:
            model_data = pickle.load(f)
        return (
            model_data["model"],
            model_data["scaler"],
            model_data["feature_names"],
        )


def preprocess_for_models(df):
    """
    Preprocess data for both models
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


def get_model_predictions(X, xgb_model, logistic_model, scaler):
    """
    Get predictions from both models
    Returns dictionary with probabilities and binary predictions
    """
    # XGBoost predictions
    dmatrix = xgb.DMatrix(X)
    xgb_proba = xgb_model.predict(dmatrix)
    xgb_pred = (xgb_proba > 0.5).astype(int)

    # Logistic Regression predictions
    X_scaled = scaler.transform(X)
    logistic_proba = logistic_model.predict_proba(X_scaled)[:, 1]
    logistic_pred = (logistic_proba > 0.5).astype(int)

    return {
        "xgb_proba": xgb_proba,
        "xgb_pred": xgb_pred,
        "logistic_proba": logistic_proba,
        "logistic_pred": logistic_pred,
    }


def compare_model_metrics(y_true, predictions):
    """
    Compare metrics between models
    Returns dictionary with metrics for both models
    """
    import numpy as np

    metrics = {}

    # Check if we have valid data for metrics calculation
    if y_true is None or len(np.unique(y_true)) < 2 or np.sum(y_true) == 0:
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


def get_feature_importance_comparison(xgb_model, logistic_model, feature_names):
    """
    Compare feature importance between models
    Returns DataFrame with importance scores from both models
    """
    # XGBoost importance
    xgb_importance = xgb_model.get_score(importance_type="weight")

    # Logistic Regression coefficients (absolute values)
    logistic_coef = np.abs(logistic_model.coef_[0])

    # Create comparison DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "logistic_importance": logistic_coef,
        }
    )

    # Add XGBoost importance (some features might be missing)
    importance_df["xgb_importance"] = (
        importance_df["feature"].map(xgb_importance).fillna(0)
    )

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

    # Sort by average normalized importance
    importance_df["avg_importance"] = (
        importance_df["xgb_importance_norm"]
        + importance_df["logistic_importance_norm"]
    ) / 2

    return importance_df.sort_values("avg_importance", ascending=False)


def create_prediction_comparison_df(team_info, predictions, y_true=None):
    """
    Create a DataFrame comparing predictions from both models
    """
    df = pd.DataFrame(
        {
            "team": team_info["TEAM"]
            if team_info is not None
            else range(len(predictions["xgb_proba"])),
            "league": team_info["LEAGUE"]
            if team_info is not None
            else "Unknown",
            "xgb_probability": predictions["xgb_proba"],
            "logistic_probability": predictions["logistic_proba"],
            "xgb_prediction": predictions["xgb_pred"],
            "logistic_prediction": predictions["logistic_pred"],
        }
    )

    # Add probability difference
    df["prob_difference"] = df["xgb_probability"] - df["logistic_probability"]

    # Add agreement indicator
    df["models_agree"] = df["xgb_prediction"] == df["logistic_prediction"]

    # Add actual results if available
    if y_true is not None:
        df["actual"] = y_true
        df["xgb_correct"] = df["xgb_prediction"] == df["actual"]
        df["logistic_correct"] = df["logistic_prediction"] == df["actual"]
        df["both_correct"] = df["xgb_correct"] & df["logistic_correct"]

    return df


def get_model_agreement_stats(comparison_df):
    """
    Get statistics about model agreement
    """
    if len(comparison_df) == 0:
        return {
            "total_teams": 0,
            "agreement_count": 0,
            "disagreement_count": 0,
            "agreement_rate": 0.0,
            "avg_prob_difference": 0.0,
            "max_prob_difference": 0.0,
        }

    total_teams = len(comparison_df)
    agree_count = comparison_df["models_agree"].sum()
    agreement_rate = agree_count / total_teams if total_teams > 0 else 0.0

    stats = {
        "total_teams": total_teams,
        "agreement_count": agree_count,
        "disagreement_count": total_teams - agree_count,
        "agreement_rate": agreement_rate,
        "avg_prob_difference": comparison_df["prob_difference"].abs().mean(),
        "max_prob_difference": comparison_df["prob_difference"].abs().max(),
    }

    # Add accuracy comparison if actual results are available
    if (
        "actual" in comparison_df.columns
        and comparison_df["actual"].notna().any()
    ):
        stats["xgb_accuracy"] = comparison_df["xgb_correct"].mean()
        stats["logistic_accuracy"] = comparison_df["logistic_correct"].mean()
        stats["both_correct_rate"] = comparison_df["both_correct"].mean()

    return stats


def analyze_disagreements(comparison_df):
    """
    Analyze cases where models disagree
    """
    disagreements = comparison_df[~comparison_df["models_agree"]].copy()

    if len(disagreements) == 0:
        return pd.DataFrame()

    # Sort by probability difference (largest disagreements first)
    disagreements = disagreements.sort_values(
        "prob_difference", key=abs, ascending=False
    )

    return disagreements


def get_ensemble_prediction(predictions, method="average"):
    """
    Create ensemble predictions from both models
    """
    if method == "average":
        ensemble_proba = (
            predictions["xgb_proba"] + predictions["logistic_proba"]
        ) / 2
    elif method == "weighted":
        # Weight XGBoost slightly higher (can be adjusted)
        ensemble_proba = (
            0.6 * predictions["xgb_proba"] + 0.4 * predictions["logistic_proba"]
        )
    else:
        raise ValueError("Method must be 'average' or 'weighted'")

    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    return {
        "ensemble_proba": ensemble_proba,
        "ensemble_pred": ensemble_pred,
    }


def get_xgboost_hyperparameters(xgb_model):
    """
    Extract hyperparameters from trained XGBoost model
    """
    # Get attributes from the booster object
    import json

    # Get configuration and attributes
    config = xgb_model.save_config()
    config_dict = json.loads(config)

    # Extract parameters from the saved config
    learner = config_dict.get("learner", {})
    learner_params = learner.get("learner_model_param", {})
    objective_params = learner.get("objective", {})

    # Get booster attributes which contain actual training parameters
    attributes = xgb_model.attributes()

    # Extract key hyperparameters with actual values used during training
    hyperparams = {
        "objective": objective_params.get("name", "binary:logistic"),
        "learning_rate": float(learner_params.get("eta", "0.1")),
        "max_depth": int(learner_params.get("max_depth", "6")),
        "subsample": float(learner_params.get("subsample", "0.8")),
        "colsample_bytree": float(
            learner_params.get("colsample_bytree", "0.8")
        ),
        "tree_method": learner_params.get("tree_method", "hist"),
        "num_boost_round": xgb_model.num_boosted_rounds(),
        "random_state": int(learner_params.get("seed", "42")),
        "eval_metric": "logloss",
        "early_stopping_rounds": 10,  # From training script
    }

    return hyperparams


def get_logistic_hyperparameters(logistic_model):
    """
    Extract hyperparameters from trained Logistic Regression model
    """
    hyperparams = {
        "penalty": logistic_model.penalty,
        "C": logistic_model.C,
        "solver": logistic_model.solver,
        "max_iter": logistic_model.max_iter,
        "class_weight": str(logistic_model.class_weight),
        "random_state": logistic_model.random_state,
        "fit_intercept": logistic_model.fit_intercept,
        "intercept_scaling": logistic_model.intercept_scaling,
    }

    return hyperparams


def get_all_model_hyperparameters():
    """
    Get hyperparameters for all trained models
    Returns dictionary with hyperparameters for each model
    """
    try:
        # Load CV models first, fallback to original models
        try:
            # Try to load CV model data with hyperparameters
            with open("assets/xgb_playoffs_cv.pkl", "rb") as f:
                xgb_model_data = pickle.load(f)
            if (
                isinstance(xgb_model_data, dict)
                and "best_params" in xgb_model_data
            ):
                xgb_params = xgb_model_data["best_params"].copy()
                xgb_params["model_type"] = "XGBoost (Cross-Validated)"
                xgb_params["cv_score"] = xgb_model_data.get(
                    "best_cv_score", "N/A"
                )
            else:
                xgb_model = load_xgboost_model()
                xgb_params = get_xgboost_hyperparameters(xgb_model)
        except:
            xgb_model = load_xgboost_model()
            xgb_params = get_xgboost_hyperparameters(xgb_model)

        try:
            # Try to load CV model data with hyperparameters
            with open("assets/logistic_playoffs_cv.pkl", "rb") as f:
                logistic_model_data = pickle.load(f)
            if (
                isinstance(logistic_model_data, dict)
                and "best_params" in logistic_model_data
            ):
                logistic_params = logistic_model_data["best_params"].copy()
                logistic_params["model_type"] = (
                    "Logistic Regression (Cross-Validated)"
                )
                logistic_params["cv_score"] = logistic_model_data.get(
                    "best_cv_score", "N/A"
                )
            else:
                logistic_model, _, _ = load_logistic_model()
                logistic_params = get_logistic_hyperparameters(logistic_model)
        except:
            logistic_model, _, _ = load_logistic_model()
            logistic_params = get_logistic_hyperparameters(logistic_model)

        return {"XGBoost": xgb_params, "Logistic Regression": logistic_params}
    except Exception as e:
        return {"error": f"Could not load model hyperparameters: {str(e)}"}
