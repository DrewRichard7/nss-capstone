#!/usr/bin/env python3
# ======== Demonstration of Cross-Validated Models ========
"""
This script demonstrates how to use the cross-validated models for making predictions
and comparing their performance.
"""

import os
import sys

import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_utils_cv import (
    compare_cv_model_metrics,
    compare_cv_training_results,
    create_cv_prediction_comparison_df,
    get_cv_feature_importance_comparison,
    get_cv_model_predictions,
    get_ensemble_cv_prediction,
    get_roc_curve_data,
    load_both_cv_models,
    preprocess_for_cv_models,
)


def load_sample_data():
    """Load sample data for demonstration"""
    import glob

    # Find the most recent data file
    pattern = "data/mlb_team_stats_*_pre_all_star.csv"
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        print("No data files found!")
        return None

    # Use 2024 data for demonstration
    sample_file = None
    for file in all_files:
        if "2024" in file:
            sample_file = file
            break

    if sample_file is None:
        sample_file = all_files[-1]  # Use most recent file

    print(f"Loading sample data from: {sample_file}")
    return pd.read_csv(sample_file)


def demonstrate_model_loading():
    """Demonstrate loading cross-validated models"""
    print("=" * 60)
    print("LOADING CROSS-VALIDATED MODELS")
    print("=" * 60)

    xgb_data, logistic_data = load_both_cv_models()

    if xgb_data is not None:
        xgb_model, feature_names, model_summary, cv_results = xgb_data
        print("✅ XGBoost model loaded successfully")
        print(f"   Features: {len(feature_names)}")
        print(f"   Best CV Score: {model_summary['best_cv_score']:.4f}")
        print(
            f"   Training Time: {model_summary['training_time_seconds']:.2f}s"
        )
    else:
        print("❌ XGBoost model not found")

    if logistic_data is not None:
        logistic_model, scaler, feature_names, model_summary, cv_results = (
            logistic_data
        )
        print("✅ Logistic model loaded successfully")
        print(f"   Features: {len(feature_names)}")
        print(f"   Best CV Score: {model_summary['best_cv_score']:.4f}")
        print(
            f"   Training Time: {model_summary['training_time_seconds']:.2f}s"
        )
    else:
        print("❌ Logistic model not found")

    return xgb_data, logistic_data


def demonstrate_predictions(sample_data, xgb_data, logistic_data):
    """Demonstrate making predictions with both models"""
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)

    # Preprocess the data
    X, y, team_info = preprocess_for_cv_models(sample_data)

    print(f"Sample data shape: {X.shape}")
    if y is not None:
        print(f"Target distribution: {y.value_counts().to_dict()}")

    # Get predictions from both models
    predictions = get_cv_model_predictions(X, xgb_data, logistic_data)

    print(f"\nPredictions available: {list(predictions.keys())}")

    # Create comparison DataFrame
    comparison_df = create_cv_prediction_comparison_df(
        team_info, predictions, y
    )

    print("\nSample predictions (first 10 teams):")
    display_cols = ["team", "league"]
    if "xgb_probability" in comparison_df.columns:
        display_cols.extend(["xgb_probability", "xgb_prediction"])
    if "logistic_probability" in comparison_df.columns:
        display_cols.extend(["logistic_probability", "logistic_prediction"])
    if "actual" in comparison_df.columns:
        display_cols.append("actual")

    print(comparison_df[display_cols].head(10).to_string(index=False))

    return X, y, predictions, comparison_df


def demonstrate_model_comparison(y, predictions):
    """Demonstrate model performance comparison"""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    if y is not None:
        # Calculate metrics for both models
        metrics = compare_cv_model_metrics(y, predictions)

        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()} Model Performance:")
            for metric_name, value in model_metrics.items():
                if metric_name != "confusion_matrix":
                    print(f"  {metric_name:>12}: {value:.4f}")

        # ROC curve data
        roc_data = get_roc_curve_data(y, predictions)
        if roc_data:
            print("\nROC AUC Scores:")
            for model_name, data in roc_data.items():
                if data is not None:
                    print(f"  {model_name:>12}: {data['auc']:.4f}")

    # Training comparison
    training_comparison = compare_cv_training_results()
    if training_comparison:
        print("\nCross-Validation Training Comparison:")
        cv_scores = training_comparison.get("cv_scores", {})
        for model, score in cv_scores.items():
            print(f"  {model:>12} CV Score: {score:.4f}")

        if "better_model" in training_comparison:
            print(f"  Better Model: {training_comparison['better_model']}")
            print(
                f"  Score Difference: {training_comparison['score_difference']:.4f}"
            )


def demonstrate_feature_importance(xgb_data, logistic_data):
    """Demonstrate feature importance comparison"""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE COMPARISON")
    print("=" * 60)

    importance_df = get_cv_feature_importance_comparison(
        xgb_data, logistic_data
    )

    if not importance_df.empty:
        print("Top 15 most important features (normalized):")
        print(
            importance_df[
                [
                    "feature",
                    "xgb_importance_norm",
                    "logistic_importance_norm",
                    "avg_importance",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

        # Find features where models disagree most
        importance_df["importance_difference"] = abs(
            importance_df["xgb_importance_norm"]
            - importance_df["logistic_importance_norm"]
        )
        disagreement_df = importance_df.nlargest(10, "importance_difference")

        print("\nTop 10 features where models disagree most:")
        print(
            disagreement_df[
                [
                    "feature",
                    "xgb_importance_norm",
                    "logistic_importance_norm",
                    "importance_difference",
                ]
            ].to_string(index=False)
        )


def demonstrate_ensemble_predictions(predictions):
    """Demonstrate ensemble predictions"""
    print("\n" + "=" * 60)
    print("ENSEMBLE PREDICTIONS")
    print("=" * 60)

    if len(predictions) >= 2:
        # Average ensemble
        ensemble_avg = get_ensemble_cv_prediction(predictions, method="average")
        print("✅ Average ensemble created")
        print(f"   Models used: {ensemble_avg['models_used']}")

        # Weighted ensemble (favor XGBoost slightly)
        ensemble_weighted = get_ensemble_cv_prediction(
            predictions, method="weighted", weights=[0.6, 0.4]
        )
        print("✅ Weighted ensemble created (60% XGBoost, 40% Logistic)")

        # Compare ensemble predictions with individual models
        n_samples = min(10, len(ensemble_avg["ensemble_proba"]))

        print(
            f"\nEnsemble vs Individual Predictions (first {n_samples} samples):"
        )
        for i in range(n_samples):
            xgb_prob = (
                predictions.get("xgb_proba", [0])[i]
                if "xgb_proba" in predictions
                else 0
            )
            logistic_prob = (
                predictions.get("logistic_proba", [0])[i]
                if "logistic_proba" in predictions
                else 0
            )
            avg_prob = ensemble_avg["ensemble_proba"][i]
            weighted_prob = ensemble_weighted["ensemble_proba"][i]

            print(
                f"  Sample {i + 1:2d}: XGB={xgb_prob:.3f}, Log={logistic_prob:.3f}, Avg={avg_prob:.3f}, Wtd={weighted_prob:.3f}"
            )
    else:
        print("❌ Need both models for ensemble predictions")


def demonstrate_model_stability():
    """Demonstrate model stability analysis"""
    print("\n" + "=" * 60)
    print("MODEL STABILITY ANALYSIS")
    print("=" * 60)

    try:
        from models.model_utils_cv import load_cv_model_comparison_report

        report = load_cv_model_comparison_report()

        if "error" not in report:
            # XGBoost stability
            if "xgb_stability" in report:
                xgb_stability = report["xgb_stability"]
                print("XGBoost Stability:")
                print(f"  Mean CV Score: {xgb_stability['mean_cv_score']:.4f}")
                print(f"  Std CV Score:  {xgb_stability['std_cv_score']:.4f}")
                print(f"  Score Range:   {xgb_stability['cv_score_range']:.4f}")
                print(
                    f"  Coef of Var:   {xgb_stability['coefficient_of_variation']:.4f}"
                )
                print(
                    f"  High Performers: {xgb_stability['stable_high_performers']}"
                )

            # Logistic stability
            if "logistic_stability" in report:
                logistic_stability = report["logistic_stability"]
                print("\nLogistic Regression Stability:")
                print(
                    f"  Mean CV Score: {logistic_stability['mean_cv_score']:.4f}"
                )
                print(
                    f"  Std CV Score:  {logistic_stability['std_cv_score']:.4f}"
                )
                print(
                    f"  Score Range:   {logistic_stability['cv_score_range']:.4f}"
                )
                print(
                    f"  Coef of Var:   {logistic_stability['coefficient_of_variation']:.4f}"
                )
                print(
                    f"  High Performers: {logistic_stability['stable_high_performers']}"
                )
        else:
            print("❌ Could not load stability report")

    except Exception as e:
        print(f"❌ Error analyzing stability: {e}")


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("CROSS-VALIDATED MODELS DEMONSTRATION")
    print("=" * 80)

    # Load sample data
    sample_data = load_sample_data()
    if sample_data is None:
        print("❌ Could not load sample data. Exiting.")
        return

    # Load models
    xgb_data, logistic_data = demonstrate_model_loading()

    if xgb_data is None and logistic_data is None:
        print("❌ No models available. Please run the training pipeline first.")
        print("   python models/run_cv_training.py")
        return

    # Make predictions
    X, y, predictions, comparison_df = demonstrate_predictions(
        sample_data, xgb_data, logistic_data
    )

    # Compare model performance
    demonstrate_model_comparison(y, predictions)

    # Feature importance
    demonstrate_feature_importance(xgb_data, logistic_data)

    # Ensemble predictions
    demonstrate_ensemble_predictions(predictions)

    # Model stability
    demonstrate_model_stability()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)

    # Summary
    print("\nWhat we demonstrated:")
    print("✅ Loading cross-validated models")
    print("✅ Making predictions with both models")
    print("✅ Comparing model performance")
    print("✅ Analyzing feature importance")
    print("✅ Creating ensemble predictions")
    print("✅ Analyzing model stability")

    print("\nKey findings:")
    training_comparison = compare_cv_training_results()
    if training_comparison and "better_model" in training_comparison:
        better_model = training_comparison["better_model"]
        score_diff = training_comparison["score_difference"]
        print(f"• {better_model} performed better by {score_diff:.4f} ROC AUC")

    if not comparison_df.empty and "models_agree" in comparison_df.columns:
        agreement_rate = comparison_df["models_agree"].mean()
        print(f"• Models agree on {agreement_rate:.1%} of predictions")

    print("\nFiles generated during training:")
    output_files = [
        "assets/logistic_playoffs_cv.pkl",
        "assets/xgb_playoffs_cv.pkl",
        "assets/cv_model_comparison_report.json",
    ]

    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")


if __name__ == "__main__":
    main()
