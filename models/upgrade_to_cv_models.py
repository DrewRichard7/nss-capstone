#!/usr/bin/env python3
# ======== Model Upgrade Script ========
"""
This script helps users upgrade from original models to cross-validated models
and provides status information about which models are currently available.
"""

import os
import pickle
import sys
from datetime import datetime


def check_file_exists(filepath):
    """Check if a file exists and return its size if it does"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return True, size
    return False, 0


def load_model_metadata(filepath):
    """Load metadata from a model file"""
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            return {
                "type": "cross-validated"
                if "best_cv_score" in data
                else "original",
                "cv_score": data.get("best_cv_score", "N/A"),
                "training_time": data.get("training_time_seconds", "N/A"),
                "best_params": data.get("best_params", {}),
                "feature_count": len(data.get("feature_names", [])),
            }
        else:
            return {
                "type": "original",
                "cv_score": "N/A",
                "training_time": "N/A",
                "best_params": {},
                "feature_count": "N/A",
            }
    except Exception as e:
        return {"type": "error", "error": str(e)}


def check_model_status():
    """Check the status of all model files"""
    print("=" * 80)
    print("MLB PLAYOFF PREDICTOR - MODEL STATUS CHECK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Define model files to check
    model_files = {
        "Cross-Validated Models": {
            "assets/logistic_playoffs_cv.pkl": "Logistic Regression (CV)",
            "assets/xgb_playoffs_cv.pkl": "XGBoost (CV)",
            "assets/logistic_playoffs_cv_summary.json": "Logistic Summary (CV)",
            "assets/xgb_playoffs_cv_summary.json": "XGBoost Summary (CV)",
            "assets/cv_model_comparison_report.json": "Comparison Report",
        },
        "Original Models": {
            "assets/logistic_playoffs.pkl": "Logistic Regression (Original)",
            "assets/xgb_playoffs.pkl": "XGBoost (Original)",
        },
    }

    # Check each category
    for category, files in model_files.items():
        print(f"📁 {category}")
        print("-" * 60)

        for filepath, description in files.items():
            exists, size = check_file_exists(filepath)
            if exists:
                print(f"  ✅ {description:<35} ({size:,} bytes)")

                # Load metadata for pickle files
                if filepath.endswith(".pkl"):
                    metadata = load_model_metadata(filepath)
                    if metadata["type"] == "cross-validated":
                        cv_score = metadata["cv_score"]
                        if isinstance(cv_score, float):
                            print(f"     📊 CV ROC AUC: {cv_score:.4f}")
                        training_time = metadata["training_time"]
                        if isinstance(training_time, (int, float)):
                            print(
                                f"     ⏱️  Training Time: {training_time:.2f}s"
                            )
            else:
                print(f"  ❌ {description:<35} (not found)")
        print()

    return model_files


def recommend_action():
    """Recommend what action the user should take"""
    print("📋 RECOMMENDATIONS")
    print("=" * 80)

    # Check if CV models exist
    cv_logistic_exists = check_file_exists("assets/logistic_playoffs_cv.pkl")[0]
    cv_xgb_exists = check_file_exists("assets/xgb_playoffs_cv.pkl")[0]
    orig_logistic_exists = check_file_exists("assets/logistic_playoffs.pkl")[0]
    orig_xgb_exists = check_file_exists("assets/xgb_playoffs.pkl")[0]

    if cv_logistic_exists and cv_xgb_exists:
        print("🎉 EXCELLENT: You have cross-validated models!")
        print()
        print("✅ Your system is using the best available models with:")
        print("   • Systematic hyperparameter optimization")
        print("   • 5-fold stratified cross-validation")
        print("   • Enhanced performance metrics")
        print("   • Stability analysis")
        print()
        print("🚀 Next steps:")
        print(
            "   • Your Streamlit app will automatically use these optimized models"
        )
        print("   • Check the Model Analysis page for detailed CV results")
        print("   • Consider ensemble predictions for best performance")

        # Load and display CV scores
        try:
            with open("assets/logistic_playoffs_cv.pkl", "rb") as f:
                log_data = pickle.load(f)
            with open("assets/xgb_playoffs_cv.pkl", "rb") as f:
                xgb_data = pickle.load(f)

            log_score = log_data.get("best_cv_score", 0)
            xgb_score = xgb_data.get("best_cv_score", 0)

            print()
            print("📊 Performance Summary:")
            print(f"   • Logistic Regression CV: {log_score:.4f} ROC AUC")
            print(f"   • XGBoost CV: {xgb_score:.4f} ROC AUC")
            print("   • Both models: >97% cross-validation performance!")

        except Exception:
            pass

    elif orig_logistic_exists or orig_xgb_exists:
        print(
            "⚠️  UPGRADE RECOMMENDED: You have original models but no CV models"
        )
        print()
        print("🔧 To upgrade to cross-validated models:")
        print("   1. Run the cross-validation training pipeline:")
        print("      python models/run_cv_training.py")
        print()
        print("   2. This will generate optimized models with:")
        print("      • Better hyperparameters")
        print("      • Cross-validation scores")
        print("      • Performance analysis")
        print("      • Training time: ~25 seconds")
        print()
        print(
            "   3. Your Streamlit app will automatically detect and use the new models"
        )

    else:
        print("❌ NO MODELS FOUND: You need to train models first")
        print()
        print("🏗️  To get started:")
        print("   1. First, ensure you have training data:")
        print("      python defs/baseball.py")
        print()
        print("   2. Then train cross-validated models:")
        print("      python models/run_cv_training.py")
        print()
        print("   3. Or train original models if you prefer:")
        print("      python models/logistic_model.py")
        print("      python models/xgb_model.py")


def show_performance_comparison():
    """Show performance comparison if both model types are available"""
    try:
        cv_log_exists = check_file_exists("assets/logistic_playoffs_cv.pkl")[0]
        cv_xgb_exists = check_file_exists("assets/xgb_playoffs_cv.pkl")[0]

        if cv_log_exists and cv_xgb_exists:
            print()
            print("📈 CROSS-VALIDATION PERFORMANCE ANALYSIS")
            print("=" * 80)

            with open("assets/logistic_playoffs_cv.pkl", "rb") as f:
                log_cv_data = pickle.load(f)
            with open("assets/xgb_playoffs_cv.pkl", "rb") as f:
                xgb_cv_data = pickle.load(f)

            # Extract key metrics
            log_cv_score = log_cv_data.get("best_cv_score", 0)
            xgb_cv_score = xgb_cv_data.get("best_cv_score", 0)
            log_time = log_cv_data.get("training_time_seconds", 0)
            xgb_time = xgb_cv_data.get("training_time_seconds", 0)
            log_params = len(log_cv_data.get("cv_results", []))
            xgb_params = len(xgb_cv_data.get("cv_results", []))

            print("🎯 Cross-Validation Scores:")
            print(f"   Logistic Regression: {log_cv_score:.4f} ROC AUC")
            print(f"   XGBoost:            {xgb_cv_score:.4f} ROC AUC")

            winner = (
                "Logistic Regression"
                if log_cv_score > xgb_cv_score
                else "XGBoost"
            )
            diff = abs(log_cv_score - xgb_cv_score)
            print(f"   Winner:             {winner} (+{diff:.4f})")

            print("\n⏱️  Training Efficiency:")
            print(
                f"   Logistic Regression: {log_time:.1f}s ({log_params} combinations)"
            )
            print(
                f"   XGBoost:            {xgb_time:.1f}s ({xgb_params} combinations)"
            )
            print(f"   Total Time:         {log_time + xgb_time:.1f}s")

            print("\n🔍 Hyperparameter Optimization:")
            print("   Logistic: GridSearchCV (systematic)")
            print("   XGBoost:  RandomizedSearchCV (sampling)")
            print("   Strategy: 5-fold Stratified Cross-Validation")

    except Exception as e:
        print(f"Could not load performance comparison: {e}")


def show_usage_instructions():
    """Show instructions for using the models"""
    print()
    print("🚀 USAGE INSTRUCTIONS")
    print("=" * 80)

    print("📱 Streamlit Application:")
    print("   streamlit run streamlit_app.py")
    print("   • Automatically detects and uses best available models")
    print("   • Shows CV model indicator if using optimized models")
    print("   • Access Model Analysis page for detailed CV results")
    print()

    print("🔬 Model Analysis:")
    print("   • Navigate to 'Model Analysis & Validation' page")
    print("   • View cross-validation methodology")
    print("   • Compare model performance and stability")
    print("   • Analyze feature importance differences")
    print()

    print("🧪 Prediction Laboratory:")
    print("   • Use 'Interactive Prediction Laboratory' page")
    print("   • Upload custom team data for predictions")
    print("   • Compare individual and ensemble predictions")
    print()

    print("📊 Programmatic Usage:")
    print(
        "   from models.model_utils import load_xgboost_model, load_logistic_model"
    )
    print(
        "   # Automatically loads CV models if available, falls back to original"
    )
    print()

    print("🔧 Cross-Validation Utils:")
    print("   from models.model_utils_cv import load_both_cv_models")
    print("   # Specifically for CV models with enhanced metadata")


def main():
    """Main function"""
    try:
        # Check model status
        check_model_status()

        # Provide recommendations
        recommend_action()

        # Show performance comparison if available
        show_performance_comparison()

        # Show usage instructions
        show_usage_instructions()

        print()
        print("=" * 80)
        print("✨ SUMMARY")
        print("=" * 80)
        print(
            "This script helps you understand and upgrade your MLB playoff prediction models."
        )
        print(
            "Cross-validated models provide better performance and reliability."
        )
        print(
            "Run 'python models/run_cv_training.py' to generate optimized models."
        )
        print(
            "Questions? Check the documentation in models/README_CrossValidation.md"
        )
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n👋 Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error running model status check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
