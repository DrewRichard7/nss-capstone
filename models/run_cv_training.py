#!/usr/bin/env python3
# ======== Cross-Validation Training Runner ========
"""
This script runs both cross-validated models (Logistic Regression and XGBoost)
and generates a comprehensive comparison report.
"""

import os
import sys
import time
from datetime import datetime

# Add the project root to Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_logistic_cv_training():
    """Run the logistic regression cross-validation training"""
    print("=" * 80)
    print("STARTING LOGISTIC REGRESSION CROSS-VALIDATION TRAINING")
    print("=" * 80)

    start_time = time.time()

    try:
        # Import and run the logistic CV training
        import subprocess

        result = subprocess.run(
            [sys.executable, "models/logistic_model_cv.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        if result.returncode == 0:
            print("✅ Logistic Regression CV training completed successfully!")
            print(
                "STDOUT:",
                result.stdout[-500:]
                if len(result.stdout) > 500
                else result.stdout,
            )
        else:
            print("❌ Logistic Regression CV training failed!")
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running Logistic Regression CV training: {e}")
        return False

    end_time = time.time()
    print(f"Logistic CV training took {end_time - start_time:.2f} seconds")
    return True


def run_xgboost_cv_training():
    """Run the XGBoost cross-validation training"""
    print("=" * 80)
    print("STARTING XGBOOST CROSS-VALIDATION TRAINING")
    print("=" * 80)

    start_time = time.time()

    try:
        # Import and run the XGBoost CV training
        import subprocess

        result = subprocess.run(
            [sys.executable, "models/xgb_model_cv.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        if result.returncode == 0:
            print("✅ XGBoost CV training completed successfully!")
            print(
                "STDOUT:",
                result.stdout[-500:]
                if len(result.stdout) > 500
                else result.stdout,
            )
        else:
            print("❌ XGBoost CV training failed!")
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running XGBoost CV training: {e}")
        return False

    end_time = time.time()
    print(f"XGBoost CV training took {end_time - start_time:.2f} seconds")
    return True


def generate_comparison_report():
    """Generate comprehensive comparison report"""
    print("=" * 80)
    print("GENERATING CROSS-VALIDATION COMPARISON REPORT")
    print("=" * 80)

    try:
        from models.model_utils_cv import (
            compare_cv_training_results,
            save_cv_model_comparison_report,
        )

        # Generate and save comparison report
        report = save_cv_model_comparison_report()

        print("✅ Comparison report generated successfully!")

        # Display summary
        training_comparison = compare_cv_training_results()

        if training_comparison:
            print("\n" + "=" * 60)
            print("CROSS-VALIDATION RESULTS SUMMARY")
            print("=" * 60)

            cv_scores = training_comparison.get("cv_scores", {})
            if cv_scores:
                print(
                    f"XGBoost CV ROC AUC:     {cv_scores.get('XGBoost', 'N/A'):.4f}"
                )
                print(
                    f"Logistic CV ROC AUC:    {cv_scores.get('Logistic', 'N/A'):.4f}"
                )

                if "better_model" in training_comparison:
                    print(
                        f"Better Model:           {training_comparison['better_model']}"
                    )
                    if "score_difference" in training_comparison:
                        print(
                            f"Score Difference:       {training_comparison['score_difference']:.4f}"
                        )

            training_times = training_comparison.get("training_time", {})
            if training_times:
                print(
                    f"XGBoost Training Time:  {training_times.get('XGBoost', 'N/A'):.2f} seconds"
                )
                print(
                    f"Logistic Training Time: {training_times.get('Logistic', 'N/A'):.2f} seconds"
                )

            param_counts = training_comparison.get(
                "parameter_combinations_tested", {}
            )
            if param_counts:
                print(
                    f"XGBoost Param Combos:   {param_counts.get('XGBoost', 'N/A')}"
                )
                print(
                    f"Logistic Param Combos:  {param_counts.get('Logistic', 'N/A')}"
                )

        return True

    except Exception as e:
        print(f"❌ Error generating comparison report: {e}")
        return False


def validate_data_availability():
    """Check if training data is available"""
    import glob

    pattern = "data/mlb_team_stats_*_pre_all_star.csv"
    all_files = glob.glob(pattern)

    if len(all_files) == 0:
        print("❌ No training data found!")
        print(f"Looking for pattern: {pattern}")
        print("Please run the data collection script first:")
        print("  python defs/baseball.py")
        print("or use FirstTimeRun.sh for complete setup")
        return False

    print(f"✅ Found {len(all_files)} data files for training")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ["sklearn", "xgboost", "pandas", "numpy", "scipy"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("Please install them using:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False

    print("✅ All required packages are available")
    return True


def main():
    """Main execution function"""
    print("=" * 80)
    print("CROSS-VALIDATION TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    overall_start_time = time.time()

    # Preliminary checks
    print("Performing preliminary checks...")

    if not check_dependencies():
        print("❌ Dependency check failed. Exiting.")
        return 1

    if not validate_data_availability():
        print("❌ Data validation failed. Exiting.")
        return 1

    print("✅ All preliminary checks passed!")
    print()

    # Track success of each step
    logistic_success = False
    xgboost_success = False

    # Run Logistic Regression CV training
    try:
        logistic_success = run_logistic_cv_training()
    except Exception as e:
        print(f"❌ Unexpected error in Logistic CV training: {e}")

    print()

    # Run XGBoost CV training
    try:
        xgboost_success = run_xgboost_cv_training()
    except Exception as e:
        print(f"❌ Unexpected error in XGBoost CV training: {e}")

    print()

    # Generate comparison report if at least one model succeeded
    if logistic_success or xgboost_success:
        try:
            report_success = generate_comparison_report()
        except Exception as e:
            print(f"❌ Unexpected error generating report: {e}")
            report_success = False
    else:
        print("❌ Both model trainings failed. Skipping comparison report.")
        report_success = False

    # Final summary
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(
        f"Logistic Regression CV: {'✅ SUCCESS' if logistic_success else '❌ FAILED'}"
    )
    print(
        f"XGBoost CV:            {'✅ SUCCESS' if xgboost_success else '❌ FAILED'}"
    )
    print(
        f"Comparison Report:     {'✅ SUCCESS' if report_success else '❌ FAILED'}"
    )
    print(f"Total Pipeline Time:   {total_time:.2f} seconds")
    print(
        f"Completed at:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Check for output files
    print("\nGenerated Files:")
    output_files = [
        "assets/logistic_playoffs_cv.pkl",
        "assets/logistic_playoffs_cv_summary.json",
        "assets/xgb_playoffs_cv.pkl",
        "assets/xgb_playoffs_cv_summary.json",
        "assets/cv_model_comparison_report.json",
    ]

    for file_path in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"  ❌ {file_path} (not found)")

    # Return appropriate exit code
    if logistic_success and xgboost_success and report_success:
        print("\n🎉 All components completed successfully!")
        return 0
    elif logistic_success or xgboost_success:
        print("\n⚠️  Pipeline completed with some failures.")
        return 1
    else:
        print("\n💥 Pipeline failed completely.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
