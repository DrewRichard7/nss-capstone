import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from models.model_utils import (
    analyze_disagreements,
    compare_model_metrics,
    create_prediction_comparison_df,
    get_ensemble_prediction,
    get_feature_importance_comparison,
    get_model_agreement_stats,
    get_model_predictions,
    load_logistic_model,
    load_xgboost_model,
    preprocess_for_models,
)

# ======== Page config ========
st.set_page_config(
    page_title="Model Analysis & Validation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Machine Learning Model Analysis Hub")
st.markdown(
    "### Deep dive into model performance, validation, and algorithm comparison"
)

# Journey introduction
st.markdown("""
🎓 **Welcome to the Statistical Analysis Center!** This page is where the magic happens - you'll understand
how machine learning algorithms make predictions and learn to compare different approaches.

💡 **What you'll discover:**
- How accurate are our models? (Spoiler: Pretty good!)
- Which features matter most for predicting playoffs?
- When do XGBoost and Logistic Regression agree or disagree?
- How to interpret ROC curves and confusion matrices
""")


# ======== CACHED RESOURCE: load both models ========
@st.cache_resource
def load_models():
    """Load both XGBoost and Logistic Regression models"""
    try:
        xgb_model = load_xgboost_model()
        logistic_model, scaler, feature_names = load_logistic_model()
        return xgb_model, logistic_model, scaler, feature_names, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False


@st.cache_data
def get_available_years():
    """Get list of years available in the data directory"""
    data_files = glob.glob("data/mlb_team_stats_*_pre_all_star.csv")
    years = []
    for file in data_files:
        basename = os.path.basename(file)
        parts = basename.split("_")
        if len(parts) >= 4:
            try:
                year = int(parts[3])
                years.append(year)
            except ValueError:
                continue
    return sorted(years, reverse=True)


@st.cache_data
def load_raw_data(year):
    """Load raw data for a specific year"""
    filepath = f"data/mlb_team_stats_{year}_pre_all_star.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


# ======== Main Analysis Content ========

# Load models
xgb_model, logistic_model, scaler, feature_names, models_loaded = load_models()

if not models_loaded:
    st.error(
        "Failed to load models. Please ensure both models are trained and available."
    )
    st.stop()

available_years = get_available_years()

# Sidebar configuration
st.sidebar.subheader("🎛️ Analysis Configuration")

# Model selection for comparative analysis
st.sidebar.subheader("🤖 Primary Model Focus")
focus_model = st.sidebar.selectbox(
    "Choose model to highlight in analysis:",
    [
        "Compare All Models",
        "XGBoost Focus",
        "Logistic Regression Focus",
        "Ensemble Focus",
    ],
    index=0,
    help="This affects which model is emphasized in charts and summaries",
)

# Year selection
st.sidebar.subheader("📅 Year Selection")
# Default to 2024 if available, otherwise first available year
default_index = 0
if available_years and 2024 in available_years:
    default_index = available_years.index(2024)

selected_year = st.sidebar.selectbox(
    "Select year for validation analysis:",
    available_years,
    index=default_index if available_years else None,
    help="Choose a year with actual playoff results for meaningful statistical analysis",
)

if selected_year:
    # Load data for selected year
    data = load_raw_data(selected_year)

    if data is not None:
        # Preprocess data
        X, y, team_info = preprocess_for_models(data)

        # Check if we have valid playoff data
        has_valid_data = y is not None and len(y.unique()) > 1 and y.sum() > 0

        if not has_valid_data:
            st.warning(
                f"⚠️ {selected_year} data has no playoff results or invalid data. Showing predictions only."
            )
            if y is not None:
                st.info(
                    f"Playoff teams in data: {y.sum()} out of {len(y)} teams"
                )

        # Get predictions from both models
        predictions = get_model_predictions(
            X, xgb_model, logistic_model, scaler
        )

        # Create ensemble predictions
        ensemble_preds = get_ensemble_prediction(predictions, method="average")
        predictions.update(ensemble_preds)

        # Create comparison dataframe
        comparison_df = create_prediction_comparison_df(
            team_info, predictions, y if has_valid_data else None
        )

        # Get agreement statistics
        agreement_stats = get_model_agreement_stats(comparison_df)

        # ======== Model Performance Overview ========
        st.header(f"🎯 Model Performance Overview ({selected_year})")

        # Learning callout
        st.markdown("""
        📚 **Learning Moment**: We're about to see how well our AI models performed on *real* baseball data.
        These metrics tell us if our models are actually good at predicting playoffs or just getting lucky!
        """)

        if has_valid_data:
            # Calculate metrics for all models
            metrics_comparison = compare_model_metrics(y, predictions)

            # Add ensemble metrics (with error handling)
            try:
                ensemble_metrics = {
                    "accuracy": accuracy_score(y, predictions["ensemble_pred"]),
                    "roc_auc": roc_auc_score(y, predictions["ensemble_proba"]),
                }
            except ValueError:
                ensemble_metrics = {
                    "accuracy": accuracy_score(y, predictions["ensemble_pred"]),
                    "roc_auc": float("nan"),
                }

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            if "xgb" in metrics_comparison:
                col1.metric(
                    "XGBoost Accuracy",
                    f"{metrics_comparison['xgb']['accuracy']:.1%}",
                    delta=f"AUC: {metrics_comparison['xgb']['roc_auc']:.3f}",
                )

            if "logistic" in metrics_comparison:
                col2.metric(
                    "Logistic Regression Accuracy",
                    f"{metrics_comparison['logistic']['accuracy']:.1%}",
                    delta=f"AUC: {metrics_comparison['logistic']['roc_auc']:.3f}",
                )

            col3.metric(
                "Ensemble Accuracy",
                f"{ensemble_metrics['accuracy']:.1%}",
                delta=f"AUC: {ensemble_metrics['roc_auc']:.3f}",
            )

            col4.metric(
                "Model Agreement",
                f"{agreement_stats['agreement_rate']:.1%}",
                delta=f"{agreement_stats['disagreement_count']} disagreements",
            )

            # Detailed metrics table
            st.subheader("📋 Detailed Performance Metrics")

            if metrics_comparison:
                metrics_df = pd.DataFrame(metrics_comparison).T

                # Add ensemble row
                ensemble_row = pd.DataFrame(
                    {
                        "accuracy": [ensemble_metrics["accuracy"]],
                        "roc_auc": [ensemble_metrics["roc_auc"]],
                        "precision": [
                            np.nan
                        ],  # Would need to calculate separately
                        "recall": [np.nan],
                        "f1_score": [np.nan],
                    },
                    index=["ensemble"],
                )

                metrics_df = pd.concat([metrics_df, ensemble_row])

                # Remove confusion_matrix column if it exists (can't display in dataframe)
                display_metrics = metrics_df.drop(
                    columns=["confusion_matrix"], errors="ignore"
                )

                # Display with formatting
                st.dataframe(
                    display_metrics.round(4),
                    use_container_width=True,
                )

            # ======== ROC Curves Comparison ========
            st.subheader("📈 ROC Curves Comparison")

            col1, col2 = st.columns([3, 2])

            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))

                # Plot ROC curves for each model
                models_to_plot = [
                    ("XGBoost", predictions["xgb_proba"], "blue"),
                    (
                        "Logistic Regression",
                        predictions["logistic_proba"],
                        "red",
                    ),
                    ("Ensemble", predictions["ensemble_proba"], "green"),
                ]

                for model_name, y_proba, color in models_to_plot:
                    if y_proba is not None:
                        try:
                            fpr, tpr, _ = roc_curve(y, y_proba)
                            auc_score = roc_auc_score(y, y_proba)
                            ax.plot(
                                fpr,
                                tpr,
                                color=color,
                                linewidth=2,
                                label=f"{model_name} (AUC = {auc_score:.3f})",
                            )
                        except ValueError:
                            # Handle case where ROC curve can't be calculated
                            ax.plot(
                                [0, 1],
                                [0, 1],
                                color=color,
                                linewidth=2,
                                linestyle="--",
                                alpha=0.5,
                                label=f"{model_name} (AUC = nan)",
                            )

                # Plot diagonal line
                ax.plot(
                    [0, 1],
                    [0, 1],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                    label="Random Classifier",
                )
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves Comparison")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("#### 📖 ROC Curve Guide")
                st.markdown("""
                **How to interpret ROC curves:**

                🎯 **AUC Score (Area Under Curve):**
                - **0.9-1.0**: Excellent performance
                - **0.8-0.9**: Good performance
                - **0.7-0.8**: Fair performance
                - **0.6-0.7**: Poor performance
                - **0.5**: Random guessing

                📊 **Curve Position:**
                - **Top-left corner**: Perfect classifier
                - **Diagonal line**: Random classifier
                - **Below diagonal**: Worse than random

                🔍 **Model Comparison:**
                - **Higher curve**: Better performance
                - **Larger AUC**: More accurate model
                - **Steeper rise**: Better true positive rate

                💡 **What this means:**
                Models with curves closer to the top-left corner and higher AUC scores are better at distinguishing between playoff and non-playoff teams.
                """)

            # ======== Confusion Matrices ========
            st.subheader("🔢 Confusion Matrices")

            col1, col2, col3 = st.columns(3)

            # XGBoost confusion matrix
            with col1:
                st.write("**XGBoost**")
                if "xgb" in metrics_comparison:
                    cm_xgb = metrics_comparison["xgb"]["confusion_matrix"]
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        cm_xgb, annot=True, fmt="d", cmap="Blues", ax=ax
                    )
                    ax.set_title("XGBoost Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                    plt.close()

            # Logistic Regression confusion matrix
            with col2:
                st.write("**Logistic Regression**")
                if "logistic" in metrics_comparison:
                    cm_lr = metrics_comparison["logistic"]["confusion_matrix"]
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Reds", ax=ax)
                    ax.set_title("Logistic Regression Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                    plt.close()

            # Ensemble confusion matrix
            with col3:
                st.write("**Ensemble**")
                cm_ensemble = confusion_matrix(y, predictions["ensemble_pred"])
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm_ensemble, annot=True, fmt="d", cmap="Greens", ax=ax
                )
                ax.set_title("Ensemble Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close()

        else:
            st.info(
                f"No valid actual results available for {selected_year}. Showing prediction comparison only."
            )

            # Show basic prediction statistics
            st.subheader("📊 Prediction Statistics")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "XGBoost Avg Prob", f"{predictions['xgb_proba'].mean():.3f}"
            )
            col2.metric(
                "Logistic Avg Prob",
                f"{predictions['logistic_proba'].mean():.3f}",
            )
            col3.metric(
                "Ensemble Avg Prob",
                f"{predictions['ensemble_proba'].mean():.3f}",
            )
            col4.metric(
                "Model Agreement", f"{agreement_stats['agreement_rate']:.1%}"
            )

        # ======== Feature Importance Comparison ========
        st.header("🔍 Feature Importance Comparison")

        # Get feature importance comparison
        importance_df = get_feature_importance_comparison(
            xgb_model, logistic_model, feature_names
        )

        # Display top features
        st.subheader("📊 Top 15 Most Important Features")

        top_features = importance_df.head(15)

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(top_features))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            top_features["xgb_importance_norm"],
            width,
            label="XGBoost",
            color="blue",
            alpha=0.7,
        )
        bars2 = ax.bar(
            x + width / 2,
            top_features["logistic_importance_norm"],
            width,
            label="Logistic Regression",
            color="red",
            alpha=0.7,
        )

        ax.set_xlabel("Features")
        ax.set_ylabel("Normalized Importance")
        ax.set_title("Feature Importance Comparison (Top 15)")
        ax.set_xticks(x)
        ax.set_xticklabels(top_features["feature"], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature importance table
        st.subheader("📋 Feature Importance Details")

        display_importance = top_features[
            [
                "feature",
                "xgb_importance",
                "logistic_importance",
                "avg_importance",
            ]
        ].round(4)

        st.dataframe(
            display_importance.rename(
                columns={
                    "feature": "Feature",
                    "xgb_importance": "XGBoost Importance",
                    "logistic_importance": "Logistic Importance (|coef|)",
                    "avg_importance": "Average Normalized",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        # ======== Model Agreement Analysis ========
        st.header("🤝 Model Agreement Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Agreement Rate", f"{agreement_stats['agreement_rate']:.1%}"
            )
            with st.expander("📖 How Agreement Rate is Calculated"):
                st.markdown("""
                **Formula**:
                ```
                Agreement Rate = Teams with same prediction ÷ Total teams
                ```

                **Definition**: Percentage of teams where both models make the same binary prediction (both predict playoffs OR both predict no playoffs).

                **Example**: If 28 out of 30 teams have matching predictions, Agreement Rate = 28/30 = 93.3%
                """)

            st.metric(
                "Teams in Agreement", f"{agreement_stats['agreement_count']}"
            )
            with st.expander("📖 How Teams in Agreement is Calculated"):
                st.markdown("""
                **Definition**: Count of teams where both models predict the same outcome.

                **Agreement occurs when**:
                - Both models predict team WILL make playoffs (probability > 0.5)
                - Both models predict team will NOT make playoffs (probability ≤ 0.5)

                **Example**: XGBoost says "Yes" and Logistic says "Yes" = Agreement
                """)

            st.metric(
                "Teams in Disagreement",
                f"{agreement_stats['disagreement_count']}",
            )
            with st.expander("📖 How Teams in Disagreement is Calculated"):
                st.markdown("""
                **Formula**:
                ```
                Disagreement Count = Total teams - Teams in agreement
                ```

                **Disagreement occurs when**:
                - XGBoost predicts playoffs, Logistic predicts no playoffs
                - XGBoost predicts no playoffs, Logistic predicts playoffs

                **Analysis**: Higher disagreement indicates models have different decision boundaries.
                """)

        with col2:
            st.metric(
                "Average Probability Difference",
                f"{agreement_stats['avg_prob_difference']:.3f}",
            )
            with st.expander(
                "📖 How Average Probability Difference is Calculated"
            ):
                st.markdown("""
                **Formula**:
                ```
                Avg Prob Diff = mean(|XGBoost_prob - Logistic_prob|)
                ```

                **Explanation**: Average absolute difference between model probabilities across all teams.

                **Interpretation**:
                - **Low values (0.0-0.1)**: Models are very similar
                - **Medium values (0.1-0.2)**: Some differences but generally aligned
                - **High values (0.2+)**: Significant disagreement between models
                """)

            st.metric(
                "Maximum Probability Difference",
                f"{agreement_stats['max_prob_difference']:.3f}",
            )
            with st.expander(
                "📖 How Maximum Probability Difference is Calculated"
            ):
                st.markdown("""
                **Formula**:
                ```
                Max Prob Diff = max(|XGBoost_prob - Logistic_prob|)
                ```

                **Explanation**: Largest absolute difference between model probabilities for any single team.

                **Analysis**:
                - **Values near 1.0**: One model is very confident, the other is not
                - **Values near 0.0**: Models agree closely even on edge cases
                - **Identifies**: Teams with highest model uncertainty/disagreement
                """)

            if y is not None:
                st.metric(
                    "Both Models Correct Rate",
                    f"{agreement_stats['both_correct_rate']:.1%}",
                )
                with st.expander(
                    "📖 How Both Models Correct Rate is Calculated"
                ):
                    st.markdown("""
                    **Formula**:
                    ```
                    Both Correct Rate = Teams where both models match actual result ÷ Total teams
                    ```

                    **Definition**: Percentage of teams where both XGBoost AND Logistic Regression correctly predicted the actual playoff outcome.

                    **Use Case**: Identifies teams where model consensus was reliable and accurate.

                    **High Rate**: Indicates strong model agreement on correct predictions.
                    """)

        # Probability difference distribution
        st.subheader("📊 Probability Difference Distribution")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            comparison_df["prob_difference"],
            bins=20,
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_xlabel("Probability Difference (XGBoost - Logistic)")
        ax.set_ylabel("Number of Teams")
        ax.set_title("Distribution of Probability Differences Between Models")
        ax.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, label="No Difference"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

        # ======== Teams with Largest Disagreements ========
        st.subheader("⚡ Teams with Largest Model Disagreements")

        disagreements = analyze_disagreements(comparison_df)

        if len(disagreements) > 0:
            # Add league names for display
            disagreements["league_name"] = disagreements["league"].map(
                {0: "AL", 1: "NL"}
            )

            display_cols = [
                "team",
                "league_name",
                "xgb_probability",
                "logistic_probability",
                "prob_difference",
                "xgb_prediction",
                "logistic_prediction",
            ]

            col_names = {
                "team": "Team",
                "league_name": "League",
                "xgb_probability": "XGB Probability",
                "logistic_probability": "Logistic Probability",
                "prob_difference": "Probability Difference",
                "xgb_prediction": "XGB Prediction",
                "logistic_prediction": "Logistic Prediction",
            }

            if y is not None:
                display_cols.extend(
                    ["actual", "xgb_correct", "logistic_correct"]
                )
                col_names.update(
                    {
                        "actual": "Actual Result",
                        "xgb_correct": "XGB Correct",
                        "logistic_correct": "Logistic Correct",
                    }
                )

            st.dataframe(
                disagreements[display_cols].rename(columns=col_names),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("🎉 All models agree on their predictions!")

        # ======== Detailed Team Comparison ========
        if st.checkbox("👥 Show All Team Comparisons"):
            st.subheader("📊 Complete Team-by-Team Analysis")

            # Add league names for display
            display_comparison = comparison_df.copy()
            display_comparison["league_name"] = display_comparison[
                "league"
            ].map({0: "AL", 1: "NL"})

            # Sort by absolute probability difference
            display_comparison = display_comparison.sort_values(
                "prob_difference", key=abs, ascending=False
            )

            display_cols = [
                "team",
                "league_name",
                "xgb_probability",
                "logistic_probability",
                "prob_difference",
                "models_agree",
            ]

            col_names = {
                "team": "Team",
                "league_name": "League",
                "xgb_probability": "XGB Probability",
                "logistic_probability": "Logistic Probability",
                "prob_difference": "Probability Difference",
                "models_agree": "Models Agree",
            }

            if has_valid_data:
                display_cols.extend(
                    [
                        "actual",
                        "xgb_correct",
                        "logistic_correct",
                        "both_correct",
                    ]
                )
                col_names.update(
                    {
                        "actual": "Actual Result",
                        "xgb_correct": "XGB Correct",
                        "logistic_correct": "Logistic Correct",
                        "both_correct": "Both Correct",
                    }
                )

            st.dataframe(
                display_comparison[display_cols].rename(columns=col_names),
                hide_index=True,
                use_container_width=True,
            )

    else:
        st.error(f"No data available for {selected_year}")

else:
    st.warning(
        "No years available for analysis. Please ensure data files are present."
    )

# Add recommendation for better years
if available_years:
    st.sidebar.subheader("💡 Analysis Tips")
    st.sidebar.info(
        "**🎯 Best Years for Learning:**\n"
        "- **2024**: Most recent validation data\n"
        "- **2023**: Complete season results\n"
        "- **2022**: Historical comparison\n\n"
        "**⚠️ Avoid 2025**: Future predictions only"
    )

    # Learning journey CTA
    st.sidebar.subheader("🎓 Learning Path")
    st.sidebar.markdown("""
    **📊 Beginner**: Start with Performance Overview
    **📈 Intermediate**: Explore ROC Curves
    **🔬 Advanced**: Dive into Feature Importance
    **🤝 Expert**: Analyze Model Disagreements
    """)

# Additional info and journey continuation
st.markdown("---")
st.subheader("🎓 Continue Your Data Science Journey")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 **What You've Learned**

    ✅ **Model Accuracy**: How often predictions are correct

    ✅ **ROC Curves**: Visual model performance comparison

    ✅ **Feature Importance**: What stats matter most

    ✅ **Model Agreement**: When algorithms agree/disagree

    ### 🤔 **Think About This**
    - Why might XGBoost and Logistic Regression disagree on certain teams?

    - Which features surprise you as being important?

    - How could we improve model accuracy further?
    """)

with col2:
    st.markdown("""
    ### 🚀 **Next Steps**

    **🔮 Try Your Own Data**: Upload team stats and see predictions

    **📈 Explore Visualizations**: See interactive charts and trends

    **🏠 Return to Dashboard**: View current season predictions

    ### 💡 **Pro Tips**

    - Compare multiple years to see consistency

    - Look for bubble teams where models disagree

    - Use ensemble predictions for most reliable results
    """)

st.info(
    "🔬 **Data Science Insight**: Model disagreement isn't bad - it often highlights the most interesting "
    "and uncertain cases where human expertise and statistical analysis both provide value!"
)
