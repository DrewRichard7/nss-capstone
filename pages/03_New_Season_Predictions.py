import pickle

import numpy as np
import pandas as pd
import streamlit as st

from models.model_utils import (
    create_prediction_comparison_df,
    get_ensemble_prediction,
    get_model_agreement_stats,
    get_model_predictions,
    load_logistic_model,
    load_xgboost_model,
    preprocess_for_models,
)

# ======== Page config ========
st.set_page_config(
    page_title="New Season Predictions",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Interactive Prediction Laboratory")
st.markdown(
    "### Upload your own data and watch machine learning models make predictions in real-time"
)

# Journey introduction
st.markdown("""
🧪 **Welcome to the Prediction Lab!** This is where you become the data scientist. Upload team statistics
and watch as our trained models analyze the data and make playoff predictions before your eyes.

💡 **What you'll experience:**
- Upload real or hypothetical team data
- See how different ML algorithms interpret the same statistics
- Compare XGBoost vs Logistic Regression vs Ensemble predictions
- Download results and understand model confidence levels
""")

# Check if using cross-validated models
try:
    with open("assets/logistic_playoffs_cv.pkl", "rb") as f:
        logistic_cv_data = pickle.load(f)
    with open("assets/xgb_playoffs_cv.pkl", "rb") as f:
        xgb_cv_data = pickle.load(f)

    st.success(
        """
    ✅ **Using Cross-Validated Models**: Enhanced models optimized through systematic hyperparameter tuning.
    Logistic: {:.4f} CV ROC AUC | XGBoost: {:.4f} CV ROC AUC
    """.format(
            logistic_cv_data.get("best_cv_score", 0),
            xgb_cv_data.get("best_cv_score", 0),
        )
    )
except FileNotFoundError:
    st.info("""
    ℹ️ **Using Standard Models**: For enhanced performance, run `python models/run_cv_training.py`
    to generate cross-validated models with optimized hyperparameters.
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


# ======== MLB Division and Wild Card Logic ========
MLB_DIVISIONS = {
    # Current team names
    "Baltimore Orioles": "American League East",
    "Boston Red Sox": "American League East",
    "New York Yankees": "American League East",
    "Tampa Bay Rays": "American League East",
    "Toronto Blue Jays": "American League East",
    "Chicago White Sox": "American League Central",
    "Cleveland Guardians": "American League Central",
    "Detroit Tigers": "American League Central",
    "Kansas City Royals": "American League Central",
    "Minnesota Twins": "American League Central",
    "Houston Astros": "American League West",
    "Los Angeles Angels": "American League West",
    "Oakland Athletics": "American League West",
    "Seattle Mariners": "American League West",
    "Texas Rangers": "American League West",
    "Atlanta Braves": "National League East",
    "Miami Marlins": "National League East",
    "New York Mets": "National League East",
    "Philadelphia Phillies": "National League East",
    "Washington Nationals": "National League East",
    "Chicago Cubs": "National League Central",
    "Cincinnati Reds": "National League Central",
    "Milwaukee Brewers": "National League Central",
    "Pittsburgh Pirates": "National League Central",
    "St. Louis Cardinals": "National League Central",
    "Arizona Diamondbacks": "National League West",
    "Colorado Rockies": "National League West",
    "Los Angeles Dodgers": "National League West",
    "San Diego Padres": "National League West",
    "San Francisco Giants": "National League West",
    # Historical team names for backward compatibility
    "Cleveland Indians": "American League Central",
    "Tampa Bay Devil Rays": "American League East",
    "Florida Marlins": "National League East",
    "Montreal Expos": "National League East",
    "California Angels": "American League West",
    "Anaheim Angels": "American League West",
    "Los Angeles Angels of Anaheim": "American League West",
    "St Louis Cardinals": "National League Central",  # No period version
    "Arizona DiamondbacksDiamondbacks": "National League West",  # Duplicate issue
    "Montreal ExposExpos": "National League East",  # Duplicate issue
}


def enforce_mlb_playoff_rules(probabilities, leagues, team_names):
    """
    Enforce MLB playoff rules: 3 division winners + 3 wild cards per league.
    """
    # Clean team names to handle Unicode characters and duplicates
    cleaned_team_names = []
    for team in team_names:
        # Remove Unicode characters and normalize
        import re

        cleaned = re.sub(r"[^\w\s]", "", str(team)).strip()

        # Remove duplicate words first (fix DiamondbacksDiamondbacks issue)
        words = cleaned.split()
        cleaned = " ".join(dict.fromkeys(words))

        # Check for specific duplicate patterns and fix them
        if (
            "DiamondbacksDiamondbacks" in str(team)
            or cleaned.count("Diamondbacks") > 1
        ):
            cleaned = "Arizona Diamondbacks"
        elif "ExposExpos" in str(team) or cleaned.count("Expos") > 1:
            cleaned = "Montreal Expos"
        elif "IndiansIndians" in str(team) or cleaned.count("Indians") > 1:
            cleaned = "Cleveland Guardians"

        # Handle specific team name cases and duplications
        elif "Athletics" in cleaned and "Athletics" not in [
            "Oakland Athletics"
        ]:
            cleaned = "Oakland Athletics"
        elif "Guardians" in cleaned and cleaned != "Cleveland Guardians":
            cleaned = "Cleveland Guardians"
        elif "Indians" in cleaned or "Cleveland" in cleaned:
            cleaned = "Cleveland Guardians"  # Updated to current team name
        elif "St Louis Cardinals" in cleaned:
            cleaned = "St. Louis Cardinals"
        elif "Cardinals" in cleaned and "St" in cleaned:
            cleaned = "St. Louis Cardinals"
        elif "Angels" in cleaned and (
            "Los Angeles" in cleaned
            or "Anaheim" in cleaned
            or "California" in cleaned
        ):
            cleaned = "Los Angeles Angels"
        elif "Devil Rays" in cleaned or (
            "Tampa Bay" in cleaned and "Rays" in cleaned
        ):
            cleaned = "Tampa Bay Rays"
        elif "Marlins" in cleaned and (
            "Florida" in cleaned or "Miami" in cleaned
        ):
            cleaned = "Miami Marlins"

        cleaned_team_names.append(cleaned)

    df = pd.DataFrame(
        {
            "team": cleaned_team_names,
            "probability": probabilities,
            "league": leagues,
            "league_name": ["AL" if l == 0 else "NL" for l in leagues],
        }
    )

    # Add division information
    df["division"] = df["team"].map(MLB_DIVISIONS)

    # Handle teams not in our division mapping
    missing_divisions = df[df["division"].isna()]
    if not missing_divisions.empty:
        st.warning(
            f"⚠️ Some teams not found in division mapping: {missing_divisions['team'].to_list()}"
        )
        for idx, row in missing_divisions.iterrows():
            if row["league_name"] == "AL":
                df.loc[idx, "division"] = "American League East"
            else:
                df.loc[idx, "division"] = "National League East"

    # Extract division type
    df["division_type"] = df["division"].str.extract(r"(East|Central|West)$")

    playoff_teams = []

    # Process each league separately
    for league in ["AL", "NL"]:
        league_df = df[df["league_name"] == league].copy()

        if len(league_df) == 0:
            continue

        # Determine division winners
        division_winners = []
        division_winner_indices = []
        for div_type in ["East", "Central", "West"]:
            div_teams = league_df[league_df["division_type"] == div_type]
            if len(div_teams) > 0:
                winner_idx = div_teams["probability"].idxmax()
                winner = league_df.loc[winner_idx]
                division_winners.append(winner)
                division_winner_indices.append(winner_idx)

        # Determine wild card teams
        non_winners = league_df[~league_df.index.isin(division_winner_indices)]
        wild_cards = non_winners.nlargest(3, "probability")

        # Combine division winners and wild cards
        league_playoff_teams = division_winners + wild_cards.to_dict("records")
        playoff_teams.extend([team["team"] for team in league_playoff_teams])

    # Create final predictions
    df["makes_playoffs_constrained"] = df["team"].isin(playoff_teams)

    # Add ranking information
    df["league_rank"] = df.groupby("league_name")["probability"].rank(
        method="dense", ascending=False
    )
    df["division_rank"] = df.groupby("division")["probability"].rank(
        method="dense", ascending=False
    )

    # Add playoff type classification
    df["playoff_type"] = "Miss"
    for league in ["AL", "NL"]:
        league_df = df[df["league_name"] == league].copy()

        # Division winners
        for div_type in ["East", "Central", "West"]:
            div_teams = league_df[league_df["division_type"] == div_type]
            if len(div_teams) > 0:
                winner_idx = div_teams["probability"].idxmax()
                if df.loc[winner_idx, "makes_playoffs_constrained"]:
                    df.loc[winner_idx, "playoff_type"] = f"{div_type} Winner"

        # Wild cards
        division_winner_indices = []
        for div_type in ["East", "Central", "West"]:
            div_teams = league_df[league_df["division_type"] == div_type]
            if len(div_teams) > 0:
                winner_idx = div_teams["probability"].idxmax()
                division_winner_indices.append(winner_idx)

        non_winners = league_df[~league_df.index.isin(division_winner_indices)]
        wild_cards = non_winners.nlargest(3, "probability")
        for _, wild_card in wild_cards.iterrows():
            if wild_card["makes_playoffs_constrained"]:
                df.loc[wild_card.name, "playoff_type"] = "Wild Card"

    # Sort for display
    league_rankings = df.sort_values(
        ["league_name", "division_type", "probability"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return df["makes_playoffs_constrained"].values, league_rankings


# ======== Load models ========
xgb_model, logistic_model, scaler, feature_names, models_loaded = load_models()

if not models_loaded:
    st.error(
        "Failed to load models. Please ensure both models are trained and available."
    )
    st.stop()

# ======== Sidebar - Model Selection ========
st.sidebar.subheader("🤖 Model Configuration")
prediction_model = st.sidebar.selectbox(
    "Primary model for playoff predictions:",
    ["XGBoost", "Logistic Regression", "Ensemble (Average)"],
    index=0,
)

show_comparison = st.sidebar.checkbox("Show model comparison", value=True)

# ======== Sidebar Configuration ========
st.sidebar.subheader("🎛️ Prediction Settings")

# Enhanced model selection with descriptions
st.sidebar.subheader("🤖 Model Selection")
prediction_model = st.sidebar.selectbox(
    "Choose your prediction algorithm:",
    ["XGBoost", "Logistic Regression", "Ensemble (Average)"],
    index=2,  # Default to ensemble
    help="Each model has different strengths - Ensemble combines the best of both!",
)

# Model info in sidebar
model_descriptions = {
    "XGBoost": "🌳 **Gradient Boosting**: Uses decision trees, great for complex patterns",
    "Logistic Regression": "📈 **Linear Model**: Simple, interpretable, fast predictions",
    "Ensemble (Average)": "🤝 **Best of Both**: Combines XGBoost + Logistic for robust results",
}
st.sidebar.info(model_descriptions[prediction_model])

show_comparison = st.sidebar.checkbox(
    "Show detailed model comparison", value=True
)
show_confidence = st.sidebar.checkbox(
    "Show prediction confidence levels", value=True
)

# ======== Main Content ========
st.header("📊 Upload Your Data")

# Learning callout
st.markdown("""
📚 **Learning Moment**: Machine learning models need data in a specific format. The better and more complete
your data, the more confident and accurate the predictions will be!
""")

# Sample data format
with st.expander("📋 Expected Data Format"):
    st.markdown("""
    **Required columns:**
    - `TEAM`: Team name (e.g., "New York Yankees")
    - `LEAGUE`: League designation ("AL" or "NL")

    **Statistical columns (examples):**
    - Hitting stats: `H_R`, `H_HR`, `H_RBI`, `H_AVG`, etc.
    - Pitching stats: `P_W`, `P_L`, `P_ERA`, `P_SO`, etc.

    **Optional columns:**
    - `MADE_PLAYOFFS`: Actual playoff results (True/False) for evaluation
    - `WON_WORLD_SERIES`: World Series results (True/False)

    The model expects the same statistical categories as the training data.
    """)

# File upload
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file with team statistics in the expected format",
)

if uploaded_file is not None:
    try:
        # Load the uploaded data
        uploaded_data = pd.read_csv(uploaded_file)

        st.subheader("📈 Uploaded Data Preview")
        st.dataframe(uploaded_data.head(), use_container_width=True)

        # Validate required columns
        required_cols = ["TEAM", "LEAGUE"]
        missing_cols = [
            col for col in required_cols if col not in uploaded_data.columns
        ]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        # Preprocess the data
        X, y, team_info = preprocess_for_models(uploaded_data)

        # Get predictions from both models
        predictions = get_model_predictions(
            X, xgb_model, logistic_model, scaler
        )

        # Create ensemble predictions
        ensemble_preds = get_ensemble_prediction(predictions, method="average")
        predictions.update(ensemble_preds)

        # Select primary model probabilities
        if prediction_model == "XGBoost":
            primary_proba = predictions["xgb_proba"]
        elif prediction_model == "Logistic Regression":
            primary_proba = predictions["logistic_proba"]
        else:  # Ensemble
            primary_proba = predictions["ensemble_proba"]

        # Enforce MLB playoff rules
        playoff_predictions, rankings = enforce_mlb_playoff_rules(
            primary_proba, X["LEAGUE"].values, team_info["TEAM"].values
        )

        # ======== Results Display ========
        st.header(f"🏆 Playoff Predictions ({prediction_model})")

        # Success and learning callout
        st.success(
            "✅ **Prediction Complete!** Your data has been processed by our trained models."
        )
        st.markdown("""
        🎯 **What just happened?** Our algorithms analyzed your team statistics and calculated the probability
        each team makes the playoffs, then applied real MLB rules (3 division winners + 3 wild cards per league).
        """)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        al_playoff_teams = rankings[
            (rankings["league_name"] == "AL")
            & (rankings["makes_playoffs_constrained"])
        ]
        nl_playoff_teams = rankings[
            (rankings["league_name"] == "NL")
            & (rankings["makes_playoffs_constrained"])
        ]

        col1.metric("AL Playoff Teams", len(al_playoff_teams))
        col2.metric("NL Playoff Teams", len(nl_playoff_teams))
        col3.metric(
            "Total Playoff Teams", len(al_playoff_teams) + len(nl_playoff_teams)
        )
        col4.metric(
            "Non-Playoff Teams",
            len(rankings) - len(al_playoff_teams) - len(nl_playoff_teams),
        )

        # Display playoff teams by league and division
        st.subheader("📋 Playoff Predictions by Division")

        # American League
        st.write("**🇺🇸 American League**")
        for div_type in ["East", "Central", "West"]:
            div_teams = rankings[
                (rankings["league_name"] == "AL")
                & (rankings["division_type"] == div_type)
            ].copy()

            if not div_teams.empty:
                st.write(f"**{div_type} Division:**")
                div_teams["status"] = div_teams[
                    "makes_playoffs_constrained"
                ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

                display_cols = [
                    "team",
                    "probability",
                    "division_rank",
                    "playoff_type",
                    "status",
                ]
                column_names = {
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "division_rank": "Div Rank",
                    "playoff_type": "Playoff Type",
                    "status": "Status",
                }

                st.dataframe(
                    div_teams[display_cols].rename(columns=column_names),
                    hide_index=True,
                    use_container_width=True,
                )
                st.write("---")

        # National League
        st.write("**🇺🇸 National League**")
        for div_type in ["East", "Central", "West"]:
            div_teams = rankings[
                (rankings["league_name"] == "NL")
                & (rankings["division_type"] == div_type)
            ].copy()

            if not div_teams.empty:
                st.write(f"**{div_type} Division:**")
                div_teams["status"] = div_teams[
                    "makes_playoffs_constrained"
                ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

                display_cols = [
                    "team",
                    "probability",
                    "division_rank",
                    "playoff_type",
                    "status",
                ]
                column_names = {
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "division_rank": "Div Rank",
                    "playoff_type": "Playoff Type",
                    "status": "Status",
                }

                st.dataframe(
                    div_teams[display_cols].rename(columns=column_names),
                    hide_index=True,
                    use_container_width=True,
                )
                st.write("---")

        # ======== Model Comparison ========
        if show_comparison:
            st.header("🔍 Model Comparison Analysis")

            # Learning moment
            st.markdown("""
            🧠 **Data Science Insight**: Different algorithms can interpret the same data differently.
            Comparing models helps us understand prediction confidence and find potential edge cases!
            """)

            # Create comparison dataframe
            comparison_df = create_prediction_comparison_df(
                team_info, predictions, y
            )
            agreement_stats = get_model_agreement_stats(comparison_df)

            # Agreement statistics
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Model Agreement", f"{agreement_stats['agreement_rate']:.1%}"
            )
            col2.metric(
                "Disagreements", f"{agreement_stats['disagreement_count']}"
            )
            col3.metric(
                "Avg Prob Difference",
                f"{agreement_stats['avg_prob_difference']:.3f}",
            )

            # Detailed comparison table
            st.subheader("📊 Team-by-Team Model Comparison")

            # Interactive learning
            st.markdown("""
            💡 **Look for**: Teams where models disagree significantly - these are the most uncertain predictions!
            """)

            if show_confidence:
                st.markdown("""
                **🎯 Confidence Guide:**
                - **High Agreement + High Probabilities**: Very confident playoff prediction
                - **High Agreement + Low Probabilities**: Very confident non-playoff prediction
                - **Low Agreement**: Uncertain - could go either way!
                """)

            display_comparison = comparison_df.copy()
            display_comparison["league_name"] = display_comparison[
                "league"
            ].map({0: "AL", 1: "NL"})

            display_cols = [
                "team",
                "league_name",
                "xgb_probability",
                "logistic_probability",
                "ensemble_proba",
                "prob_difference",
                "models_agree",
            ]

            col_names = {
                "team": "Team",
                "league_name": "League",
                "xgb_probability": "XGB Prob",
                "logistic_probability": "Logistic Prob",
                "ensemble_proba": "Ensemble Prob",
                "prob_difference": "XGB-Logistic Diff",
                "models_agree": "Models Agree",
            }

            # Add actual results if available
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
                display_comparison[display_cols]
                .rename(columns=col_names)
                .sort_values("XGB-Logistic Diff", key=abs, ascending=False),
                hide_index=True,
                use_container_width=True,
            )

            # Highlight disagreements
            disagreements = display_comparison[
                ~display_comparison["models_agree"]
            ]
            if len(disagreements) > 0:
                st.subheader("⚡ Teams with Model Disagreements")
                st.dataframe(
                    disagreements[display_cols].rename(columns=col_names),
                    hide_index=True,
                    use_container_width=True,
                )

        # ======== Download Results ========
        st.header("📥 Download Results")

        # Prepare download data
        download_data = rankings.copy()
        download_data["model_used"] = prediction_model

        # Add individual model probabilities
        download_data["xgb_probability"] = predictions["xgb_proba"]
        download_data["logistic_probability"] = predictions["logistic_proba"]
        download_data["ensemble_probability"] = predictions["ensemble_proba"]

        # Convert to CSV
        csv = download_data.to_csv(index=False)

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"playoff_predictions_{prediction_model.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.write(
            "Please check that your file format matches the expected structure."
        )

else:
    # ======== Sample Data Generation ========
    st.header("🎲 Try with Sample Data")

    st.markdown("""
    🎮 **No data to upload?** No problem! Generate realistic sample data to see how the models work.
    This is perfect for learning and experimentation!
    """)

    if st.button("Generate Sample Predictions"):
        st.info("Generating sample data for demonstration...")

        # Create sample data
        np.random.seed(42)

        # Sample teams
        sample_teams = [
            "New York Yankees",
            "Boston Red Sox",
            "Tampa Bay Rays",
            "Toronto Blue Jays",
            "Baltimore Orioles",
            "Houston Astros",
            "Seattle Mariners",
            "Los Angeles Angels",
            "Oakland Athletics",
            "Texas Rangers",
            "Cleveland Guardians",
            "Minnesota Twins",
            "Chicago White Sox",
            "Detroit Tigers",
            "Kansas City Royals",
            "Atlanta Braves",
            "New York Mets",
            "Philadelphia Phillies",
            "Miami Marlins",
            "Washington Nationals",
            "Los Angeles Dodgers",
            "San Diego Padres",
            "San Francisco Giants",
            "Colorado Rockies",
            "Arizona Diamondbacks",
            "Milwaukee Brewers",
            "St. Louis Cardinals",
            "Chicago Cubs",
            "Cincinnati Reds",
            "Pittsburgh Pirates",
        ]

        # Sample leagues
        sample_leagues = ["AL"] * 15 + ["NL"] * 15

        # Generate realistic sample statistics with all expected features
        sample_data = pd.DataFrame(
            {
                "TEAM": sample_teams,
                "LEAGUE": sample_leagues,
                "H_G": np.random.randint(80, 100, 30),
                "H_AB": np.random.randint(2500, 3500, 30),
                "H_R": np.random.randint(300, 500, 30),
                "H_H": np.random.randint(700, 900, 30),
                "H_2B": np.random.randint(150, 250, 30),
                "H_3B": np.random.randint(20, 60, 30),
                "H_HR": np.random.randint(100, 200, 30),
                "H_RBI": np.random.randint(400, 600, 30),
                "H_BB": np.random.randint(300, 500, 30),
                "H_SO": np.random.randint(800, 1200, 30),
                "H_SB": np.random.randint(50, 150, 30),
                "H_CS": np.random.randint(20, 60, 30),
                "H_AVG": np.random.uniform(0.240, 0.280, 30),
                "H_OBP": np.random.uniform(0.310, 0.360, 30),
                "H_SLG": np.random.uniform(0.400, 0.500, 30),
                "H_OPS": np.random.uniform(0.710, 0.860, 30),
                "P_W": np.random.randint(40, 80, 30),
                "P_L": np.random.randint(40, 80, 30),
                "P_ERA": np.random.uniform(3.50, 5.00, 30),
                "P_G": np.random.randint(80, 100, 30),
                "P_GS": np.random.randint(80, 100, 30),
                "P_CG": np.random.randint(2, 8, 30),
                "P_SHO": np.random.randint(5, 15, 30),
                "P_SV": np.random.randint(30, 50, 30),
                "P_SVO": np.random.randint(40, 60, 30),
                "P_IP": np.random.uniform(700.0, 900.0, 30),
                "P_H": np.random.randint(700, 900, 30),
                "P_R": np.random.randint(400, 600, 30),
                "P_ER": np.random.randint(350, 550, 30),
                "P_HR": np.random.randint(100, 180, 30),
                "P_HB": np.random.randint(40, 80, 30),
                "P_BB": np.random.randint(300, 500, 30),
                "P_SO": np.random.randint(800, 1200, 30),
                "P_WHIP": np.random.uniform(1.20, 1.50, 30),
                "P_AVG": np.random.uniform(0.240, 0.280, 30),
            }
        )

        st.write("**Sample Data Generated:**")
        st.dataframe(sample_data.head(10), use_container_width=True)

        # Process sample data
        X_sample, _, team_info_sample = preprocess_for_models(sample_data)
        predictions_sample = get_model_predictions(
            X_sample, xgb_model, logistic_model, scaler
        )
        ensemble_preds_sample = get_ensemble_prediction(
            predictions_sample, method="average"
        )
        predictions_sample.update(ensemble_preds_sample)

        # Use ensemble for sample
        playoff_predictions_sample, rankings_sample = enforce_mlb_playoff_rules(
            predictions_sample["ensemble_proba"],
            X_sample["LEAGUE"].values,
            team_info_sample["TEAM"].values,
        )

        st.subheader("🏆 Sample Playoff Predictions (Ensemble Model)")

        # Learning moment for sample data
        st.markdown("""
        🎲 **What you're seeing**: These predictions are based on randomly generated (but realistic) team statistics.
        Notice how the models still follow MLB playoff rules and create reasonable-looking standings!
        """)

        # Show playoff teams
        playoff_teams_sample = rankings_sample[
            rankings_sample["makes_playoffs_constrained"]
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.write("**American League Playoff Teams:**")
            al_playoff = playoff_teams_sample[
                playoff_teams_sample["league_name"] == "AL"
            ]
            for _, team in al_playoff.iterrows():
                st.write(
                    f"- {team['team']} ({team['playoff_type']}) - {team['probability']:.1%}"
                )

        with col2:
            st.write("**National League Playoff Teams:**")
            nl_playoff = playoff_teams_sample[
                playoff_teams_sample["league_name"] == "NL"
            ]
            for _, team in nl_playoff.iterrows():
                st.write(
                    f"- {team['team']} ({team['playoff_type']}) - {team['probability']:.1%}"
                )

# ======== Learning Journey Continuation ========
st.markdown("---")
st.subheader("🎓 What You've Accomplished")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ **Skills Unlocked**
    - **Data Upload**: Prepared data for ML models
    - **Model Selection**: Chose between different algorithms
    - **Result Interpretation**: Understood probability outputs
    - **Model Comparison**: Analyzed algorithm differences

    ### 🤔 **Reflection Questions**
    - Which model gave the most realistic predictions?
    - What surprised you about the results?
    - How might you improve the data quality?
    """)

with col2:
    st.markdown("""
    ### 🚀 **Next Steps in Your Journey**
    **📊 Model Analysis**: Deep dive into how these predictions were made
    **📈 Visualizations**: See interactive charts and statistical trends
    **🏠 Dashboard**: View live 2025 season predictions

    ### 💡 **Pro Data Scientist Tips**
    - Always validate your data before uploading
    - Use ensemble methods for most reliable predictions
    - Look for patterns in model disagreements
    """)

st.info("""
🎯 **Quick Start Guide:**
1. **Beginners**: Try the sample data first to see how models work
2. **Upload your data**: Use the expected format for best results
3. **Compare models**: See how XGBoost vs Logistic Regression perform
4. **Analyze results**: Look for high-confidence vs uncertain predictions
5. **Download & share**: Save your predictions for further analysis

💡 **Remember**: Machine learning is about finding patterns in data - the better your data quality, the better your predictions!
""")
