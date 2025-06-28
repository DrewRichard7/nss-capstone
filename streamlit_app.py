import glob
import os
import pickle

import pandas as pd
import streamlit as st

from models.model_utils import (
    compare_model_metrics,
    create_prediction_comparison_df,
    get_all_model_hyperparameters,
    get_ensemble_prediction,
    get_model_agreement_stats,
    get_model_predictions,
    load_logistic_model,
    load_xgboost_model,
)

# ======== Page config ========
st.set_page_config(
    page_title="MLB Playoff Predictor",
    page_icon="⚾",
    layout="wide",
)

st.title("🏆 MLB 2025 Season Dashboard")
st.markdown("### Current Season Standings & Playoff Predictions")

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


# ======== CACHED DATA: load & preprocess validation data ========
@st.cache_data
def load_validation_data(path="data/mlb_team_stats_2024_pre_all_star.csv"):
    df = pd.read_csv(path)
    X, y, team_info = preprocess(df)
    return X, y, team_info


# ======== Get available years from data directory ========
@st.cache_data
def get_available_years():
    """Get list of years available in the data directory"""
    data_files = glob.glob("data/mlb_team_stats_*_pre_all_star.csv")
    years = []
    for file in data_files:
        # Extract year from filename like "mlb_team_stats_2023_pre_all_star.csv"
        basename = os.path.basename(file)
        parts = basename.split("_")
        if len(parts) >= 4:
            try:
                year = int(parts[3])
                years.append(year)
            except ValueError:
                continue
    return sorted(years, reverse=True)


# ======== Load raw data for a specific year ========
@st.cache_data
def load_raw_data(year):
    """Load raw data for a specific year"""
    filepath = f"data/mlb_team_stats_{year}_pre_all_star.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


# ======== Preprocessing function ========
def preprocess(df):
    df = df.copy()
    # store team info before dropping
    team_info = df[["TEAM", "LEAGUE"]].copy() if "TEAM" in df.columns else None

    # drop unused columns - handle missing columns gracefully
    cols_to_drop = ["TEAM"]
    if "WON_WORLD_SERIES" in df.columns:
        cols_to_drop.append("WON_WORLD_SERIES")
    df = df.drop(columns=cols_to_drop)

    # encode league as binary
    df["LEAGUE"] = df["LEAGUE"].map({"AL": 0, "NL": 1})

    # target → int
    if "MADE_PLAYOFFS" in df.columns:
        df["MADE_PLAYOFFS"] = df["MADE_PLAYOFFS"].astype(int)

    # convert any object‐typed columns to float
    for c in df.select_dtypes("object").columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )

    # fill missing
    df = df.fillna(0)

    # split features/target
    if "MADE_PLAYOFFS" in df.columns:
        X = df.drop(columns=["MADE_PLAYOFFS"])
        y = df["MADE_PLAYOFFS"]
        return X, y, team_info
    else:
        return df, None, team_info


# ======== MLB Division and Wild Card Logic ========
MLB_DIVISIONS = {
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

    Args:
        probabilities: Array of playoff probabilities
        leagues: Array of league indicators (0=AL, 1=NL)
        team_names: Array of team names

    Returns:
        constrained_predictions: Binary array of playoff predictions
        league_rankings: DataFrame with detailed rankings including division info
    """
    # Clean team names to handle Unicode characters and duplicates
    import re

    cleaned_team_names = []
    for team in team_names:
        # Remove Unicode characters and normalize
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
        elif "Indians" in cleaned or (
            "Cleveland" in cleaned and "Guardians" not in cleaned
        ):
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
        elif "Rays" in cleaned and cleaned != "Tampa Bay Rays":
            cleaned = "Tampa Bay Rays"
        elif "Marlins" in cleaned and cleaned != "Miami Marlins":
            cleaned = "Miami Marlins"
        elif "Nationals" in cleaned and cleaned != "Washington Nationals":
            cleaned = "Washington Nationals"
        elif "Rangers" in cleaned and cleaned != "Texas Rangers":
            cleaned = "Texas Rangers"
        elif "Angels" in cleaned and cleaned != "Los Angeles Angels":
            cleaned = "Los Angeles Angels"
        elif "Dodgers" in cleaned and cleaned != "Los Angeles Dodgers":
            cleaned = "Los Angeles Dodgers"
        elif "Padres" in cleaned and cleaned != "San Diego Padres":
            cleaned = "San Diego Padres"
        elif "Giants" in cleaned and cleaned != "San Francisco Giants":
            cleaned = "San Francisco Giants"
        elif "Rockies" in cleaned and cleaned != "Colorado Rockies":
            cleaned = "Colorado Rockies"
        elif "Diamondbacks" in cleaned and cleaned != "Arizona Diamondbacks":
            cleaned = "Arizona Diamondbacks"
        elif "Cardinals" in cleaned and cleaned != "St. Louis Cardinals":
            cleaned = "St. Louis Cardinals"
        elif "Pirates" in cleaned and cleaned != "Pittsburgh Pirates":
            cleaned = "Pittsburgh Pirates"
        elif "Brewers" in cleaned and cleaned != "Milwaukee Brewers":
            cleaned = "Milwaukee Brewers"
        elif "Reds" in cleaned and cleaned != "Cincinnati Reds":
            cleaned = "Cincinnati Reds"
        elif "Cubs" in cleaned and cleaned != "Chicago Cubs":
            cleaned = "Chicago Cubs"
        elif "Mets" in cleaned and cleaned != "New York Mets":
            cleaned = "New York Mets"
        elif "Phillies" in cleaned and cleaned != "Philadelphia Phillies":
            cleaned = "Philadelphia Phillies"
        elif "Braves" in cleaned and cleaned != "Atlanta Braves":
            cleaned = "Atlanta Braves"
        elif "Twins" in cleaned and cleaned != "Minnesota Twins":
            cleaned = "Minnesota Twins"
        elif "Royals" in cleaned and cleaned != "Kansas City Royals":
            cleaned = "Kansas City Royals"
        elif "Tigers" in cleaned and cleaned != "Detroit Tigers":
            cleaned = "Detroit Tigers"
        elif "White Sox" in cleaned and cleaned != "Chicago White Sox":
            cleaned = "Chicago White Sox"
        elif "Yankees" in cleaned and cleaned != "New York Yankees":
            cleaned = "New York Yankees"
        elif "Red Sox" in cleaned and cleaned != "Boston Red Sox":
            cleaned = "Boston Red Sox"
        elif "Orioles" in cleaned and cleaned != "Baltimore Orioles":
            cleaned = "Baltimore Orioles"
        elif "Blue Jays" in cleaned and cleaned != "Toronto Blue Jays":
            cleaned = "Toronto Blue Jays"
        elif "Mariners" in cleaned and cleaned != "Seattle Mariners":
            cleaned = "Seattle Mariners"
        elif "Astros" in cleaned and cleaned != "Houston Astros":
            cleaned = "Houston Astros"

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

    # Handle teams not in our division mapping (fallback to league-based ranking)
    missing_divisions = df[df["division"].isna()]
    if not missing_divisions.empty:
        st.warning(
            f"⚠️ Some teams not found in division mapping: {missing_divisions['team'].to_list()}"
        )
        # For missing teams, assign to a default division based on league
        for idx, row in missing_divisions.iterrows():
            if row["league_name"] == "AL":
                df.loc[idx, "division"] = (
                    "American League East"  # Default fallback
                )
            else:
                df.loc[idx, "division"] = (
                    "National League East"  # Default fallback
                )

    # Extract division type (East, Central, West)
    df["division_type"] = df["division"].str.extract(r"(East|Central|West)$")

    playoff_teams = []

    # Process each league separately
    for league in ["AL", "NL"]:
        league_df = df[df["league_name"] == league].copy()

        if len(league_df) == 0:
            continue

        # 1. Determine division winners (top team in each division)
        division_winners = []
        division_winner_indices = []
        for div_type in ["East", "Central", "West"]:
            div_teams = league_df[league_df["division_type"] == div_type]
            if len(div_teams) > 0:
                # Get the team with highest probability in this division
                winner_idx = div_teams["probability"].idxmax()
                winner = league_df.loc[winner_idx]
                division_winners.append(winner)
                division_winner_indices.append(winner_idx)

        # 2. Determine wild card teams (top 3 non-division winners)
        non_winners = league_df[~league_df.index.isin(division_winner_indices)]

        # Sort non-winners by probability and take top 3
        wild_cards = non_winners.nlargest(3, "probability")

        # Combine division winners and wild cards
        league_playoff_teams = division_winners + wild_cards.to_dict("records")
        playoff_teams.extend([team["team"] for team in league_playoff_teams])

    # Create final predictions
    df["makes_playoffs_constrained"] = df["team"].isin(playoff_teams)

    # Add ranking information for display
    df["league_rank"] = df.groupby("league_name")["probability"].rank(
        method="dense", ascending=False
    )

    # Add division ranking
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

        # Wild cards - get non-division winners again
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


# ======== Playoff constraint enforcement ========
def enforce_playoff_constraints(probabilities, leagues, team_names):
    """
    Enforce MLB playoff rules: 3 division winners + 3 wild cards per league.

    Args:
        probabilities: Array of playoff probabilities
        leagues: Array of league indicators (0=AL, 1=NL)
        team_names: Array of team names

    Returns:
        constrained_predictions: Binary array of playoff predictions
        league_rankings: DataFrame with detailed rankings
    """
    return enforce_mlb_playoff_rules(probabilities, leagues, team_names)


# ======== Main App Content ========

# Load models and get latest year data
xgb_model, logistic_model, scaler, feature_names, models_loaded = load_models()
available_years = get_available_years()
latest_year = available_years[0] if available_years else 2024

if not models_loaded:
    st.error(
        "Failed to load models. Please ensure both models are trained and available."
    )
    st.stop()

# Load latest year data
latest_data = load_raw_data(latest_year)

if latest_data is not None:
    # Preprocess data
    X_latest, y_latest, team_info_latest = preprocess(latest_data)

    # Get predictions from both models
    predictions = get_model_predictions(
        X_latest, xgb_model, logistic_model, scaler
    )

    # Create ensemble prediction
    ensemble_preds = get_ensemble_prediction(predictions, method="average")

    # Check if we have valid playoff data for model comparison
    has_valid_data = (
        y_latest is not None
        and len(y_latest.unique()) > 1
        and y_latest.sum() > 0
    )

    # Compare model performance if we have valid actual results
    if has_valid_data:
        metrics_comparison = compare_model_metrics(y_latest, predictions)
        comparison_df = create_prediction_comparison_df(
            team_info_latest, predictions, y_latest
        )
        agreement_stats = get_model_agreement_stats(comparison_df)
    else:
        metrics_comparison = None
        comparison_df = create_prediction_comparison_df(
            team_info_latest, predictions
        )
        agreement_stats = get_model_agreement_stats(comparison_df)

    # Model selection for playoff constraint enforcement
    st.sidebar.subheader("🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose prediction model:",
        ["XGBoost", "Logistic Regression", "Ensemble (Average)"],
        index=0,
    )

    # ======== Model Hyperparameters (Sidebar) ========
    with st.sidebar.expander("⚙️ View Model Hyperparameters"):
        st.markdown("**Current model settings:**")

        hyperparams = get_all_model_hyperparameters()

        if "error" not in hyperparams:
            if selected_model == "XGBoost":
                xgb_params = hyperparams["XGBoost"]
                st.write("🌳 **XGBoost Parameters:**")
                st.write(
                    f"• Learning Rate: {xgb_params.get('learning_rate', 'N/A')}"
                )
                st.write(f"• Max Depth: {xgb_params.get('max_depth', 'N/A')}")
                st.write(
                    f"• Trees Built: {xgb_params.get('num_boost_round', 'N/A')}"
                )

            elif selected_model == "Logistic Regression":
                lr_params = hyperparams["Logistic Regression"]
                st.write("📈 **Logistic Regression Parameters:**")
                st.write(f"• Regularization (C): {lr_params.get('C', 'N/A')}")
                st.write(
                    f"• Class Weight: {lr_params.get('class_weight', 'N/A')}"
                )
                st.write(
                    f"• Max Iterations: {lr_params.get('max_iter', 'N/A')}"
                )

            else:  # Ensemble
                st.write("🤝 **Ensemble Model:**")
                st.write("• Combines XGBoost + Logistic Regression")
                st.write("• Average of both model probabilities")

            st.markdown("*See Model Analysis page for full details*")
        else:
            st.error("Could not load hyperparameters")

    # Use selected model probabilities for playoff predictions
    if selected_model == "XGBoost":
        y_proba_latest = predictions["xgb_proba"]
    elif selected_model == "Logistic Regression":
        y_proba_latest = predictions["logistic_proba"]
    else:  # Ensemble
        y_proba_latest = ensemble_preds["ensemble_proba"]

    # Always enforce MLB playoff rules
    y_pred_constrained, latest_rankings = enforce_playoff_constraints(
        y_proba_latest,
        X_latest["LEAGUE"].values,
        team_info_latest["TEAM"].values,
    )

    # Add actual playoff results if available and valid
    if has_valid_data:
        latest_rankings["actual_playoffs"] = y_latest.values
        latest_rankings["correct"] = (
            latest_rankings["makes_playoffs_constrained"]
            == latest_rankings["actual_playoffs"]
        )

    # Display current season title and model info
    st.header(f"🏆 {latest_year} Season Playoff Predictions")

    # Model info with journey CTA
    st.info(
        f"📊 **Active Model**: {selected_model} | 🔄 Switch models using the sidebar"
    )

    # Journey callout
    st.markdown("""
    💡 **New to Machine Learning?** This dashboard shows predictions from AI models trained on 34 years of MLB data.
    Navigate to **Model Analysis** to dive deeper into how these predictions are made and compare different algorithms!
    """)

    # Define league teams in broader scope for prediction summary
    al_teams = latest_rankings[latest_rankings["league_name"] == "AL"].copy()
    nl_teams = latest_rankings[latest_rankings["league_name"] == "NL"].copy()

    # Show summary metrics for MLB rules
    al_playoff_teams = latest_rankings[
        (latest_rankings["league_name"] == "AL")
        & (latest_rankings["makes_playoffs_constrained"])
    ]
    nl_playoff_teams = latest_rankings[
        (latest_rankings["league_name"] == "NL")
        & (latest_rankings["makes_playoffs_constrained"])
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AL Playoff Teams", len(al_playoff_teams))
    col2.metric("NL Playoff Teams", len(nl_playoff_teams))
    col3.metric(
        "Total Playoff Teams", len(al_playoff_teams) + len(nl_playoff_teams)
    )
    col4.metric(
        "Non-Playoff Teams", 30 - len(al_playoff_teams) - len(nl_playoff_teams)
    )

    # Display league-by-division breakdown
    st.subheader("📋 League Standings")

    # American League
    st.write("**🇺🇸 American League**")

    for div_type in ["East", "Central", "West"]:
        div_teams = latest_rankings[
            (latest_rankings["league_name"] == "AL")
            & (latest_rankings["division_type"] == div_type)
        ].copy()

        if not div_teams.empty:
            st.write(f"**{div_type} Division:**")

            # Add status and playoff type
            div_teams["status"] = div_teams["makes_playoffs_constrained"].apply(
                lambda x: "✅ Playoffs" if x else "❌ Miss"
            )

            # Create display columns
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

            # Add actual results if available
            if has_valid_data and (
                "actual_playoffs" in div_teams.columns
                and "correct" in div_teams.columns
            ):
                div_teams["actual_status"] = div_teams["actual_playoffs"].apply(
                    lambda x: "✅ Made" if x else "❌ Missed"
                )
                div_teams["result"] = div_teams["correct"].apply(
                    lambda x: "✅ Correct" if x else "❌ Wrong"
                )
                display_cols.extend(["actual_status", "result"])
                column_names.update(
                    {"actual_status": "Actual", "result": "Result"}
                )

            st.dataframe(
                div_teams[display_cols].rename(columns=column_names),
                hide_index=True,
                use_container_width=True,
            )
            st.write("---")

    # National League
    st.write("**🇺🇸 National League**")

    for div_type in ["East", "Central", "West"]:
        div_teams = latest_rankings[
            (latest_rankings["league_name"] == "NL")
            & (latest_rankings["division_type"] == div_type)
        ].copy()

        if not div_teams.empty:
            st.write(f"**{div_type} Division:**")

            # Add status and playoff type
            div_teams["status"] = div_teams["makes_playoffs_constrained"].apply(
                lambda x: "✅ Playoffs" if x else "❌ Miss"
            )

            # Create display columns
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

            # Add actual results if available and valid
            if has_valid_data and (
                "actual_playoffs" in div_teams.columns
                and "correct" in div_teams.columns
            ):
                div_teams["actual_status"] = div_teams["actual_playoffs"].apply(
                    lambda x: "✅ Made" if x else "❌ Missed"
                )
                div_teams["result"] = div_teams["correct"].apply(
                    lambda x: "✅ Correct" if x else "❌ Wrong"
                )
                display_cols.extend(["actual_status", "result"])
                column_names.update(
                    {"actual_status": "Actual", "result": "Result"}
                )

            st.dataframe(
                div_teams[display_cols].rename(columns=column_names),
                hide_index=True,
                use_container_width=True,
            )
            st.write("---")

    # Show playoff teams summary
    st.subheader("🏆 Playoff Teams Summary")

    playoff_teams = latest_rankings[
        latest_rankings["makes_playoffs_constrained"]
    ].copy()

    # Show detailed breakdown
    st.write("**Playoff Teams by Type:**")

    # Display in a more readable format
    for league in ["AL", "NL"]:
        st.write(f"**{league}:**")
        league_playoffs = playoff_teams[playoff_teams["league_name"] == league]

        # Division winners
        div_winners = league_playoffs[
            league_playoffs["playoff_type"].str.contains("Winner")
        ]
        if not div_winners.empty:
            st.write("  *Division Winners:*")
            for _, team in div_winners.iterrows():
                st.write(
                    f"    - {team['team']} ({team['playoff_type']}) - {team['probability']:.1%}"
                )

        # Wild cards
        wild_cards = league_playoffs[
            league_playoffs["playoff_type"] == "Wild Card"
        ]
        if not wild_cards.empty:
            st.write("  *Wild Cards:*")
            for _, team in wild_cards.iterrows():
                st.write(f"    - {team['team']} - {team['probability']:.1%}")

        st.write("")

    # Show prediction summary if we have valid actual results
    if has_valid_data and (
        "actual_playoffs" in latest_rankings.columns
        and "correct" in latest_rankings.columns
    ):
        st.subheader(f"{latest_year} Playoff Prediction Summary")

        # Calculate overall accuracy
        total_teams = len(latest_rankings)
        correct_predictions = latest_rankings["correct"].sum()
        accuracy_pct = (correct_predictions / total_teams) * 100

        # Calculate league-specific accuracy
        al_correct = al_teams["correct"].sum()
        al_total = len(al_teams)
        al_accuracy = (al_correct / al_total) * 100

        nl_correct = nl_teams["correct"].sum()
        nl_total = len(nl_teams)
        nl_accuracy = (nl_correct / nl_total) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Overall Accuracy",
            f"{accuracy_pct:.1f}%",
            f"{correct_predictions}/{total_teams}",
        )
        col2.metric(
            "AL Accuracy", f"{al_accuracy:.1f}%", f"{al_correct}/{al_total}"
        )
        col3.metric(
            "NL Accuracy", f"{nl_accuracy:.1f}%", f"{nl_correct}/{nl_total}"
        )

    # Show key insights
    st.subheader("📊 Key Insights")

    # Playoff probabilities
    avg_playoff_prob = latest_rankings[
        latest_rankings["makes_playoffs_constrained"]
    ]["probability"].mean()
    min_playoff_prob = latest_rankings[
        latest_rankings["makes_playoffs_constrained"]
    ]["probability"].min()
    max_playoff_prob = latest_rankings[
        latest_rankings["makes_playoffs_constrained"]
    ]["probability"].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Playoff Team Prob", f"{avg_playoff_prob:.1%}")
    col2.metric("Lowest Playoff Prob", f"{min_playoff_prob:.1%}")
    col3.metric("Highest Playoff Prob", f"{max_playoff_prob:.1%}")

    # Bubble teams (teams just above/below cutoff)
    st.subheader("🎯 Bubble Teams")

    # CTA for deeper analysis
    st.info(
        "🔍 **Want to understand WHY these teams are predicted as bubble teams?** Check out the **Model Analysis** page to see feature importance and model decision-making!"
    )

    st.write("Teams just outside playoff contention:")

    # Find teams that just missed playoffs (next best in each league)
    bubble_teams = []
    for league in ["AL", "NL"]:
        league_teams = latest_rankings[
            latest_rankings["league_name"] == league
        ].copy()
        playoff_teams_in_league = league_teams[
            league_teams["makes_playoffs_constrained"]
        ]
        non_playoff_teams = league_teams[
            ~league_teams["makes_playoffs_constrained"]
        ]

        if len(non_playoff_teams) > 0:
            # Get top 3 non-playoff teams as bubble teams
            top_bubble = non_playoff_teams.nlargest(3, "probability")
            bubble_teams.append(top_bubble)

    if bubble_teams:
        bubble_df = pd.concat(bubble_teams)
        bubble_df["gap_to_playoffs"] = bubble_df.apply(
            lambda row: (
                latest_rankings[
                    (latest_rankings["league_name"] == row["league_name"])
                    & (latest_rankings["makes_playoffs_constrained"])
                ]["probability"].min()
                - row["probability"]
            ),
            axis=1,
        )

        # Check if division_type column exists
        if "division_type" in bubble_df.columns:
            bubble_display = bubble_df[
                [
                    "team",
                    "league_name",
                    "division_type",
                    "probability",
                    "gap_to_playoffs",
                ]
            ].rename(
                columns={
                    "team": "Team",
                    "league_name": "League",
                    "division_type": "Division",
                    "probability": "Playoff Prob",
                    "gap_to_playoffs": "Gap to Playoffs",
                }
            )
        else:
            bubble_display = bubble_df[
                ["team", "league_name", "probability", "gap_to_playoffs"]
            ].rename(
                columns={
                    "team": "Team",
                    "league_name": "League",
                    "probability": "Playoff Prob",
                    "gap_to_playoffs": "Gap to Playoffs",
                }
            )

        st.dataframe(
            bubble_display.sort_values("Playoff Prob", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

else:
    st.error(f"No data available for {latest_year}")
    st.write("Available years:", available_years)

# Navigation info
st.markdown("---")
st.subheader("🎓 Continue Your Machine Learning Journey")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 **Model Analysis**
    **Learn how predictions are made:**
    - Compare XGBoost vs Logistic Regression
    - See ROC curves and confusion matrices
    - Understand feature importance
    - Explore model agreement patterns
    """)

with col2:
    st.markdown("""
    ### 🔮 **Try Your Own Data**
    **Upload new season data:**
    - Test different scenarios
    - Compare model predictions
    - Download results for analysis
    - Generate sample predictions
    """)

with col3:
    st.markdown("""
    ### 📈 **Visualizations & Insights**
    **Explore the data science:**
    - Interactive probability charts
    - League performance trends
    - Feature importance breakdowns
    - Statistical interpretations
    """)

st.info(
    "🎯 **Pro Tip**: Start with **Model Analysis** to understand how these predictions work, then try **uploading your own data** to see the models in action!"
)
