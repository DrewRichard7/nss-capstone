import glob
import os
import pickle

import pandas as pd
import streamlit as st
import xgboost as xgb

# ======== Page config ========
st.set_page_config(
    page_title="MLB Playoff Predictor",
    page_icon="⚾",
    layout="wide",
)

st.title("🏆 Current Season League Rankings & Predictions")
st.markdown("### Latest MLB Pre-All-Star Break Analysis")


# ======== CACHED RESOURCE: load the XGBoost model ========
@st.cache_resource
def load_model(path="assets/xgb_playoffs.pkl"):
    raw = pickle.load(open(path, "rb"))
    bst = xgb.Booster()
    bst.load_model(raw)
    return bst


# ======== CACHED DATA: load & preprocess 2024 validation data ========
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


# ======== Playoff constraint enforcement ========
def enforce_playoff_constraints(probabilities, leagues, team_names):
    """
    Enforce exactly 6 teams from AL and 6 teams from NL make playoffs.

    Args:
        probabilities: Array of playoff probabilities
        leagues: Array of league indicators (0=AL, 1=NL)
        team_names: Array of team names

    Returns:
        constrained_predictions: Binary array of playoff predictions
        league_rankings: DataFrame with detailed rankings
    """
    df = pd.DataFrame(
        {
            "team": team_names,
            "probability": probabilities,
            "league": leagues,
            "league_name": ["AL" if l == 0 else "NL" for l in leagues],
        }
    )

    # Rank teams within each league
    df["league_rank"] = df.groupby("league")["probability"].rank(
        method="dense", ascending=False
    )

    # Top 6 from each league make playoffs
    df["makes_playoffs_constrained"] = df["league_rank"] <= 6

    # Sort for display
    league_rankings = df.sort_values(
        ["league_name", "league_rank"]
    ).reset_index(drop=True)

    return df["makes_playoffs_constrained"].values, league_rankings


# ======== Main App Content ========

# Load model and get latest year data
bst = load_model()
available_years = get_available_years()
latest_year = available_years[0] if available_years else 2024

# Load latest year data
latest_data = load_raw_data(latest_year)

if latest_data is not None:
    # Preprocess data
    X_latest, y_latest, team_info_latest = preprocess(latest_data)

    # Make predictions
    dlatest = xgb.DMatrix(X_latest, label=y_latest)
    y_proba_latest = bst.predict(dlatest)

    # Apply constraints
    y_pred_constrained, latest_rankings = enforce_playoff_constraints(
        y_proba_latest,
        X_latest["LEAGUE"].values,
        team_info_latest["TEAM"].values,
    )

    # Add actual playoff results if available
    if y_latest is not None:
        latest_rankings["actual_playoffs"] = y_latest.values
        latest_rankings["correct"] = (
            latest_rankings["makes_playoffs_constrained"]
            == latest_rankings["actual_playoffs"]
        )

    # Display current season title
    st.header(f"{latest_year} Season Playoff Predictions")

    # Show league standings
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇺🇸 American League")
        al_teams = latest_rankings[
            latest_rankings["league_name"] == "AL"
        ].copy()

        # Create display columns
        al_teams["playoff_status"] = al_teams[
            "makes_playoffs_constrained"
        ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

        # If we have actual results, show correctness
        display_cols = ["team", "probability", "league_rank", "playoff_status"]
        column_names = {
            "team": "Team",
            "probability": "Playoff Prob",
            "league_rank": "Rank",
            "playoff_status": "Prediction",
        }

        if "actual_playoffs" in al_teams.columns:
            al_teams["actual_status"] = al_teams["actual_playoffs"].apply(
                lambda x: "✅ Made" if x else "❌ Missed"
            )
            al_teams["result"] = al_teams["correct"].apply(
                lambda x: "✅ Correct" if x else "❌ Wrong"
            )
            display_cols.extend(["actual_status", "result"])
            column_names.update({"actual_status": "Actual", "result": "Result"})

        st.dataframe(
            al_teams[display_cols].rename(columns=column_names),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.subheader("🇺🇸 National League")
        nl_teams = latest_rankings[
            latest_rankings["league_name"] == "NL"
        ].copy()

        # Create display columns
        nl_teams["playoff_status"] = nl_teams[
            "makes_playoffs_constrained"
        ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

        # If we have actual results, show correctness
        display_cols = ["team", "probability", "league_rank", "playoff_status"]
        column_names = {
            "team": "Team",
            "probability": "Playoff Prob",
            "league_rank": "Rank",
            "playoff_status": "Prediction",
        }

        if "actual_playoffs" in nl_teams.columns:
            nl_teams["actual_status"] = nl_teams["actual_playoffs"].apply(
                lambda x: "✅ Made" if x else "❌ Missed"
            )
            nl_teams["result"] = nl_teams["correct"].apply(
                lambda x: "✅ Correct" if x else "❌ Wrong"
            )
            display_cols.extend(["actual_status", "result"])
            column_names.update({"actual_status": "Actual", "result": "Result"})

        st.dataframe(
            nl_teams[display_cols].rename(columns=column_names),
            hide_index=True,
            use_container_width=True,
        )

    # Show prediction summary if we have actual results
    if "actual_playoffs" in latest_rankings.columns:
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
    st.write("Teams on the playoff bubble (ranks 5-8 in each league):")

    bubble_teams = latest_rankings[
        (latest_rankings["league_rank"] >= 5)
        & (latest_rankings["league_rank"] <= 8)
    ].sort_values(["league_name", "league_rank"])

    for league in ["AL", "NL"]:
        league_bubble = bubble_teams[bubble_teams["league_name"] == league]
        if not league_bubble.empty:
            st.write(f"**{league}:**")
            for _, team in league_bubble.iterrows():
                status = (
                    "✅ In" if team["makes_playoffs_constrained"] else "❌ Out"
                )
                st.write(
                    f"  {team['league_rank']:.0f}. {team['team']} ({team['probability']:.1%}) - {status}"
                )

else:
    st.error(f"No data available for {latest_year}")
    st.write("Available years:", available_years)

# Navigation info
st.markdown("---")
st.info(
    "📍 Navigate to other pages using the sidebar to explore model analysis, visualizations, and upload new data for predictions."
)
