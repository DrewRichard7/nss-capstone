import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ======== Page config ========
st.set_page_config(
    page_title="Model Analysis & Validation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Analysis & Validation")
st.markdown("### In-depth model performance and feature analysis")


# ======== Import shared functions ========


@st.cache_resource
def load_model(path="assets/xgb_playoffs.pkl"):
    raw = pickle.load(open(path, "rb"))
    bst = xgb.Booster()
    bst.load_model(raw)
    return bst


@st.cache_data
def load_validation_data(path="data/mlb_team_stats_2024_pre_all_star.csv"):
    df = pd.read_csv(path)
    X, y, team_info = preprocess(df)
    return X, y, team_info


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
    cleaned_team_names = []
    for team in team_names:
        # Remove invisible Unicode characters and normalize
        cleaned = team.strip()
        # Handle specific cases
        if "Athletics" in cleaned and cleaned != "Oakland Athletics":
            cleaned = "Oakland Athletics"
        elif "Guardians" in cleaned and cleaned != "Cleveland Guardians":
            cleaned = "Cleveland Guardians"
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
            f"WARNING: Some teams not found in division mapping: {missing_divisions['team'].to_list()}"
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


def enforce_playoff_constraints(probabilities, leagues, team_names):
    """
    Enforce exactly 6 teams from AL and 6 teams from NL make playoffs.
    Now uses proper MLB rules with divisions and wild cards.
    """
    return enforce_mlb_playoff_rules(probabilities, leagues, team_names)


# ======== Sidebar Controls ========
st.sidebar.header("Analysis Settings")
threshold = st.sidebar.slider(
    "Playoff Probability Threshold (Unconstrained)", 0.0, 1.0, 0.50
)
enforce_constraints = st.sidebar.checkbox(
    "Enforce MLB Playoff Rules (Division Winners + Wild Cards)", value=True
)

st.sidebar.header("Visualization Options")
show_confusion_matrix = st.sidebar.checkbox("Show Confusion Matrix", value=True)
show_roc_curve = st.sidebar.checkbox("Show ROC Curve", value=True)

st.sidebar.header("Data Explorer")
available_years = get_available_years()
selected_year = st.sidebar.selectbox(
    "Select Year for Raw Data",
    available_years,
    index=0,  # Default to most recent year
)
show_raw_data = st.sidebar.checkbox("Show Raw Data")

# ======== Load model & validation data ========
bst = load_model()
X_val, y_val, team_info_val = load_validation_data()
dval = xgb.DMatrix(X_val, label=y_val)
y_proba = bst.predict(dval)

# Apply constraints if enabled
if enforce_constraints and team_info_val is not None:
    y_pred_constrained, val_rankings = enforce_playoff_constraints(
        y_proba, X_val["LEAGUE"].values, team_info_val["TEAM"].values
    )
    y_pred = y_pred_constrained.astype(int)
    # Add actual playoff results to val_rankings
    val_rankings["actual_playoffs"] = y_val.values
    # Calculate correctness for summary
    val_rankings["correct"] = (
        val_rankings["makes_playoffs_constrained"]
        == val_rankings["actual_playoffs"]
    )
else:
    y_pred = (y_proba > threshold).astype(int)
    val_rankings = None

# ======== Model Performance Metrics ========
st.header("🎯 Unconstrained Model Performance Metrics")
st.markdown("- Not accounting for teams in divisions and wildcard structure")

accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy:.2%}")
col2.metric("ROC AUC", f"{roc_auc:.3f}")
if enforce_constraints:
    predicted_playoffs = np.sum(y_pred)
    col3.metric("Predicted Playoff Teams (Check)", f"{predicted_playoffs}/12")
else:
    predicted_playoffs = np.sum(y_pred)
    col3.metric("Predicted Playoff Teams", f"{predicted_playoffs}")

# ======== Visualizations ========
if show_confusion_matrix or show_roc_curve:
    st.header("📈 Model Performance Visualizations")
    col1, col2 = st.columns(2)

    if show_confusion_matrix:
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

    if show_roc_curve:
        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
            ax2.plot([0, 1], [0, 1], "--", color="gray", alpha=0.8)
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

# ======== 2024 Validation Results ========
if enforce_constraints and val_rankings is not None:
    st.header("🔍 2024 Validation: League-Constrained Playoff Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇺🇸 American League")
        al_teams = val_rankings[val_rankings["league_name"] == "AL"].copy()
        al_teams["predicted"] = al_teams["makes_playoffs_constrained"].apply(
            lambda x: "✅ Yes" if x else "❌ No"
        )
        al_teams["actual"] = al_teams["actual_playoffs"].apply(
            lambda x: "✅ Yes" if x else "❌ No"
        )
        al_teams["result"] = al_teams["correct"].apply(
            lambda x: "✅ Correct" if x else "❌ Wrong"
        )
        st.dataframe(
            al_teams[
                [
                    "team",
                    "probability",
                    "league_rank",
                    "predicted",
                    "actual",
                    "result",
                ]
            ].rename(
                columns={
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "league_rank": "Rank",
                    "predicted": "Predicted",
                    "actual": "Actual",
                    "result": "Result",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.subheader("🇺🇸 National League")
        nl_teams = val_rankings[val_rankings["league_name"] == "NL"].copy()
        nl_teams["predicted"] = nl_teams["makes_playoffs_constrained"].apply(
            lambda x: "✅ Yes" if x else "❌ No"
        )
        nl_teams["actual"] = nl_teams["actual_playoffs"].apply(
            lambda x: "✅ Yes" if x else "❌ No"
        )
        nl_teams["result"] = nl_teams["correct"].apply(
            lambda x: "✅ Correct" if x else "❌ Wrong"
        )
        st.dataframe(
            nl_teams[
                [
                    "team",
                    "probability",
                    "league_rank",
                    "predicted",
                    "actual",
                    "result",
                ]
            ].rename(
                columns={
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "league_rank": "Rank",
                    "predicted": "Predicted",
                    "actual": "Actual",
                    "result": "Result",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    # Show prediction accuracy summary
    st.subheader("📊 2024 Playoff Prediction Summary")

    # Calculate overall accuracy
    total_teams = len(val_rankings)
    correct_predictions = val_rankings["correct"].sum()
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

# ======== Feature Importances ========
st.header("🎯 Feature Importances by Category")
imps = bst.get_score(importance_type="weight")
imp_df = pd.DataFrame.from_dict(
    imps, orient="index", columns=["weight"]
).sort_values("weight", ascending=False)

# Create three columns for the tables
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⚾ Top Hitting Features")
    hitting_features = imp_df[imp_df.index.str.startswith("H_")].head(10)
    if not hitting_features.empty:
        hitting_display = hitting_features.reset_index()
        hitting_display.columns = ["Feature", "Importance"]
        hitting_display["Feature"] = hitting_display["Feature"].str.replace(
            "H_", ""
        )
        st.dataframe(hitting_display, hide_index=True, use_container_width=True)
    else:
        st.write("No hitting features found")

with col2:
    st.subheader("🥎 Top Pitching Features")
    pitching_features = imp_df[imp_df.index.str.startswith("P_")].head(10)
    if not pitching_features.empty:
        pitching_display = pitching_features.reset_index()
        pitching_display.columns = ["Feature", "Importance"]
        pitching_display["Feature"] = pitching_display["Feature"].str.replace(
            "P_", ""
        )
        st.dataframe(
            pitching_display, hide_index=True, use_container_width=True
        )
    else:
        st.write("No pitching features found")

with col3:
    st.subheader("🏆 Top Overall Features")
    top_features = imp_df.head(10)
    if not top_features.empty:
        top_display = top_features.reset_index()
        top_display.columns = ["Feature", "Importance"]
        st.dataframe(top_display, hide_index=True, use_container_width=True)
    else:
        st.write("No features found")

# ======== Sample Data Testing ========
st.header("🧪 Test Playoff Constraints with Sample Data")


def create_sample_data():
    """Create sample data for testing playoff constraints"""
    np.random.seed(42)

    # Use actual MLB team names
    al_teams = [
        "Baltimore Orioles",
        "Boston Red Sox",
        "New York Yankees",
        "Tampa Bay Rays",
        "Toronto Blue Jays",
        "Chicago White Sox",
        "Cleveland Guardians",
        "Detroit Tigers",
        "Kansas City Royals",
        "Minnesota Twins",
        "Houston Astros",
        "Los Angeles Angels",
        "Oakland Athletics",
        "Seattle Mariners",
        "Texas Rangers",
    ]

    nl_teams = [
        "Atlanta Braves",
        "Miami Marlins",
        "New York Mets",
        "Philadelphia Phillies",
        "Washington Nationals",
        "Chicago Cubs",
        "Cincinnati Reds",
        "Milwaukee Brewers",
        "Pittsburgh Pirates",
        "St. Louis Cardinals",
        "Arizona Diamondbacks",
        "Colorado Rockies",
        "Los Angeles Dodgers",
        "San Diego Padres",
        "San Francisco Giants",
    ]

    teams = al_teams + nl_teams
    leagues = [0] * 15 + [1] * 15  # 0 = AL, 1 = NL

    # Generate random probabilities (higher for some teams to simulate reality)
    probabilities = np.random.beta(
        2, 5, 30
    )  # Beta distribution for realistic probabilities

    return teams, leagues, probabilities


if st.button("🎲 Test with Random Sample Data"):
    st.subheader("Sample Data Test: Playoff Constraint Enforcement")

    sample_teams, sample_leagues, sample_probs = create_sample_data()

    # Show unconstrained vs constrained
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Unconstrained (Threshold-based)**")
        unconstrained_preds = sample_probs > threshold
        unconstrained_count = np.sum(unconstrained_preds)

        sample_df = pd.DataFrame(
            {
                "Team": sample_teams,
                "League": ["AL" if l == 0 else "NL" for l in sample_leagues],
                "Probability": sample_probs,
                "Makes_Playoffs": unconstrained_preds,
            }
        ).sort_values("Probability", ascending=False)

        st.write(f"**Total Playoff Teams: {unconstrained_count}**")
        st.dataframe(sample_df, hide_index=True, use_container_width=True)

    with col2:
        st.write("**Constrained (6 AL + 6 NL)**")
        constrained_preds, rankings = enforce_playoff_constraints(
            sample_probs, sample_leagues, sample_teams
        )

        constrained_count = np.sum(constrained_preds)
        st.write(f"**Total Playoff Teams: {constrained_count}**")

        # Show AL teams
        st.write("*American League*")
        al_rankings = rankings[rankings["league_name"] == "AL"].head(8)
        al_rankings["Status"] = al_rankings["makes_playoffs_constrained"].apply(
            lambda x: "✅ Playoffs" if x else "❌ Miss"
        )
        st.dataframe(
            al_rankings[
                ["team", "probability", "league_rank", "Status"]
            ].rename(
                columns={
                    "team": "Team",
                    "probability": "Prob",
                    "league_rank": "Rank",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        # Show NL teams
        st.write("*National League*")
        nl_rankings = rankings[rankings["league_name"] == "NL"].head(8)
        nl_rankings["Status"] = nl_rankings["makes_playoffs_constrained"].apply(
            lambda x: "✅ Playoffs" if x else "❌ Miss"
        )
        st.dataframe(
            nl_rankings[
                ["team", "probability", "league_rank", "Status"]
            ].rename(
                columns={
                    "team": "Team",
                    "probability": "Prob",
                    "league_rank": "Rank",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

# ======== Raw Data Display ========
if show_raw_data:
    st.header(f"📋 {selected_year} MLB Pre-All-Star Break Team Statistics")

    raw_data = load_raw_data(selected_year)
    if raw_data is not None:
        # Display summary stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Teams", len(raw_data))
        col2.metric("AL Teams", len(raw_data[raw_data["LEAGUE"] == "AL"]))
        col3.metric("NL Teams", len(raw_data[raw_data["LEAGUE"] == "NL"]))

        if "MADE_PLAYOFFS" in raw_data.columns:
            playoff_teams = raw_data["MADE_PLAYOFFS"].sum()
            col4.metric("Playoff Teams", playoff_teams)
        else:
            col4.metric("Playoff Teams", "N/A")

        # Filter options
        st.subheader("Filter Options")
        col1, col2 = st.columns(2)

        with col1:
            league_filter = st.selectbox(
                "Filter by League",
                ["All", "AL", "NL"],
                key=f"league_filter_{selected_year}",
            )

        with col2:
            if "MADE_PLAYOFFS" in raw_data.columns:
                playoff_filter = st.selectbox(
                    "Filter by Playoff Status",
                    ["All", "Made Playoffs", "Missed Playoffs"],
                    key=f"playoff_filter_{selected_year}",
                )
            else:
                playoff_filter = "All"

        # Apply filters
        filtered_data = raw_data.copy()

        if league_filter != "All":
            filtered_data = filtered_data[
                filtered_data["LEAGUE"] == league_filter
            ]

        if playoff_filter != "All" and "MADE_PLAYOFFS" in raw_data.columns:
            if playoff_filter == "Made Playoffs":
                filtered_data = filtered_data[
                    filtered_data["MADE_PLAYOFFS"] == True
                ]
            else:
                filtered_data = filtered_data[
                    filtered_data["MADE_PLAYOFFS"] == False
                ]

        # Column selection
        st.subheader("Column Selection")
        col_categories = st.multiselect(
            "Show Categories",
            ["Basic Info", "Hitting Stats", "Pitching Stats", "Results"],
            default=["Basic Info", "Results"],
            key=f"col_categories_{selected_year}",
        )

        # Define column groups
        basic_cols = ["TEAM", "LEAGUE"]
        hitting_cols = [col for col in raw_data.columns if col.startswith("H_")]
        pitching_cols = [
            col for col in raw_data.columns if col.startswith("P_")
        ]
        result_cols = []
        if "MADE_PLAYOFFS" in raw_data.columns:
            result_cols.append("MADE_PLAYOFFS")
        if "WON_WORLD_SERIES" in raw_data.columns:
            result_cols.append("WON_WORLD_SERIES")

        # Build display columns
        display_cols = []
        if "Basic Info" in col_categories:
            display_cols.extend(basic_cols)
        if "Hitting Stats" in col_categories:
            display_cols.extend(hitting_cols)
        if "Pitching Stats" in col_categories:
            display_cols.extend(pitching_cols)
        if "Results" in col_categories:
            display_cols.extend(result_cols)

        # Display the filtered data
        st.subheader(f"Data ({len(filtered_data)} teams)")
        if display_cols:
            st.dataframe(
                filtered_data[display_cols].sort_values("TEAM"),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Please select at least one category to display data.")

        # Download option
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {selected_year} Data as CSV",
            data=csv,
            file_name=f"mlb_stats_{selected_year}_filtered.csv",
            mime="text/csv",
        )

    else:
        st.error(
            f"Could not load data for {selected_year}. File may not exist."
        )
