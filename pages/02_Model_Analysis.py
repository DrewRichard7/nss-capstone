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


def enforce_playoff_constraints(probabilities, leagues, team_names):
    """
    Enforce exactly 6 teams from AL and 6 teams from NL make playoffs.
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


# ======== Sidebar Controls ========
st.sidebar.header("Analysis Settings")
threshold = st.sidebar.slider(
    "Playoff Probability Threshold (Unconstrained)", 0.0, 1.0, 0.50
)
enforce_constraints = st.sidebar.checkbox(
    "Enforce Playoff Constraints (6 AL + 6 NL)", value=True
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
st.header("🎯 Model Performance Metrics")

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

    # Create 30 teams (15 AL, 15 NL)
    al_teams = [f"AL Team {i}" for i in range(1, 16)]
    nl_teams = [f"NL Team {i}" for i in range(1, 16)]

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
