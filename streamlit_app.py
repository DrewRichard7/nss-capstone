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
    page_title="MLB Playoff Predictor",
    page_icon="⚾",
    layout="wide",
)

st.title("MLB Pre‐All‐Star Break Playoff Predictor Dashboard")


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


# ======== Sidebar ========
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Playoff Probability Threshold (Unconstrained)", 0.0, 1.0, 0.50
)
enforce_constraints = st.sidebar.checkbox(
    "Enforce Playoff Constraints (6 AL + 6 NL)", value=True
)
uploaded_file = st.sidebar.file_uploader(
    "Upload 2025 Pre‐All‐Star CSV", type="csv"
)

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
else:
    y_pred = (y_proba > threshold).astype(int)
    val_rankings = None

# ======== Metrics ========
accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("ROC AUC", f"{roc_auc:.3f}")

if enforce_constraints:
    # Show playoff distribution
    predicted_playoffs = np.sum(y_pred)
    col3.metric("Predicted Playoff Teams", f"{predicted_playoffs}/12")
else:
    predicted_playoffs = np.sum(y_pred)
    col3.metric("Predicted Playoff Teams", f"{predicted_playoffs}")

# ======== Confusion Matrix ========
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ======== ROC Curve ========
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_val, y_proba)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0, 1], [0, 1], "--", color="gray")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")
st.pyplot(fig2)

# ======== Validation Results with Constraints ========
if enforce_constraints and val_rankings is not None:
    st.subheader("2024 Validation: League-Constrained Playoff Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**American League**")
        al_teams = val_rankings[val_rankings["league_name"] == "AL"].copy()
        al_teams["status"] = al_teams["makes_playoffs_constrained"].apply(
            lambda x: "✅ Playoffs" if x else "❌ Miss"
        )
        st.dataframe(
            al_teams[["team", "probability", "league_rank", "status"]].rename(
                columns={
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "league_rank": "Rank",
                }
            ),
            hide_index=True,
        )

    with col2:
        st.write("**National League**")
        nl_teams = val_rankings[val_rankings["league_name"] == "NL"].copy()
        nl_teams["status"] = nl_teams["makes_playoffs_constrained"].apply(
            lambda x: "✅ Playoffs" if x else "❌ Miss"
        )
        st.dataframe(
            nl_teams[["team", "probability", "league_rank", "status"]].rename(
                columns={
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "league_rank": "Rank",
                }
            ),
            hide_index=True,
        )

# ======== Feature Importances ========
st.subheader("Top 10 Feature Importances")
imps = bst.get_score(importance_type="weight")
imp_df = (
    pd.DataFrame.from_dict(imps, orient="index", columns=["weight"])
    .sort_values("weight", ascending=False)
    .head(10)
)
st.bar_chart(imp_df)


# ======== Test playoff constraints with sample data ========
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


if st.sidebar.button("Test with Sample Data"):
    st.header("Sample Data Test: Playoff Constraint Enforcement")

    sample_teams, sample_leagues, sample_probs = create_sample_data()

    # Show unconstrained vs constrained
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Unconstrained (Threshold-based)")
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
        st.dataframe(sample_df, hide_index=True)

    with col2:
        st.subheader("Constrained (6 AL + 6 NL)")
        constrained_preds, rankings = enforce_playoff_constraints(
            sample_probs, sample_leagues, sample_teams
        )

        constrained_count = np.sum(constrained_preds)
        st.write(f"**Total Playoff Teams: {constrained_count}**")

        # Show AL teams
        st.write("**American League**")
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
        )

        # Show NL teams
        st.write("**National League**")
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
        )

# ======== New‐Season Predictions ========
if uploaded_file:
    st.header("2025 Playoff Predictions")
    df_new = pd.read_csv(uploaded_file)
    X_new, _, team_info_new = preprocess(df_new)
    dnew = xgb.DMatrix(X_new)
    p_new = bst.predict(dnew)

    if enforce_constraints and team_info_new is not None:
        # Apply playoff constraints
        constrained_preds, new_rankings = enforce_playoff_constraints(
            p_new, X_new["LEAGUE"].values, team_info_new["TEAM"].values
        )

        st.subheader("Constrained Playoff Predictions (6 AL + 6 NL)")

        # Summary metrics
        al_cutoff = new_rankings[
            (new_rankings["league_name"] == "AL")
            & (new_rankings["league_rank"] == 6)
        ]["probability"].iloc[0]
        nl_cutoff = new_rankings[
            (new_rankings["league_name"] == "NL")
            & (new_rankings["league_rank"] == 6)
        ]["probability"].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("AL Cutoff Probability", f"{al_cutoff:.3f}")
        col2.metric("NL Cutoff Probability", f"{nl_cutoff:.3f}")
        col3.metric("Total Playoff Teams", "12/12")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**American League**")
            al_predictions = new_rankings[
                new_rankings["league_name"] == "AL"
            ].copy()
            al_predictions["status"] = al_predictions[
                "makes_playoffs_constrained"
            ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

            # Highlight the cutoff line
            al_display = al_predictions[
                ["team", "probability", "league_rank", "status"]
            ].rename(
                columns={
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "league_rank": "Rank",
                }
            )
            st.dataframe(al_display, hide_index=True)

        with col2:
            st.write("**National League**")
            nl_predictions = new_rankings[
                new_rankings["league_name"] == "NL"
            ].copy()
            nl_predictions["status"] = nl_predictions[
                "makes_playoffs_constrained"
            ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

            nl_display = nl_predictions[
                ["team", "probability", "league_rank", "status"]
            ].rename(
                columns={
                    "team": "Team",
                    "probability": "Playoff Prob",
                    "league_rank": "Rank",
                }
            )
            st.dataframe(nl_display, hide_index=True)

        # Show bubble teams
        st.subheader("Bubble Teams Analysis")
        bubble_teams = new_rankings[
            new_rankings["league_rank"].isin([7, 8])
        ].copy()
        if not bubble_teams.empty:
            bubble_teams["gap_to_playoffs"] = bubble_teams.apply(
                lambda row: new_rankings[
                    (new_rankings["league_name"] == row["league_name"])
                    & (new_rankings["league_rank"] == 6)
                ]["probability"].iloc[0]
                - row["probability"],
                axis=1,
            )

            st.write("Teams just missing the playoffs:")
            st.dataframe(
                bubble_teams[
                    [
                        "team",
                        "league_name",
                        "probability",
                        "league_rank",
                        "gap_to_playoffs",
                    ]
                ].rename(
                    columns={
                        "team": "Team",
                        "league_name": "League",
                        "probability": "Playoff Prob",
                        "league_rank": "League Rank",
                        "gap_to_playoffs": "Gap to Playoffs",
                    }
                ),
                hide_index=True,
            )

        # Comparison with unconstrained
        if st.checkbox("Show Comparison with Unconstrained Predictions"):
            st.subheader("Constrained vs Unconstrained Comparison")

            unconstrained_preds = (p_new > threshold).astype(int)
            comparison_df = team_info_new.copy()
            comparison_df["Probability"] = p_new
            comparison_df["Unconstrained"] = unconstrained_preds
            comparison_df["Constrained"] = constrained_preds.astype(int)
            comparison_df["Difference"] = (
                comparison_df["Constrained"] - comparison_df["Unconstrained"]
            )

            # Show teams where predictions differ
            different_preds = comparison_df[comparison_df["Difference"] != 0]
            if not different_preds.empty:
                st.write("Teams with different predictions:")
                different_preds["Change"] = different_preds["Difference"].apply(
                    lambda x: "Added to Playoffs"
                    if x > 0
                    else "Removed from Playoffs"
                )
                st.dataframe(
                    different_preds[["TEAM", "LEAGUE", "Probability", "Change"]]
                    .rename(columns={"TEAM": "Team", "LEAGUE": "League"})
                    .sort_values("Probability", ascending=False),
                    hide_index=True,
                )
            else:
                st.info(
                    "No differences between constrained and unconstrained predictions at this threshold."
                )

    else:
        # Unconstrained predictions
        df_display = team_info_new.copy()
        df_display["Playoff_Prob"] = p_new
        df_display["Will_Make_Playoff"] = p_new > threshold

        total_predicted = np.sum(p_new > threshold)
        st.subheader(
            f"Unconstrained Predictions ({total_predicted} teams predicted)"
        )
        st.dataframe(
            df_display[["TEAM", "LEAGUE", "Playoff_Prob", "Will_Make_Playoff"]]
            .sort_values("Playoff_Prob", ascending=False)
            .reset_index(drop=True)
            .rename(columns={"TEAM": "Team", "LEAGUE": "League"})
        )
