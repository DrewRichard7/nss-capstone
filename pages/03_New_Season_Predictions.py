import pickle

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# ======== Page config ========
st.set_page_config(
    page_title="New Season Predictions",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 New Season Predictions & Data Upload")
st.markdown("### Upload new data to generate playoff predictions")


# ======== Import shared functions ========


@st.cache_resource
def load_model(path="assets/xgb_playoffs.pkl"):
    raw = pickle.load(open(path, "rb"))
    bst = xgb.Booster()
    bst.load_model(raw)
    return bst


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
st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider(
    "Playoff Probability Threshold (Unconstrained)", 0.0, 1.0, 0.50
)
enforce_constraints = st.sidebar.checkbox(
    "Enforce Playoff Constraints (6 AL + 6 NL)", value=True
)

st.sidebar.header("Upload Instructions")
st.sidebar.markdown(
    """
    **Upload Format Requirements:**
    - CSV file with team statistics
    - Must include TEAM and LEAGUE columns
    - Should have hitting (H_*) and pitching (P_*) stats
    - Same format as historical data
    """
)

# ======== File Upload Section ========
st.header("📁 Upload New Season Data")

uploaded_file = st.file_uploader(
    "Upload Pre-All-Star CSV for New Season Predictions",
    type="csv",
    help="Upload a CSV file with the same format as the historical data",
)

# Example of expected format
with st.expander("📋 View Expected Data Format"):
    st.markdown(
        """
        Your CSV should include these columns:
        - **TEAM**: Team name (e.g., "Los Angeles Dodgers")
        - **LEAGUE**: League designation ("AL" or "NL")
        - **Hitting Stats**: H_G, H_AB, H_R, H_H, H_2B, H_3B, H_HR, H_RBI, H_BB, H_SO, etc.
        - **Pitching Stats**: P_W, P_L, P_ERA, P_G, P_GS, P_CG, P_SHO, P_SV, P_IP, P_SO, etc.

        Optional columns:
        - **MADE_PLAYOFFS**: If you want to compare predictions (True/False)
        - **WON_WORLD_SERIES**: For additional analysis (True/False)
        """
    )

# ======== New Season Predictions ========
if uploaded_file:
    try:
        # Load and process the uploaded data
        df_new = pd.read_csv(uploaded_file)

        st.success(f"✅ Successfully loaded data with {len(df_new)} teams")

        # Display basic info about uploaded data
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Teams", len(df_new))

        if "LEAGUE" in df_new.columns:
            al_count = len(df_new[df_new["LEAGUE"] == "AL"])
            nl_count = len(df_new[df_new["LEAGUE"] == "NL"])
            col2.metric("AL Teams", al_count)
            col3.metric("NL Teams", nl_count)

        # Preprocess the data
        X_new, y_new, team_info_new = preprocess(df_new)

        # Load model and make predictions
        bst = load_model()
        dnew = xgb.DMatrix(X_new)
        p_new = bst.predict(dnew)

        st.header("🏆 Playoff Predictions")

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

            # Display predictions by league
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🇺🇸 American League**")
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
                        "status": "Prediction",
                    }
                )
                st.dataframe(
                    al_display, hide_index=True, use_container_width=True
                )

            with col2:
                st.write("**🇺🇸 National League**")
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
                        "status": "Prediction",
                    }
                )
                st.dataframe(
                    nl_display, hide_index=True, use_container_width=True
                )

            # Show bubble teams analysis
            st.subheader("🎯 Bubble Teams Analysis")
            bubble_teams = new_rankings[
                new_rankings["league_rank"].isin([5, 6, 7, 8])
            ].copy()

            if not bubble_teams.empty:
                # Calculate gap to playoffs for teams that missed
                bubble_teams["gap_to_playoffs"] = bubble_teams.apply(
                    lambda row: (
                        new_rankings[
                            (new_rankings["league_name"] == row["league_name"])
                            & (new_rankings["league_rank"] == 6)
                        ]["probability"].iloc[0]
                        - row["probability"]
                        if row["league_rank"] > 6
                        else 0
                    ),
                    axis=1,
                )

                st.write("**Teams on the playoff bubble:**")
                bubble_display = bubble_teams[
                    [
                        "team",
                        "league_name",
                        "probability",
                        "league_rank",
                        "makes_playoffs_constrained",
                    ]
                ].copy()
                bubble_display["status"] = bubble_display[
                    "makes_playoffs_constrained"
                ].apply(lambda x: "✅ In" if x else "❌ Out")

                st.dataframe(
                    bubble_display[
                        [
                            "team",
                            "league_name",
                            "probability",
                            "league_rank",
                            "status",
                        ]
                    ].rename(
                        columns={
                            "team": "Team",
                            "league_name": "League",
                            "probability": "Playoff Prob",
                            "league_rank": "League Rank",
                            "status": "Status",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            # Comparison with unconstrained if requested
            if st.checkbox("Show Comparison with Unconstrained Predictions"):
                st.subheader("⚖️ Constrained vs Unconstrained Comparison")

                unconstrained_preds = (p_new > threshold).astype(int)
                comparison_df = team_info_new.copy()
                comparison_df["Probability"] = p_new
                comparison_df["Unconstrained"] = unconstrained_preds
                comparison_df["Constrained"] = constrained_preds.astype(int)
                comparison_df["Difference"] = (
                    comparison_df["Constrained"]
                    - comparison_df["Unconstrained"]
                )

                # Show teams where predictions differ
                different_preds = comparison_df[
                    comparison_df["Difference"] != 0
                ]
                if not different_preds.empty:
                    st.write("**Teams with different predictions:**")
                    different_preds["Change"] = different_preds[
                        "Difference"
                    ].apply(
                        lambda x: "➕ Added to Playoffs"
                        if x > 0
                        else "➖ Removed from Playoffs"
                    )
                    st.dataframe(
                        different_preds[
                            ["TEAM", "LEAGUE", "Probability", "Change"]
                        ]
                        .rename(columns={"TEAM": "Team", "LEAGUE": "League"})
                        .sort_values("Probability", ascending=False),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "✨ No differences between constrained and unconstrained predictions at this threshold."
                    )

            # Show summary statistics
            st.subheader("📈 Prediction Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)

            avg_prob = p_new.mean()
            playoff_teams_avg_prob = new_rankings[
                new_rankings["makes_playoffs_constrained"]
            ]["probability"].mean()
            non_playoff_avg_prob = new_rankings[
                ~new_rankings["makes_playoffs_constrained"]
            ]["probability"].mean()
            prob_std = p_new.std()

            col1.metric("Average Probability", f"{avg_prob:.3f}")
            col2.metric("Playoff Teams Avg", f"{playoff_teams_avg_prob:.3f}")
            col3.metric("Non-Playoff Avg", f"{non_playoff_avg_prob:.3f}")
            col4.metric("Probability Std Dev", f"{prob_std:.3f}")

        else:
            # Unconstrained predictions
            st.subheader("Unconstrained Predictions (Threshold-based)")

            df_display = team_info_new.copy()
            df_display["Playoff_Prob"] = p_new
            df_display["Will_Make_Playoff"] = p_new > threshold

            total_predicted = np.sum(p_new > threshold)
            st.write(f"**{total_predicted} teams predicted to make playoffs**")

            # Add status column for better visualization
            df_display["Status"] = df_display["Will_Make_Playoff"].apply(
                lambda x: "✅ Playoffs" if x else "❌ Miss"
            )

            st.dataframe(
                df_display[["TEAM", "LEAGUE", "Playoff_Prob", "Status"]]
                .sort_values("Playoff_Prob", ascending=False)
                .reset_index(drop=True)
                .rename(columns={"TEAM": "Team", "LEAGUE": "League"}),
                hide_index=True,
                use_container_width=True,
            )

        # If actual results are available, show comparison
        if y_new is not None:
            st.subheader("🎯 Prediction vs Actual Results")
            st.info(
                "📊 Actual playoff results detected in uploaded data - showing prediction accuracy!"
            )

            if enforce_constraints:
                # Add actual results to rankings
                new_rankings["actual_playoffs"] = np.repeat(
                    y_new.values, len(new_rankings) // len(y_new)
                )[: len(new_rankings)]
                new_rankings["correct"] = (
                    new_rankings["makes_playoffs_constrained"]
                    == new_rankings["actual_playoffs"]
                )

                # Calculate accuracy
                total_correct = new_rankings["correct"].sum()
                total_teams = len(new_rankings)
                accuracy = (total_correct / total_teams) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Overall Accuracy", f"{accuracy:.1f}%")

                al_accuracy = (
                    new_rankings[new_rankings["league_name"] == "AL"][
                        "correct"
                    ].sum()
                    / len(new_rankings[new_rankings["league_name"] == "AL"])
                ) * 100
                nl_accuracy = (
                    new_rankings[new_rankings["league_name"] == "NL"][
                        "correct"
                    ].sum()
                    / len(new_rankings[new_rankings["league_name"] == "NL"])
                ) * 100

                col2.metric("AL Accuracy", f"{al_accuracy:.1f}%")
                col3.metric("NL Accuracy", f"{nl_accuracy:.1f}%")

        # Download predictions
        st.subheader("📥 Download Predictions")

        if enforce_constraints and "new_rankings" in locals():
            # Prepare download data
            download_data = new_rankings[
                [
                    "team",
                    "league_name",
                    "probability",
                    "league_rank",
                    "makes_playoffs_constrained",
                ]
            ].rename(
                columns={
                    "team": "Team",
                    "league_name": "League",
                    "probability": "Playoff_Probability",
                    "league_rank": "League_Rank",
                    "makes_playoffs_constrained": "Predicted_Playoffs",
                }
            )

            csv_data = download_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Constrained Predictions as CSV",
                data=csv_data,
                file_name="playoff_predictions_constrained.csv",
                mime="text/csv",
            )
        else:
            # Unconstrained download
            download_data = pd.DataFrame(
                {
                    "Team": team_info_new["TEAM"],
                    "League": team_info_new["LEAGUE"],
                    "Playoff_Probability": p_new,
                    "Predicted_Playoffs": p_new > threshold,
                }
            ).sort_values("Playoff_Probability", ascending=False)

            csv_data = download_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Unconstrained Predictions as CSV",
                data=csv_data,
                file_name="playoff_predictions_unconstrained.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {str(e)}")
        st.write(
            "Please check that your file format matches the expected structure."
        )

else:
    # Show instructions when no file is uploaded
    st.info(
        "👆 Upload a CSV file above to generate playoff predictions for a new season!"
    )

    st.subheader("🔍 How It Works")
    st.markdown(
        """
        1. **Upload Data**: Upload a CSV file with pre-All-Star break team statistics
        2. **Model Processing**: The trained XGBoost model analyzes the data
        3. **Generate Predictions**: Get playoff probabilities for each team
        4. **Apply Constraints**: Optionally enforce realistic playoff constraints (6 AL + 6 NL teams)
        5. **Analyze Results**: View detailed breakdowns, bubble teams, and download results

        The model uses the same features it was trained on:
        - **Hitting Statistics**: Games, at-bats, runs, hits, doubles, triples, home runs, RBIs, walks, strikeouts, etc.
        - **Pitching Statistics**: Wins, losses, ERA, games, starts, complete games, shutouts, saves, innings pitched, etc.
        - **League Information**: American League vs National League designation
        """
    )

    st.subheader("💡 Tips for Best Results")
    st.markdown(
        """
        - Ensure your data covers the **pre-All-Star break** period for most accurate predictions
        - Include **all 30 MLB teams** for complete league analysis
        - Use **consistent team naming** (preferably full names like "Los Angeles Dodgers")
        - Verify **statistical accuracy** - the model performs best with real, accurate data
        - Consider **enabling constraints** for realistic playoff scenarios
        """
    )

# Navigation info
st.markdown("---")
st.info(
    "📍 Use the sidebar to navigate between pages: League Rankings, Model Analysis, and Visualizations & Citations."
)
