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
    # Input validation
    if len(probabilities) != len(leagues) or len(probabilities) != len(
        team_names
    ):
        raise ValueError(
            f"Input arrays must have same length. Got probabilities: {len(probabilities)}, "
            f"leagues: {len(leagues)}, team_names: {len(team_names)}"
        )

    if len(probabilities) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Check for valid probability values
    if hasattr(probabilities, "__iter__"):
        prob_array = list(probabilities)
        if any(p < 0 or p > 1 for p in prob_array if not pd.isna(p)):
            st.warning("⚠️ Some probability values are outside [0,1] range")

    # Check for valid league values
    if hasattr(leagues, "__iter__"):
        league_array = list(leagues)
        unique_leagues = set(league_array)
        if not unique_leagues.issubset({0, 1}):
            raise ValueError(
                f"League values must be 0 (AL) or 1 (NL). Got: {unique_leagues}"
            )
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

    try:
        df = pd.DataFrame(
            {
                "team": cleaned_team_names,
                "probability": probabilities,
                "league": leagues,
                "league_name": ["AL" if l == 0 else "NL" for l in leagues],
            }
        )
    except Exception as e:
        st.error(f"❌ Error creating DataFrame with playoff data: {str(e)}")
        st.error(
            "Please check that your uploaded file has the correct format with team names and league information."
        )
        raise

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


def enforce_playoff_constraints(probabilities, leagues, team_names):
    """
    Enforce exactly 6 teams from AL and 6 teams from NL make playoffs.
    Now uses proper MLB rules with divisions and wild cards.
    """
    return enforce_mlb_playoff_rules(probabilities, leagues, team_names)


# ======== Sidebar Controls ========
st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider(
    "Playoff Probability Threshold (Unconstrained)", 0.0, 1.0, 0.50
)
enforce_constraints = st.sidebar.checkbox(
    "Enforce MLB Playoff Rules (Division Winners + Wild Cards)", value=True
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
        p_new_raw = bst.predict(dnew)

        # Transform predictions to probabilities if needed
        # XGBoost might return logits, so we need to convert to probabilities
        if np.any(p_new_raw < 0) or np.any(p_new_raw > 1):
            # Apply sigmoid transformation to convert logits to probabilities
            p_new = 1 / (1 + np.exp(-p_new_raw))
            st.info(
                "ℹ️ Applied sigmoid transformation to convert model outputs to probabilities"
            )
        else:
            p_new = p_new_raw

        # Validate probability values
        if np.any(np.isnan(p_new)) or np.any(np.isinf(p_new)):
            st.error(
                "❌ Model produced invalid probability values (NaN or Inf)"
            )
            st.stop()

        st.header("🏆 Playoff Predictions")

        if enforce_constraints and team_info_new is not None:
            # Apply playoff constraints
            try:
                constrained_preds, new_rankings = enforce_playoff_constraints(
                    p_new, X_new["LEAGUE"].values, team_info_new["TEAM"].values
                )

                # Verify the output DataFrame has the expected structure
                if new_rankings is None or len(new_rankings) == 0:
                    st.error(
                        "❌ No rankings were generated from playoff constraints"
                    )
                    st.stop()

            except Exception as e:
                st.error(f"❌ Error applying playoff constraints: {str(e)}")
                st.error(
                    "This might be due to missing or incorrect team/league data in your uploaded file."
                )
                # Show some debug info to help diagnose the issue
                st.write(
                    f"Probability values range: {p_new.min():.3f} to {p_new.max():.3f}"
                )
                st.write(f"League values: {np.unique(X_new['LEAGUE'].values)}")
                st.write(f"Number of teams: {len(team_info_new)}")
                raise

            st.subheader(
                "🏆 MLB Playoff Predictions (Division Winners + Wild Cards)"
            )

            # Summary metrics
            al_playoff_teams = new_rankings[
                (new_rankings["league_name"] == "AL")
                & (new_rankings["makes_playoffs_constrained"])
            ]
            nl_playoff_teams = new_rankings[
                (new_rankings["league_name"] == "NL")
                & (new_rankings["makes_playoffs_constrained"])
            ]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AL Playoff Teams", len(al_playoff_teams), "6/6")
            col2.metric("NL Playoff Teams", len(nl_playoff_teams), "6/6")
            col3.metric("Division Winners", "6", "3 AL + 3 NL")
            col4.metric("Wild Card Teams", "6", "3 AL + 3 NL")

            # Display predictions by division
            st.subheader("📊 Playoff Predictions by Division")

            # American League
            st.write("**🇺🇸 American League**")

            for div_type in ["East", "Central", "West"]:
                div_teams = new_rankings[
                    (new_rankings["league_name"] == "AL")
                    & (new_rankings["division_type"] == div_type)
                ].copy()

                if not div_teams.empty:
                    st.write(f"**{div_type} Division:**")

                    # Add status and playoff type
                    div_teams["status"] = div_teams[
                        "makes_playoffs_constrained"
                    ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

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

                    st.dataframe(
                        div_teams[display_cols].rename(columns=column_names),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.write("---")

            # National League
            st.write("**🇺🇸 National League**")

            for div_type in ["East", "Central", "West"]:
                div_teams = new_rankings[
                    (new_rankings["league_name"] == "NL")
                    & (new_rankings["division_type"] == div_type)
                ].copy()

                if not div_teams.empty:
                    st.write(f"**{div_type} Division:**")

                    # Add status and playoff type
                    div_teams["status"] = div_teams[
                        "makes_playoffs_constrained"
                    ].apply(lambda x: "✅ Playoffs" if x else "❌ Miss")

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

                    st.dataframe(
                        div_teams[display_cols].rename(columns=column_names),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.write("---")

            # Show playoff teams summary
            st.subheader("🏆 Playoff Teams Summary")

            playoff_teams = new_rankings[
                new_rankings["makes_playoffs_constrained"]
            ].copy()

            # Group by playoff type
            playoff_summary = (
                playoff_teams.groupby(["league_name", "playoff_type"])
                .agg({"team": list, "probability": ["count", "mean"]})
                .round(3)
            )

            st.write("**Playoff Teams by Type:**")

            # Display in a more readable format
            for league in ["AL", "NL"]:
                st.write(f"**{league}:**")
                league_playoffs = playoff_teams[
                    playoff_teams["league_name"] == league
                ]

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
                        st.write(
                            f"    - {team['team']} - {team['probability']:.1%}"
                        )

                st.write("")

            # Show bubble teams analysis (teams just outside playoffs)
            st.subheader("🎯 Bubble Teams Analysis")

            # Find teams that just missed playoffs (next best in each league)
            bubble_teams = []

            # Check if required columns exist
            required_cols = [
                "probability",
                "league_name",
                "makes_playoffs_constrained",
                "team",
                "division_type",
            ]
            missing_cols = [
                col for col in required_cols if col not in new_rankings.columns
            ]
            if missing_cols:
                st.error(
                    f"❌ Missing required columns for bubble teams analysis: {missing_cols}"
                )
                st.error(
                    "Available columns: " + str(list(new_rankings.columns))
                )
                st.stop()

            for league in ["AL", "NL"]:
                league_teams = new_rankings[
                    new_rankings["league_name"] == league
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
                try:
                    bubble_df = pd.concat(bubble_teams)
                    bubble_df["gap_to_playoffs"] = bubble_df.apply(
                        lambda row: (
                            new_rankings[
                                (
                                    new_rankings["league_name"]
                                    == row["league_name"]
                                )
                                & (new_rankings["makes_playoffs_constrained"])
                            ]["probability"].min()
                            - row["probability"]
                        ),
                        axis=1,
                    )
                except Exception as e:
                    st.error(
                        f"❌ Error calculating bubble teams gap to playoffs: {str(e)}"
                    )
                    st.error(
                        "This might be due to insufficient playoff teams in one of the leagues."
                    )
                    st.stop()

                st.write("**Teams just outside playoff contention:**")
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

                st.dataframe(
                    bubble_display.sort_values("probability", ascending=False),
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
        # Show additional debugging information
        if "df_new" in locals():
            st.write(
                f"File had {len(df_new)} rows and columns: {list(df_new.columns)}"
            )
        st.write(
            "Expected columns should include: TEAM, LEAGUE, and various statistical columns"
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
