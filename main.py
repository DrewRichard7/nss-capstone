# multiclass classification for predicting the World Series winner
# import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Years for training and validation
train_years = [y for y in range(2010, 2024) if y != 2020]
val_year = 2024

# Load and concatenate training data
dfs = []
for year in train_years:
    df = pd.read_csv(f"./data/mlb_team_stats_{year}_pre_all_star.csv")
    df["YEAR"] = year  # Optionally add year as a feature
    dfs.append(df)
train_df = pd.concat(dfs, ignore_index=True)

# Load validation data
val_df = pd.read_csv(f"./data/mlb_team_stats_{val_year}_pre_all_star.csv")
val_df["YEAR"] = val_year

# Prepare features and target
drop_cols = ["TEAM", "LEAGUE", "WON_WORLD_SERIES"]  # TEAM and LEAGUE are not numeric
X_train = train_df.drop(columns=drop_cols)
y_train = train_df["WON_WORLD_SERIES"].astype(int)

X_val = val_df.drop(columns=drop_cols)
teams_2024 = val_df["TEAM"].values  # Save for output

# Convert all columns to numeric (some may be object due to parsing)
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
X_val = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# Predict probabilities of being the winner for 2024
probs = clf.predict_proba(X_val)[:, 1]  # type: ignore

# Find the predicted winner
winner_idx = np.argmax(probs)
predicted_winner = teams_2024[winner_idx]
predicted_prob = probs[winner_idx]

# Feature importances
importances = clf.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importances)[::-1]

# Output
print("=== 2024 World Series Winner Prediction ===")
print(f"Predicted Winner: {predicted_winner}")
print(f"Predicted Probability: {predicted_prob:.4f}\n")

print("=== All 2024 Teams and Their Probabilities ===")
for team, prob in zip(teams_2024, probs):
    print(f"{team:30s}  {prob:.4f}")

print("\n=== Top 10 Feature Importances ===")
for idx in sorted_idx[:10]:
    print(f"{feature_names[idx]:20s}: {importances[idx]:.4f}")
