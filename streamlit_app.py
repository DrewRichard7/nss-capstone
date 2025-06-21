import pickle
import pandas as pd
import xgboost as xgb
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
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
    return preprocess(df)


# ======== Preprocessing function ========
def preprocess(df):
    df = df.copy()
    # drop unused columns
    df = df.drop(columns=["TEAM", "WON_WORLD_SERIES"])
    # encode league as binary
    df["LEAGUE"] = df["LEAGUE"].map({"AL": 0, "NL": 1})
    # target → int
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
    X = df.drop(columns=["MADE_PLAYOFFS"])
    y = df["MADE_PLAYOFFS"]
    return X, y


# ======== Sidebar ========
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Playoff Probability Threshold", 0.0, 1.0, 0.50)
uploaded_file = st.sidebar.file_uploader("Upload 2025 Pre‐All‐Star CSV", type="csv")

# ======== Load model & validation data ========
bst = load_model()
X_val, y_val = load_validation_data()
dval = xgb.DMatrix(X_val, label=y_val)
y_proba = bst.predict(dval)
y_pred = (y_proba > threshold).astype(int)

# ======== Metrics ========
accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("ROC AUC", f"{roc_auc:.3f}")

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

# ======== Feature Importances ========
st.subheader("Top 10 Feature Importances")
imps = bst.get_score(importance_type="weight")
imp_df = (
    pd.DataFrame.from_dict(imps, orient="index", columns=["weight"])
    .sort_values("weight", ascending=False)
    .head(10)
)
st.bar_chart(imp_df)

# ======== New‐Season Predictions ========
if uploaded_file:
    st.header("2025 Playoff Predictions")
    df_new = pd.read_csv(uploaded_file)
    X_new, _ = preprocess(df_new)
    dnew = xgb.DMatrix(X_new)
    p_new = bst.predict(dnew)
    df_new["Playoff_Prob"] = p_new
    df_new["Will_Make_Playoff"] = p_new > threshold
    # show sorted table
    st.dataframe(
        df_new[["TEAM", "Playoff_Prob", "Will_Make_Playoff"]]
        .sort_values("Playoff_Prob", ascending=False)
        .reset_index(drop=True)
    )
