"""Train churn (LR, RF, MLP) and sentiment models, then save them for Streamlit."""
from pathlib import Path

import joblib
import pandas as pd
import sqlite3
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

conn = sqlite3.connect(ROOT / "ecommerce_hackathon_clean.db")
customers = pd.read_sql("SELECT * FROM customers", conn)
orders = pd.read_sql("SELECT * FROM orders", conn, parse_dates=["order_date"])
reviews = pd.read_sql("SELECT * FROM reviews", conn)
conn.close()

FEATURE_END = pd.Timestamp("2026-05-31")
TARGET_START = pd.Timestamp("2026-06-01")
TARGET_END = pd.Timestamp("2026-08-31")

orders["net_revenue"] = orders["quantity"] * orders["unit_price"] * (1 - orders["discount"])
history = orders[orders["order_date"] <= FEATURE_END]
target = orders[(orders["order_date"] >= TARGET_START) & (orders["order_date"] <= TARGET_END)]
active = set(target["customer_id"])

churn_df = history.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    total_spending=("net_revenue", "sum"),
    avg_order_value=("net_revenue", "mean"),
    last_order_date=("order_date", "max"),
    return_rate=("returned", "mean"),
    avg_delivery_days=("delivery_days", "mean"),
).reset_index()
churn_df["days_since_last_order"] = (FEATURE_END - churn_df["last_order_date"]).dt.days
churn_df = churn_df.drop(columns="last_order_date")
churn_df["churn"] = (~churn_df["customer_id"].isin(active)).astype(int)
churn_df = churn_df.merge(customers[["customer_id", "age", "membership_type"]], on="customer_id")

feature_cols = [
    "total_orders", "total_spending", "avg_order_value",
    "days_since_last_order", "return_rate", "avg_delivery_days",
    "age", "membership_type",
]
num_cols = [c for c in feature_cols if c != "membership_type"]
X = churn_df[feature_cols]
y = churn_df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pre_scaled = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["membership_type"]),
])
pre_rf = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["membership_type"]),
])

log_reg = Pipeline([
    ("pre", pre_scaled),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
rand_forest = Pipeline([
    ("pre", pre_rf),
    ("clf", RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    )),
])
mlp = Pipeline([
    ("pre", pre_scaled),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=42,
    )),
])
log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)
mlp.fit(X_train, y_train)

joblib.dump(log_reg, MODELS / "churn_logreg.joblib", protocol=4)
joblib.dump(rand_forest, MODELS / "churn_rf.joblib", protocol=4)
joblib.dump(mlp, MODELS / "churn_mlp.joblib", protocol=4)
joblib.dump(
    {"feature_cols": feature_cols, "membership_types": sorted(churn_df["membership_type"].unique())},
    MODELS / "churn_meta.joblib",
    protocol=4,
)

# Sentiment
sent = reviews.copy()
sent["review_text"] = sent["review_text"].fillna("").str.strip().str.lower()
sent = sent[sent["review_text"].str.len() >= 10].copy()

def to_label(rating):
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"

sent["sentiment"] = sent["rating"].map(to_label)
sentiment_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])
sentiment_pipe.fit(sent["review_text"], sent["sentiment"])
joblib.dump(sentiment_pipe, MODELS / "sentiment_lr.joblib", protocol=4)
print("Saved models in", MODELS)
