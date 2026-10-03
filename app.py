"""AI Powered E-Commerce Customer Intelligence — Streamlit app (Task G)."""
from pathlib import Path

import joblib
import pandas as pd
import sqlite3
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "ecommerce_hackathon_clean.db"
if not DB_PATH.exists():
    DB_PATH = ROOT / "ecommerce_hackathon.db"

CHURN_MODEL_PATH = ROOT / "models" / "churn_logreg.joblib"
SENTIMENT_MODEL_PATH = ROOT / "models" / "sentiment_lr.joblib"
META_PATH = ROOT / "models" / "churn_meta.joblib"

st.set_page_config(page_title="E-Commerce Intelligence", page_icon="🛒", layout="wide")


@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_resource
def load_churn_model():
    return joblib.load(CHURN_MODEL_PATH)


@st.cache_resource
def load_sentiment_model():
    return joblib.load(SENTIMENT_MODEL_PATH)


@st.cache_data
def run_sql(query: str):
    return pd.read_sql(query, get_conn())


st.sidebar.title("E-Commerce Intelligence")
st.sidebar.caption("Junior DS Hackathon")
page = st.sidebar.radio(
    "Section",
    ["Dashboard", "Churn Prediction", "Sentiment Analysis"],
)

# --- Dashboard (at least 3 outputs from SQL) ---
if page == "Dashboard":
    st.title("Business Dashboard")
    st.write("KPIs below are calculated with SQL on the cleaned SQLite database.")

    kpi = run_sql(
        """
        SELECT
            (SELECT ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) FROM orders) AS total_revenue,
            (SELECT COUNT(*) FROM orders) AS total_orders,
            (SELECT COUNT(*) FROM customers) AS total_customers
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total net revenue (PKR)", f"{kpi['total_revenue'].iloc[0]:,.0f}")
    c2.metric("Total orders", f"{int(kpi['total_orders'].iloc[0]):,}")
    c3.metric("Total customers", f"{int(kpi['total_customers'].iloc[0]):,}")

    monthly = run_sql(
        """
        SELECT strftime('%Y-%m', order_date) AS month,
               ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) AS net_revenue
        FROM orders
        GROUP BY month
        ORDER BY month
        """
    )
    monthly["month"] = pd.to_datetime(monthly["month"])
    st.subheader("Monthly net revenue")
    st.line_chart(monthly.set_index("month")["net_revenue"])

    cat = run_sql(
        """
        SELECT p.category,
               ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS revenue
        FROM orders o
        JOIN products p ON p.product_id = o.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
        """
    )
    st.subheader("Category net revenue")
    st.bar_chart(cat.set_index("category")["revenue"])

    top_cust = run_sql(
        """
        SELECT c.customer_name, c.city, COUNT(o.order_id) AS number_of_orders,
               ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS total_spending
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.customer_name, c.city
        ORDER BY total_spending DESC
        LIMIT 10
        """
    )
    st.subheader("Top 10 customers by spending")
    st.dataframe(top_cust, use_container_width=True)

# --- Churn ---
elif page == "Churn Prediction":
    st.title("Customer churn prediction")
    st.write("Uses the saved Logistic Regression pipeline (`models/churn_logreg.joblib`).")
    model = load_churn_model()

    col_a, col_b = st.columns(2)
    with col_a:
        total_orders = st.number_input("Total orders", min_value=1, value=5)
        total_spending = st.number_input("Total spending (PKR)", min_value=0.0, value=25000.0, step=500.0)
        avg_order_value = st.number_input("Average order value (PKR)", min_value=0.0, value=5000.0, step=100.0)
        days_since_last_order = st.number_input("Days since last order (as of 31 May 2026)", min_value=0, value=60)
    with col_b:
        return_rate = st.slider("Return rate", 0.0, 1.0, 0.05, 0.01)
        avg_delivery_days = st.number_input("Average delivery days", min_value=1.0, value=4.0, step=0.5)
        age = st.number_input("Age", min_value=18, max_value=80, value=31)
        membership_type = st.selectbox("Membership type", ["Standard", "Silver", "Gold", "Premium"])

    if st.button("Predict churn", type="primary"):
        row = pd.DataFrame([{
            "total_orders": total_orders,
            "total_spending": total_spending,
            "avg_order_value": avg_order_value,
            "days_since_last_order": days_since_last_order,
            "return_rate": return_rate,
            "avg_delivery_days": avg_delivery_days,
            "age": age,
            "membership_type": membership_type,
        }])
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)
        label = "Likely to churn" if pred == 1 else "Likely to stay active"
        st.metric("Churn probability", f"{proba:.1%}")
        if pred == 1:
            st.error(label)
        else:
            st.success(label)

# --- Sentiment ---
else:
    st.title("Review sentiment")
    st.write("Uses the saved TF-IDF + Logistic Regression model (`models/sentiment_lr.joblib`).")
    nlp = load_sentiment_model()
    text = st.text_area("Customer review", height=140, placeholder="Type or paste a product review...")
    if st.button("Predict sentiment", type="primary"):
        cleaned = (text or "").strip().lower()
        if len(cleaned) < 5:
            st.warning("Please enter a longer review.")
        else:
            pred = nlp.predict([cleaned])[0]
            probs = nlp.predict_proba([cleaned])[0]
            st.subheader(f"Predicted: {pred}")
            st.bar_chart(pd.Series(probs, index=nlp.classes_, name="probability"))
