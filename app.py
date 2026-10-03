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

LOGO_PATH = ROOT / "assets" / "jawan_logo_clear.png"
if not LOGO_PATH.exists():
    LOGO_PATH = ROOT / "assets" / "jawan_pakistan_logo.png"

st.set_page_config(
    page_title="Jawan Pakistan | Customer Intelligence",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Outfit:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.1rem; max-width: 1180px; }
h1, h2, h3 { font-family: 'Outfit', sans-serif !important; color: #14532d !important; letter-spacing: -0.02em; }

header[data-testid="stHeader"] { background: transparent; }
/* Keep the sidebar reopen chevron visible (do not hide the whole toolbar). */
.stAppDeployButton { display: none; }
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 1000000 !important;
    pointer-events: auto !important;
}
[data-testid="collapsedControl"] button,
[data-testid="stExpandSidebarButton"] {
    background: #ffffff !important;
    color: #14532d !important;
    border: 1px solid #dce7df !important;
    box-shadow: 0 4px 14px rgba(20, 83, 45, 0.18) !important;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dce7df;
}
[data-testid="stSidebar"] * { color: #14532d !important; }
[data-testid="stSidebar"] .stRadio label { font-weight: 700; }
[data-testid="stSidebar"] img {
    background: #fff;
    border: 1px solid #dce7df;
    border-radius: 16px;
    padding: 12px 14px;
}

/* Main-page logo only when the sidebar is closed (sidebar already has the logo). */
[data-testid="stSidebar"][aria-expanded="true"] ~ * .st-key-main_brand_logo {
    display: none !important;
}

.brand-card {
    background: #fff;
    border: 1px solid #dce7df;
    border-radius: 20px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(20, 83, 45, 0.08);
}
.page-title { margin: .2rem 0 .15rem 0; font-size: 2.05rem; }
.page-sub { color: #4d6b5c; margin: 0; }

.kpi-row { display: flex; gap: 16px; margin: 0 0 1.2rem 0; }
.kpi {
    flex: 1;
    background: #fff;
    border: 1px solid #dce7df;
    border-radius: 18px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 10px 24px rgba(20, 83, 45, 0.07);
}
.kpi .label { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #5f7a6c; font-weight: 700; }
.kpi .value { font-family: Outfit, sans-serif; font-size: 1.7rem; color: #14532d; margin-top: .35rem; font-weight: 700; }
.kpi.gold { border-top: 5px solid #c5a046; }
.kpi.teal { border-top: 5px solid #006b3f; }
.kpi.stone { border-top: 5px solid #2f6b4f; }

.footer-brand {
    margin-top: 1.5rem; padding: .8rem 0; color: #5f7a6c;
    font-size: .88rem; border-top: 1px solid #dce7df;
}
.result-bad { background: #b91c1c; color: #fff; border-radius: 16px; padding: 1.1rem 1.2rem; }
.result-ok { background: #15803d; color: #fff; border-radius: 16px; padding: 1.1rem 1.2rem; }
.result-pos { background: #006b3f; color: #fff; border-radius: 16px; padding: 1.1rem 1.2rem; }
.result-neu { background: #78716c; color: #fff; border-radius: 16px; padding: 1.1rem 1.2rem; }
.result-neg { background: #c2410c; color: #fff; border-radius: 16px; padding: 1.1rem 1.2rem; }

.explain-grid { display: flex; gap: 14px; margin: 0 0 1.15rem 0; }
.explain {
    flex: 1;
    background: #fff;
    border: 1px solid #dce7df;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 20px rgba(20, 83, 45, 0.06);
}
.explain h4 { font-family: Outfit, sans-serif; color: #14532d; margin: 0 0 .4rem 0; font-size: 1.05rem; }
.explain p { color: #3f5c4e; margin: 0; line-height: 1.45; font-size: 0.98rem; }
.sent-result h3 { color: #fff !important; font-size: 1.55rem !important; margin: 0 0 .35rem 0; }
.sent-result p { margin: 0; font-size: 1.05rem; line-height: 1.45; opacity: .96; }
[data-testid="stTextArea"] textarea {
    font-size: 1.05rem !important;
    line-height: 1.5 !important;
    min-height: 160px;
}
</style>
"""


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


def hero(title: str, subtitle: str):
    left, right = st.columns([1.45, 1], vertical_alignment="center")
    with left:
        st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)
    with right:
        if LOGO_PATH.exists():
            with st.container(key="main_brand_logo"):
                st.image(str(LOGO_PATH), use_container_width=True)


def brand_footer():
    st.markdown(
        '<div class="footer-brand">Jawan Pakistan · Final Data Science Hackathon · AI E-Commerce Intelligence</div>',
        unsafe_allow_html=True,
    )


def kpi_html(items):
    colors = ["teal", "gold", "stone"]
    cards = []
    for i, (label, value) in enumerate(items):
        cards.append(
            f'<div class="kpi {colors[i % 3]}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>'
        )
    st.markdown('<div class="kpi-row">' + "".join(cards) + "</div>", unsafe_allow_html=True)


st.markdown(CSS, unsafe_allow_html=True)

import streamlit.components.v1 as components
components.html(
    """
<script>
(function () {
  const doc = window.parent.document;
  function sidebarOpen() {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb) return true;
    if (sb.getAttribute("aria-expanded") === "false") return false;
    return sb.getBoundingClientRect().width > 80;
  }
  function apply() {
    const open = sidebarOpen();
    doc.querySelectorAll(".st-key-main_brand_logo").forEach(function (el) {
      el.style.display = open ? "none" : "block";
    });
  }
  apply();
  new MutationObserver(apply).observe(doc.body, {
    attributes: true,
    subtree: true,
    attributeFilter: ["aria-expanded", "style", "class"],
  });
})();
</script>
""",
    height=0,
)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

st.sidebar.caption("Final Data Science Hackathon")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Churn Prediction", "Sentiment Analysis"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Data:** cleaned SQLite")
st.sidebar.markdown("**Churn:** Logistic Regression")
st.sidebar.markdown("**NLP:** TF-IDF + LR")

if page == "Dashboard":
    hero("Business dashboard", "KPIs and charts are calculated with SQL on ecommerce_hackathon_clean.db.")

    kpi = run_sql(
        """
        SELECT
            (SELECT ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) FROM orders) AS total_revenue,
            (SELECT COUNT(*) FROM orders) AS total_orders,
            (SELECT COUNT(*) FROM customers) AS total_customers
        """
    )
    kpi_html([
        ("Total net revenue", f"PKR {kpi['total_revenue'].iloc[0]:,.0f}"),
        ("Total orders", f"{int(kpi['total_orders'].iloc[0]):,}"),
        ("Total customers", f"{int(kpi['total_customers'].iloc[0]):,}"),
    ])

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
    left, right = st.columns(2)
    with left:
        st.subheader("Monthly net revenue")
        st.area_chart(monthly.set_index("month")["net_revenue"], color="#006b3f")
    with right:
        st.subheader("Category net revenue")
        st.bar_chart(cat.set_index("category")["revenue"], color="#d97706")

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
    st.dataframe(top_cust, use_container_width=True, hide_index=True)

elif page == "Churn Prediction":
    hero("Will this customer go quiet?", "Saved Logistic Regression · features match the 31 May 2026 snapshot.")
    model = load_churn_model()

    col_a, col_b = st.columns(2)
    with col_a:
        total_orders = st.number_input("Total orders", min_value=1, value=5)
        total_spending = st.number_input("Total spending (PKR)", min_value=0.0, value=25000.0, step=500.0)
        avg_order_value = st.number_input("Average order value (PKR)", min_value=0.0, value=5000.0, step=100.0)
        days_since_last_order = st.number_input("Days since last order", min_value=0, value=60)
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
        if pred == 1:
            st.markdown(
                f'<div class="result-bad"><h3>Likely to churn</h3>'
                f'<p>Probability {proba:.1%}. Prioritise a win-back offer or a delivery check.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-ok"><h3>Likely to stay active</h3>'
                f'<p>Churn probability {proba:.1%}. Keep them in the core loyalty track.</p></div>',
                unsafe_allow_html=True,
            )
        st.progress(min(max(proba, 0.0), 1.0), text=f"Churn risk {proba:.1%}")

else:
    hero(
        "Review sentiment",
        "Paste a customer review. The model reads the words and tells you if the buyer is happy, mixed, or unhappy — so the team does not have to open thousands of reviews one by one.",
    )
    st.markdown(
        """
<div class="explain-grid">
  <div class="explain">
    <h4>What it does</h4>
    <p>It classifies review text as <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b> using the saved TF-IDF + Logistic Regression model.</p>
  </div>
  <div class="explain">
    <h4>Why it helps the business</h4>
    <p>Support can jump to angry reviews first. Product can see quality complaints. Marketing can keep happy quotes. That saves time and protects ratings.</p>
  </div>
  <div class="explain">
    <h4>How to use it</h4>
    <p>Type or paste one review below and click <b>Read this review</b>. You get a label, confidence, and a suggested next action.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    nlp = load_sentiment_model()
    samples = {
        "Happy review": "Very satisfied, quality feels premium. Good value for money. I would buy this again.",
        "Mixed review": "Decent but could be better, support response was slow. No strong opinion.",
        "Angry review": "Very disappointed, item stopped working quickly. I want a refund.",
    }
    st.caption("Try a sample, or write your own:")
    s1, s2, s3 = st.columns(3)
    for col, (label, sample) in zip((s1, s2, s3), samples.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["review_text"] = sample
                st.rerun()

    text = st.text_area(
        "Customer review text",
        key="review_text",
        height=170,
        placeholder="Example: packing was neat but the colour was different. would not buy again.",
    )
    if st.button("Read this review", type="primary"):
        cleaned = (text or "").strip().lower()
        if len(cleaned) < 5:
            st.warning("Please enter a longer review so the model has enough words to judge.")
        else:
            pred = nlp.predict([cleaned])[0]
            proba_map = dict(zip(nlp.classes_, nlp.predict_proba([cleaned])[0]))
            conf = float(proba_map[pred])
            actions = {
                "Positive": "Keep this customer close: ask for a public review or offer a related product.",
                "Neutral": "Experience was average. Follow up on delivery or product fit before they go quiet.",
                "Negative": "This buyer is unhappy. Route to support now — refund, replacement, or a call — before they churn.",
            }
            css = {"Positive": "result-pos", "Neutral": "result-neu", "Negative": "result-neg"}[pred]
            st.markdown(
                f'<div class="{css} sent-result"><h3>{pred}  ·  {conf:.0%} confidence</h3>'
                f"<p>{actions[pred]}</p></div>",
                unsafe_allow_html=True,
            )
            c_neg, c_neu, c_pos = st.columns(3)
            c_neg.metric("Negative", f"{proba_map.get('Negative', 0):.0%}")
            c_neu.metric("Neutral", f"{proba_map.get('Neutral', 0):.0%}")
            c_pos.metric("Positive", f"{proba_map.get('Positive', 0):.0%}")

brand_footer()
