# AI Powered E-Commerce Customer Intelligence System

Junior Data Scientist final hackathon: inspect a raw SQLite e-commerce database, clean it, answer business questions in SQL, explore the data, predict churn, score review sentiment, and serve the results in Streamlit.

**Live app:** https://datasciencehackathon.streamlit.app/  
**Repository:** https://github.com/msajidgithub/Data-Science-Hackathon

## How to run locally

```bash
pip install -r requirements.txt
python train_and_save_models.py
streamlit run app.py
```

The app reads `ecommerce_hackathon_clean.db` and model files under `models/` using paths relative to `app.py`.

## Data

| File | Role |
|---|---|
| `ecommerce_hackathon.db` | Raw company database (do not overwrite) |
| `ecommerce_hackathon_clean.db` | Analysis-ready copy used by SQL, EDA, models, and the app |
| `business_queries.sql` | The five required Task B queries |
| `Hackathon.ipynb` | Cleaning, EDA, ML, DL, NLP |

Net revenue is `quantity × unit_price × (1 − discount)`.

## Three business insights

1. **Demand peaked in July 2026 (~PKR 83.6M) then fell about 12% in August.** Staffing, COD cash, and inventory should be planned for a mid-year spike, not a straight climb.
2. **Electronics is ~70% of net revenue.** Fashion has more orders but little revenue and the highest return rate (~11%). Protect Electronics stock and specs; fix Fashion fit/photos to cut returns.
3. **Karachi, Lahore, and Islamabad are ~54% of sales.** Put faster delivery and ads there first. Smaller cities still order often per customer — they are thin volume, not bad customers.

## Models (Task D–F)

- Churn features use orders **on or before 31 May 2026** only. The Jun–Aug 2026 window is the label (`churn = 1` if the customer did not buy).
- Logistic Regression is the model loaded in the app (similar accuracy to Random Forest, slightly higher ROC-AUC, easier to explain).
- A small MLP `32 → 16 → 1` (ReLU, log-loss) is trained for Task E. It does not beat the tabular ML models enough to justify extra complexity in production.
- Sentiment labels: rating 1–2 Negative, 3 Neutral, 4–5 Positive. TF-IDF + Logistic Regression. Limitation: many reviews are reused templates, so the same text can appear with different ratings; labels are also not written by humans.

## Deploy on Streamlit Community Cloud

1. Confirm this repo is on GitHub (including `.db` and `models/*.joblib`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. New app → this repository → branch `master` → main file `app.py`.
4. Live URL: https://datasciencehackathon.streamlit.app/
