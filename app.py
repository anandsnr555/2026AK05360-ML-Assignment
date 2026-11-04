"""
ShopSense - will this online shopping session end in a purchase?

Streamlit front-end for five pre-trained classifiers (see train_models.py).
Upload a labelled test CSV, choose a model, and inspect its evaluation metrics,
confusion matrix and classification report. Models are only loaded, never retrained.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
]
CLASS_NAMES = ["No purchase", "Purchase"]

st.set_page_config(page_title="ShopSense | Purchase Intent", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.2rem 1.5rem; border-radius: 14px;
           background: linear-gradient(120deg, #0f766e 0%, #155e75 100%); color: #f0fdfa;}
    .hero h1 {margin: 0; font-size: 2rem; color: #ffffff;}
    .hero p  {margin: .35rem 0 0 0; opacity: .9;}
    div[data-testid="stMetric"] {background: rgba(15,118,110,.08); border-radius: 10px;
           padding: .6rem .8rem; border: 1px solid rgba(15,118,110,.25);}
    </style>
    <div class="hero">
      <h1>🛒 ShopSense</h1>
      <p>Predicting whether an e-commerce browsing session ends in a purchase &mdash;
      five classic classifiers evaluated on your uploaded test data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the uploaded frame and split it into X / y."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Uploaded CSV is missing column(s): " + ", ".join(missing))
    df = df.copy()
    for col in ("Weekend", TARGET):
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
        df[col] = df[col].astype(int)
    return df[FEATURES], df[TARGET]


def score(y_true, y_pred, y_prob) -> dict:
    auc = roc_auc_score(y_true, y_prob) if y_true.nunique() == 2 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_data(show_spinner="Scoring sessions…")
def purchase_probability(name: str, X: pd.DataFrame) -> np.ndarray:
    return load_model(name).predict_proba(X)[:, 1]


def run_model(name: str, X: pd.DataFrame, threshold: float):
    prob = purchase_probability(name, X)
    # strict ">" so that threshold 0.5 matches model.predict() (ties go to class 0)
    pred = (prob > threshold).astype(int)
    return pred, prob


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Classifier", list(MODEL_FILES), index=1)
    uploaded = st.file_uploader("Upload labelled test CSV", type="csv")
    use_sample = st.toggle("Use bundled test_data.csv", value=uploaded is None)
    threshold = st.slider(
        "Purchase probability threshold", 0.05, 0.95, 0.50, 0.05,
        help="Sessions with predicted probability ≥ threshold are labelled 'Purchase'. "
             "0.50 reproduces the README numbers.",
    )
    st.download_button(
        "⬇️ Download sample test CSV",
        data=SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    st.caption("Dataset: UCI Online Shoppers Purchasing Intention (12,330 sessions).")

# ---------------- Data ----------------
if uploaded is not None and not use_sample:
    raw = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    raw = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("Upload a CSV in the sidebar (or switch on the bundled test data) to begin.")
    st.stop()

try:
    X, y = prepare(raw)
except ValueError as err:
    st.error(str(err))
    st.stop()

st.write(
    f"**Data:** {source} &nbsp;·&nbsp; **{len(X):,} sessions** &nbsp;·&nbsp; "
    f"{y.mean():.1%} ended in a purchase &nbsp;·&nbsp; **Model:** {model_name}"
)

y_pred, y_prob = run_model(model_name, X, threshold)
metrics = score(y, y_pred, y_prob)

cols = st.columns(3) + st.columns(3)  # 2 rows x 3 cards: readable on narrow screens
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, "n/a" if np.isnan(value) else f"{value:.3f}")

tab_cm, tab_report, tab_compare, tab_preds = st.tabs(
    ["🧮 Confusion matrix", "📋 Classification report", "🏁 Compare all models", "🔍 Predictions"]
)

with tab_cm:
    left, right = st.columns([1, 1])
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(model_name)
    left.pyplot(fig, width="stretch")
    tn, fp, fn, tp = cm.ravel()
    right.markdown(
        f"""
        **Reading the matrix**
        - ✅ **{tp}** buyers correctly flagged (true positives)
        - ✅ **{tn}** browsers correctly ignored (true negatives)
        - ⚠️ **{fp}** browsers wrongly flagged as buyers (false positives)
        - ❌ **{fn}** buyers the model missed (false negatives)

        Only ~15% of sessions convert, so accuracy alone flatters a model that
        rarely predicts *Purchase* — check Recall, F1 and MCC as well.
        """
    )

with tab_report:
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

with tab_compare:
    rows = []
    for name in MODEL_FILES:
        p, pr = run_model(name, X, threshold)
        rows.append({"Model": name, **score(y, p, pr)})
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#99f6e4"),
        width="stretch",
    )
    metric_pick = st.radio("Chart metric", list(table.columns), index=5, horizontal=True)
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ordered = table[metric_pick].sort_values()
    ax2.barh(ordered.index, ordered.values,
             color=["#0f766e" if n == model_name else "#94a3b8" for n in ordered.index])
    ax2.set_xlabel(metric_pick)
    for i, v in enumerate(ordered.values):
        ax2.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2, width="stretch")

with tab_preds:
    out = raw.copy()
    out["Predicted"] = y_pred
    out["P(Purchase)"] = y_prob.round(4)
    only_wrong = st.checkbox("Show misclassified sessions only")
    if only_wrong:
        out = out[out["Predicted"] != y.values]
    st.dataframe(out, width="stretch", height=380)
    st.download_button(
        "⬇️ Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv"
    )
