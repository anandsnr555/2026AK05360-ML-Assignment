"""
ShopSense - will this online shopping session end in a purchase?

Streamlit front-end for five pre-trained classifiers (see train_models.py).
Upload a labelled test CSV, choose a model, and inspect its evaluation metrics,
confusion matrix and classification report. Models are only loaded, never retrained.

The interface is hand-built: custom theme (.streamlit/config.toml), custom CSS and
HTML components for the hero, metric cards, confusion matrix, report and leaderboard.
"""

import html
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
MODEL_BLURB = {
    "Logistic Regression": "Linear model on scaled + one-hot features; outputs a calibrated log-odds score.",
    "Decision Tree": "Shallow tree (depth 6, ≥20 samples/leaf) that splits on thresholds such as PageValues.",
    "KNN": "Votes among the 15 nearest training sessions, weighted by inverse distance.",
    "Naive Bayes": "Gaussian likelihood per feature, assuming features are independent.",
    "Support Vector Machine": "RBF-kernel margin classifier with Platt-scaled probabilities.",
}
FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
]
CLASS_NAMES = ["No purchase", "Purchase"]
METRIC_HINT = {
    "Accuracy": "share of all sessions labelled correctly",
    "AUC": "how well buyers are ranked above non-buyers",
    "Precision": "of flagged buyers, how many really bought",
    "Recall": "of real buyers, how many were caught",
    "F1": "balance of precision and recall",
    "MCC": "correlation of prediction with truth (−1…1)",
}

# Palette (kept in sync with .streamlit/config.toml)
TEAL, AMBER, ROSE, SLATE = "#2dd4bf", "#fbbf24", "#fb7185", "#94a3b8"
BG, PANEL, INK = "#0b1220", "#121c30", "#e2e8f0"

st.set_page_config(page_title="ShopSense · Purchase Intent", page_icon="🛍️", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, input, button {font-family: 'Inter', sans-serif;}
h1, h2, h3, .ss-title {font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.01em;}

/* hide default Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}

/* sidebar */
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f1a2e 0%, #0b1220 100%);
  border-right: 1px solid rgba(45,212,191,.15);}
.ss-side-brand {font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700;
  color: #f8fafc; margin-bottom: .1rem;}
.ss-side-brand span {color: #2dd4bf;}
.ss-step {display: flex; align-items: center; gap: .55rem; margin: 1.1rem 0 .35rem 0;
  font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #94a3b8;}
.ss-step b {display: inline-grid; place-items: center; width: 1.45rem; height: 1.45rem; border-radius: 50%;
  background: rgba(45,212,191,.15); color: #2dd4bf; font-size: .75rem;}

/* hero */
.ss-hero {position: relative; overflow: hidden; border-radius: 20px; padding: 1.7rem 1.9rem;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(45,212,191,.25), transparent 60%),
              radial-gradient(800px 260px at 100% 100%, rgba(251,191,36,.18), transparent 60%), #121c30;
  border: 1px solid rgba(148,163,184,.15);}
.ss-hero .ss-title {font-size: 2.2rem; font-weight: 700; color: #f8fafc; margin: 0;}
.ss-hero .ss-title span {color: #2dd4bf;}
.ss-hero p {color: #cbd5e1; margin: .4rem 0 1rem 0; max-width: 720px;}
.ss-chips {display: flex; flex-wrap: wrap; gap: .5rem;}
.ss-chip {padding: .3rem .75rem; border-radius: 999px; font-size: .8rem; color: #e2e8f0;
  background: rgba(15,23,42,.65); border: 1px solid rgba(148,163,184,.25);}
.ss-chip b {color: #2dd4bf; font-weight: 600;}

/* section heading */
.ss-h {display: flex; align-items: baseline; gap: .6rem; margin: 1.6rem 0 .7rem 0;}
.ss-h h3 {margin: 0; font-size: 1.2rem; color: #f8fafc;}
.ss-h small {color: #94a3b8;}

/* model profile */
.ss-model {display: flex; gap: 1rem; align-items: center; padding: .9rem 1.1rem; border-radius: 14px;
  background: #121c30; border: 1px solid rgba(148,163,184,.15); margin-top: 1rem;}
.ss-model .ss-badge {font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #0b1220;
  background: #2dd4bf; border-radius: 10px; padding: .45rem .7rem; white-space: nowrap;}
.ss-model div {color: #cbd5e1; font-size: .92rem;}

/* metric cards */
.ss-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem;}
@media (max-width: 900px) {.ss-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.ss-card {background: #121c30; border: 1px solid rgba(148,163,184,.15); border-radius: 14px; padding: .9rem 1rem;}
.ss-card .lbl {font-size: .75rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #94a3b8;}
.ss-card .val {font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: #f8fafc; line-height: 1.2;}
.ss-card .bar {height: 6px; border-radius: 99px; background: rgba(148,163,184,.18); margin: .45rem 0 .4rem 0; overflow: hidden;}
.ss-card .bar i {display: block; height: 100%; border-radius: 99px;}
.ss-card .hint {font-size: .78rem; color: #94a3b8;}

/* tabs as pills */
.stTabs [data-baseweb="tab-list"] {gap: .4rem; background: #121c30; padding: .35rem; border-radius: 12px;
  border: 1px solid rgba(148,163,184,.15);}
.stTabs [data-baseweb="tab"] {border-radius: 9px; padding: .35rem .9rem; height: auto; color: #cbd5e1;}
.stTabs [aria-selected="true"] {background: rgba(45,212,191,.15) !important; color: #2dd4bf !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* confusion matrix */
.ss-cm {display: grid; grid-template-columns: 7.5rem 1fr 1fr; gap: .5rem; align-items: stretch;}
.ss-cm .ax {color: #94a3b8; font-size: .8rem; font-weight: 600; display: flex; align-items: center; justify-content: center; text-align: center;}
.ss-cm .cell {border-radius: 14px; padding: 1rem; text-align: center; border: 1px solid rgba(148,163,184,.15);}
.ss-cm .cell .n {font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700;}
.ss-cm .cell .t {font-size: .78rem; color: #cbd5e1;}

/* tables */
table.ss-table {width: 100%; border-collapse: separate; border-spacing: 0; font-size: .9rem;
  background: #121c30; border: 1px solid rgba(148,163,184,.15); border-radius: 14px; overflow: hidden;}
table.ss-table th {text-align: left; color: #94a3b8; font-weight: 600; font-size: .75rem; letter-spacing: .06em;
  text-transform: uppercase; padding: .7rem .9rem; border-bottom: 1px solid rgba(148,163,184,.15);}
table.ss-table td {padding: .6rem .9rem; color: #e2e8f0; border-bottom: 1px solid rgba(148,163,184,.08);}
table.ss-table tr:last-child td {border-bottom: none;}
table.ss-table tr.me td {background: rgba(45,212,191,.07);}
.ss-cellbar {display: flex; align-items: center; gap: .5rem;}
.ss-cellbar i {display: block; height: 6px; border-radius: 99px; background: #475569;}
.ss-cellbar i.best {background: #2dd4bf;}
.ss-crown {color: #fbbf24;}

/* buttons, uploader */
.stDownloadButton button, .stButton button {border-radius: 10px; border: 1px solid rgba(45,212,191,.4);
  background: rgba(45,212,191,.08); color: #2dd4bf;}
.stDownloadButton button:hover, .stButton button:hover {background: rgba(45,212,191,.18); border-color: #2dd4bf; color: #f8fafc;}
[data-testid="stFileUploaderDropzone"] {border: 1.5px dashed rgba(45,212,191,.45); background: rgba(45,212,191,.04); border-radius: 12px;}

/* empty state + footer */
.ss-empty {text-align: center; padding: 3rem 1rem; border: 1.5px dashed rgba(148,163,184,.3); border-radius: 18px; color: #94a3b8; margin-top: 1.2rem;}
.ss-empty .big {font-size: 2.4rem;}
.ss-foot {margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(148,163,184,.15);
  color: #64748b; font-size: .8rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def ui(markup: str) -> None:
    """Render raw HTML (lines are stripped so Markdown never treats them as code)."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    ui(f'<div class="ss-h"><h3>{title}</h3><small>{note}</small></div>')


# ---------------- Model + metric helpers ----------------
@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def prepare(df: pd.DataFrame):
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


def tone(value: float) -> str:
    if np.isnan(value):
        return SLATE
    return TEAL if value >= 0.75 else AMBER if value >= 0.5 else ROSE


# ---------------- Sidebar ----------------
with st.sidebar:
    ui('<div class="ss-side-brand">Shop<span>Sense</span></div>'
       '<div style="color:#94a3b8;font-size:.82rem">Purchase-intent model lab</div>')

    ui('<div class="ss-step"><b>1</b>Test data</div>')
    uploaded = st.file_uploader("Upload labelled test CSV", type="csv", label_visibility="collapsed")
    use_sample = st.toggle("Use bundled test_data.csv", value=uploaded is None)
    st.download_button("Download sample CSV", data=SAMPLE_CSV.read_bytes(),
                       file_name="test_data.csv", mime="text/csv", width="stretch")

    ui('<div class="ss-step"><b>2</b>Classifier</div>')
    model_name = st.selectbox("Classifier", list(MODEL_FILES), index=1, label_visibility="collapsed")

    ui('<div class="ss-step"><b>3</b>Decision threshold</div>')
    threshold = st.slider(
        "Purchase probability threshold", 0.05, 0.95, 0.50, 0.05, label_visibility="collapsed",
        help="Sessions with P(purchase) above this value are labelled 'Purchase'. "
             "0.50 reproduces the README numbers.",
    )
    ui(f'<div style="color:#94a3b8;font-size:.78rem">Label as buyer when P(purchase) &gt; '
       f'<b style="color:{TEAL}">{threshold:.2f}</b></div>')

# ---------------- Data ----------------
if uploaded is not None and not use_sample:
    raw = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    raw = pd.read_csv(SAMPLE_CSV)
    source = "test_data.csv (bundled)"
else:
    raw, source = None, None

if raw is not None:
    try:
        X, y = prepare(raw)
    except ValueError as err:
        X = y = None
        error_msg = str(err)

# ---------------- Hero ----------------
if raw is not None and X is not None:
    chips = (f'<span class="ss-chip">Data <b>{html.escape(source)}</b></span>'
             f'<span class="ss-chip"><b>{len(X):,}</b> sessions</span>'
             f'<span class="ss-chip"><b>{y.mean():.1%}</b> converted</span>'
             f'<span class="ss-chip">Model <b>{model_name}</b></span>')
else:
    chips = ('<span class="ss-chip"><b>5</b> pre-trained classifiers</span>'
             '<span class="ss-chip"><b>17</b> session features</span>'
             '<span class="ss-chip">UCI Online Shoppers dataset</span>')
ui(f"""
<div class="ss-hero">
<div class="ss-title">Will this visitor <span>buy</span>?</div>
<p>ShopSense scores e-commerce browsing sessions with five saved classifiers and shows how
well each one separates buyers from window-shoppers on the test data you provide.</p>
<div class="ss-chips">{chips}</div>
</div>
""")

if raw is None:
    ui('<div class="ss-empty"><div class="big">📂</div><b>No test data yet</b><br>'
       'Upload a labelled CSV in the sidebar, or switch on the bundled test file.</div>')
    st.stop()
if X is None:
    st.error(error_msg)
    st.stop()

y_pred, y_prob = run_model(model_name, X, threshold)
metrics = score(y, y_pred, y_prob)

ui(f'<div class="ss-model"><span class="ss-badge">{model_name}</span>'
   f'<div>{MODEL_BLURB[model_name]}</div></div>')

# ---------------- Metric cards ----------------
section("Evaluation metrics", f"computed on {len(X):,} uploaded sessions · threshold {threshold:.2f}")
cards = []
for label, value in metrics.items():
    shown = "n/a" if np.isnan(value) else f"{value:.3f}"
    width = 0 if np.isnan(value) else max(0.0, min(1.0, value)) * 100
    cards.append(
        f'<div class="ss-card"><div class="lbl">{label}</div><div class="val">{shown}</div>'
        f'<div class="bar"><i style="width:{width:.1f}%;background:{tone(value)}"></i></div>'
        f'<div class="hint">{METRIC_HINT[label]}</div></div>'
    )
ui('<div class="ss-grid">' + "".join(cards) + "</div>")

st.write("")
tab_cm, tab_report, tab_compare, tab_dist, tab_preds = st.tabs(
    ["Confusion matrix", "Classification report", "Model leaderboard", "Score distribution", "Predictions"]
)

with tab_cm:
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    total = max(1, tn + fp + fn + tp)

    def cell(n: int, title: str, good: bool) -> str:
        colour = TEAL if good else ROSE
        alpha = 0.12 + 0.55 * n / total
        rgb = "45,212,191" if good else "251,113,133"
        return (f'<div class="cell" style="background:rgba({rgb},{alpha:.2f})">'
                f'<div class="n" style="color:{colour}">{n:,}</div><div class="t">{title}</div></div>')

    ui(f"""
<div class="ss-cm">
<div></div><div class="ax">Predicted: No purchase</div><div class="ax">Predicted: Purchase</div>
<div class="ax">Actual:<br>No purchase</div>{cell(tn, "True negatives · browsers correctly ignored", True)}{cell(fp, "False positives · browsers flagged as buyers", False)}
<div class="ax">Actual:<br>Purchase</div>{cell(fn, "False negatives · buyers the model missed", False)}{cell(tp, "True positives · buyers correctly caught", True)}
</div>
""")
    ui(f'<p style="color:#94a3b8;font-size:.85rem;margin-top:.8rem">The model caught '
       f'<b style="color:{TEAL}">{tp}</b> of <b>{tp + fn}</b> real buyers and raised '
       f'<b style="color:{ROSE}">{fp}</b> false alarms. With only ~15% of sessions converting, '
       f'accuracy alone flatters a model that rarely predicts "Purchase" — read Recall, F1 and MCC too.</p>')

with tab_report:
    rep = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    rows_html = ""
    for key in CLASS_NAMES + ["macro avg", "weighted avg"]:
        r = rep[key]
        rows_html += (f"<tr><td><b>{key.title()}</b></td><td>{r['precision']:.3f}</td>"
                      f"<td>{r['recall']:.3f}</td><td>{r['f1-score']:.3f}</td>"
                      f"<td>{int(r['support']):,}</td></tr>")
    ui('<table class="ss-table"><tr><th>Class</th><th>Precision</th><th>Recall</th>'
       '<th>F1-score</th><th>Support</th></tr>' + rows_html + "</table>")
    ui(f'<p style="color:#94a3b8;font-size:.85rem;margin-top:.6rem">Overall accuracy: '
       f'<b style="color:{TEAL}">{rep["accuracy"]:.3f}</b></p>')

with tab_compare:
    results = []
    for name in MODEL_FILES:
        p, pr = run_model(name, X, threshold)
        results.append({"Model": name, **score(y, p, pr)})
    board = pd.DataFrame(results).set_index("Model").sort_values("MCC", ascending=False)
    best = board.max()
    head = "".join(f"<th>{c}</th>" for c in board.columns)
    body = ""
    for rank, (name, row) in enumerate(board.iterrows(), start=1):
        crown = ' <span class="ss-crown">★</span>' if rank == 1 else ""
        tds = ""
        for c in board.columns:
            v = row[c]
            cls = "best" if np.isclose(v, best[c]) else ""
            tds += (f'<td><div class="ss-cellbar"><i class="{cls}" style="width:{max(v, 0) * 40:.0f}px"></i>'
                    f"{v:.3f}</div></td>")
        me = ' class="me"' if name == model_name else ""
        body += f"<tr{me}><td><b>{rank}. {name}</b>{crown}</td>{tds}</tr>"
    ui(f'<div style="overflow-x:auto"><table class="ss-table" style="min-width:720px">'
       f'<tr><th>Model (ranked by MCC)</th>{head}</tr>{body}</table></div>')
    ui('<p style="color:#94a3b8;font-size:.8rem;margin-top:.5rem">Teal bars mark the best value '
       'in each column; the highlighted row is the model selected in the sidebar.</p>')

with tab_dist:
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    bins = np.linspace(0, 1, 31)
    ax.hist(y_prob[y.values == 0], bins=bins, color=SLATE, alpha=.75, label="Actual: no purchase")
    ax.hist(y_prob[y.values == 1], bins=bins, color=TEAL, alpha=.85, label="Actual: purchase")
    ax.axvline(threshold, color=AMBER, lw=2, ls="--", label=f"Threshold {threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted P(purchase)", color=INK)
    ax.set_ylabel("Sessions (log scale)", color=INK)
    ax.tick_params(colors=SLATE)
    for side in ax.spines.values():
        side.set_color("#1e293b")
    ax.legend(facecolor=PANEL, edgecolor="#1e293b", labelcolor=INK, fontsize=8)
    st.pyplot(fig, width="stretch")
    ui('<p style="color:#94a3b8;font-size:.85rem">A good model pushes teal (buyers) to the right of '
       'the dashed line and grey (non-buyers) to the left. Move the threshold in the sidebar to see '
       'the precision / recall trade-off.</p>')

with tab_preds:
    out = raw.copy()
    out["Predicted"] = np.where(y_pred == 1, "Purchase", "No purchase")
    out["P(Purchase)"] = y_prob.round(4)
    out["Correct"] = y_pred == y.values
    only_wrong = st.toggle("Show misclassified sessions only")
    if only_wrong:
        out = out[~out["Correct"]]
    st.dataframe(
        out, width="stretch", height=380, hide_index=True,
        column_config={
            "P(Purchase)": st.column_config.ProgressColumn("P(Purchase)", min_value=0.0, max_value=1.0, format="%.3f"),
            "Correct": st.column_config.CheckboxColumn("Correct"),
        },
    )
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       "shopsense_predictions.csv", "text/csv")

ui('<div class="ss-foot"><span>ShopSense · Anand Sankar (2026AK05360) · M.Tech AIML, BITS Pilani WILP</span>'
   '<span>Data: UCI Online Shoppers Purchasing Intention · models trained offline, never retrained here</span></div>')
