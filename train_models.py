"""
Train five classifiers on the UCI Online Shoppers Purchasing Intention dataset.

Outputs
-------
model/<name>.joblib   one fitted sklearn Pipeline (preprocessing + classifier) per model
model/metrics.csv     evaluation metrics of every model on the held-out test split
test_data.csv         the held-out test split (raw features + target), for the Streamlit app

Run:  python train_models.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "Revenue"
SEED = 2026

# Session-behaviour features (counts, durations, Google Analytics rates) -> scaled
NUMERIC_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend",
]
# Integer codes that are really labels (OS id, browser id, ...) -> one-hot
CATEGORICAL_COLS = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # pandas reads TRUE/FALSE as bool; store as 0/1 so the CSV is unambiguous
    df["Weekend"] = df["Weekend"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=SEED
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED
        ),
    }


def evaluate(model, X, y) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    # Label from the probability (threshold 0.5) for every model, so the numbers
    # match the Streamlit app. Note: SVC.predict() uses the decision function,
    # which can disagree slightly with its Platt-scaled predict_proba().
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main() -> None:
    df = load_dataset(RAW_CSV)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]
    print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features")
    print(f"Class balance (Revenue=1): {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Hold-out split saved in raw form -> uploaded to the Streamlit app later
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved {len(test_df)} test rows -> {TEST_CSV.name}")

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name, clf in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / file_name(name))

        scores = evaluate(pipe, X_test, y_test)
        rows.append({"Model": name, **scores})
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n" + metrics.to_string(index=False))

    # MCC is the most reliable single score for this imbalanced target
    ranking = metrics.sort_values("MCC", ascending=False).reset_index(drop=True)
    winner = ranking.loc[0]
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    wins = [c for c in metric_cols if winner[c] == metrics[c].max()]
    print("\nRanking by MCC:")
    for i, row in ranking.iterrows():
        print(f"  {i + 1}. {row['Model']:<24} MCC={row['MCC']:.4f}")
    print(f"\nOVERALL WINNER: {winner['Model']} "
          f"(MCC={winner['MCC']:.4f}, F1={winner['F1']:.4f}, AUC={winner['AUC']:.4f}; "
          f"best on {len(wins)} of {len(metric_cols)} metrics: {', '.join(wins)})")


if __name__ == "__main__":
    main()
