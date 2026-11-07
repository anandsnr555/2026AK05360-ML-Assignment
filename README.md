# ShopSense: Predicting Online Shoppers' Purchase Intention

**Live app:** `<paste your Streamlit Community Cloud URL here>`

## a. Problem statement

An e-commerce site sees thousands of browsing sessions a day, and only a small share of them end in a sale. The question is: **from the way a visitor browses (pages visited, time spent, bounce and exit rates, page value, month, visitor type), can we predict whether the session will end in a purchase?**

This is a **binary classification** problem. The target is `Revenue`: 1 means the session ended in a purchase, 0 means it did not. Five classic classifiers are trained on the same data and compared with six evaluation metrics. The fitted models are served through an interactive Streamlit app that scores any uploaded, labelled test CSV without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (Sakar, Polat, Katircioglu & Kastro, 2018): https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | 12,330 sessions, each from a different user over one year |
| Features | 17 (10 numeric, 7 categorical or binary) |
| Target | `Revenue` (True/False) |
| Class balance | 10,422 No purchase (84.5%) / 1,908 Purchase (15.5%), so the data is **imbalanced** |
| Missing values | None |

**Features**

| Feature | Type | Meaning |
|---|---|---|
| Administrative, Informational, ProductRelated | int | Number of pages of each type visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | float | Seconds spent on each page type |
| BounceRates | float | Average bounce rate (Google Analytics) of the pages visited |
| ExitRates | float | Average exit rate of the pages visited |
| PageValues | float | Average value of the pages visited before a transaction |
| SpecialDay | float | How close the visit was to a special day such as Valentine's Day (0 to 1) |
| Month | categorical | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical (integer codes) | Technical and traffic-source identifiers |
| VisitorType | categorical | Returning, New or Other visitor |
| Weekend | binary | Whether the visit was on a weekend |

**Preprocessing**, built into each saved pipeline:
- `StandardScaler` on the numeric features and `Weekend`.
- `OneHotEncoder(handle_unknown="ignore")` on `Month`, `VisitorType` and the four integer-coded categorical columns. These codes are labels, not quantities, so they are one-hot encoded rather than scaled.
- An 80/20 **stratified** train/test split with `random_state=2026`. The 2,466-row test split is saved as `test_data.csv`.

## c. GitHub repository link

https://github.com/anandsnr555/2026AK05360-ML-Assignment

```
shopper-intent-classifier/
├── app.py                         # Streamlit app (loads saved models, never retrains)
├── .streamlit/config.toml         # Custom app theme
├── train_models.py                # Preprocessing, training, evaluation, model export
├── requirements.txt
├── README.md
├── test_data.csv                  # Held-out test split (17 features + Revenue)
├── data/
│   └── online_shoppers_intention.csv   # Original UCI file
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.csv                # Test-set metrics of all five models
```

**Run locally**

```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

## d. Models used

Each model is an sklearn `Pipeline` made of the preprocessing step above followed by the classifier:

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000, C=1.0)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)` |
| KNN | `KNeighborsClassifier(n_neighbors=15, weights="distance")` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | `SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)` |

### Comparison table (test set, 2,466 sessions, threshold 0.5)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8771 | 0.8797 | 0.7047 | 0.3560 | 0.4730 | 0.4428 |
| Decision Tree | **0.9006** | **0.9195** | **0.7528** | 0.5340 | **0.6248** | **0.5806** |
| KNN | 0.8715 | 0.8288 | 0.6796 | 0.3220 | 0.4369 | 0.4081 |
| Naive Bayes | 0.2810 | 0.7432 | 0.1755 | **0.9843** | 0.2978 | 0.1463 |
| Support Vector Machine | 0.8897 | 0.8743 | 0.7371 | 0.4476 | 0.5570 | 0.5185 |

Precision, Recall and F1 are for the positive class (Purchase). AUC is ROC-AUC computed from `predict_proba`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy is high (87.7%) and AUC is good (0.88), but recall is only 0.36: at the 0.5 threshold it misses about two of every three buyers. A linear boundary on the scaled features ranks sessions well, but the 85/15 class imbalance pushes predictions toward "No purchase". Lowering the threshold, or using `class_weight="balanced"`, would trade some precision for much better recall. |
| Decision Tree | Best on 5 of the 6 metrics (Accuracy 0.901, AUC 0.920, F1 0.625, MCC 0.581). `PageValues` alone accounts for about 75% of the tree's feature importance, followed by `BounceRates` and the month of November. Purchase behaviour is essentially threshold-based ("Did the visitor reach high-value pages?"), which a shallow tree (depth 6, at least 20 samples per leaf) captures directly, while staying interpretable and avoiding overfitting. |
| KNN | The weakest of the reasonable models (AUC 0.83, MCC 0.41, recall 0.32). After one-hot encoding there are more than 70 dimensions, and most of them are sparse categorical indicators. They dilute the distance contribution of the one feature that really matters (`PageValues`), so neighbourhoods are dominated by browsing look-alikes that did not buy. It is also the largest model (it stores the whole training set) and the slowest to predict. |
| Naive Bayes | Behaves almost opposite to the others: recall 0.98 but accuracy only 0.28. It labels 2,143 of 2,466 sessions as buyers, giving 1,767 false positives. GaussianNB assumes independent, normally distributed features, but the durations and page values are heavily right-skewed and zero-inflated, and the one-hot columns are binary. Those wrong likelihoods push the posterior toward "Purchase". Its AUC (0.74) shows some ranking ability, but at the default threshold it is not usable. |
| Support Vector Machine | Second best overall (MCC 0.519, F1 0.557, precision 0.737). The RBF kernel captures non-linear interactions that Logistic Regression cannot, so it beats LR on every threshold-based metric, though its AUC (0.874) is about the same. Training is the slowest of the five, because of the kernel computation plus the internal cross-validation needed for `probability=True`, and the model is hard to interpret. |
| **Overall winner for this dataset** | **Decision Tree.** It has the highest Accuracy, AUC, Precision, F1 and MCC, trains in under a second, and its splits on `PageValues`, `BounceRates` and `Month` can be explained to a business user. MCC is the most reliable single metric for this imbalanced data, and on MCC the ranking is Decision Tree > SVM > Logistic Regression > KNN > Naive Bayes. Every model would gain recall from a lower decision threshold, which you can try with the threshold slider in the app. |

## Streamlit app features

The interface is custom-built rather than a stock Streamlit template. It has its own dark theme (`.streamlit/config.toml`), custom fonts and CSS, and hand-made HTML components. The default Streamlit menu, toolbar and footer are hidden.

- **Guided sidebar** in three numbered steps: test data, classifier, decision threshold.
- **CSV upload** for labelled test data. You can also switch on the bundled `test_data.csv` or download it as a sample.
- **Model selection** dropdown covering all 5 classifiers, with a one-line profile of the chosen model.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC, shown as custom cards. Each card has a colour-coded bar and a plain-language hint.
- **Confusion matrix** drawn as a custom 2×2 grid, with labels for true and false positives and negatives and a one-line reading.
- **Classification report** as a styled table, per class plus the macro and weighted averages.
- **Model leaderboard:** all 5 models ranked by MCC, with in-cell bars and the best value in each column highlighted.
- **Score distribution:** a histogram of predicted purchase probability for buyers and non-buyers, with the decision threshold marked.
- **Predictions** tab: per-session probability bars, a filter for misclassified rows, and a CSV download.
- A **decision threshold** slider to explore the precision/recall trade-off. At 0.50 the app reproduces the table above.
