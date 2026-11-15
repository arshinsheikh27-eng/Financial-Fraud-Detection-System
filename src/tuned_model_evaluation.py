
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_engineered_fraud_data.csv"
)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# --------------------------------------------------
# 1. LOAD AND SORT DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

if df["Transaction_Date"].isna().any():
    raise ValueError("Invalid transaction dates found.")

df = df.sort_values("Transaction_Date").reset_index(drop=True)

# --------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# --------------------------------------------------

target = "Fraudulent"

excluded_columns = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Suspicious_Keyword"
]

X = df.drop(columns=excluded_columns + [target])
y = df[target]

# --------------------------------------------------
# 3. RECREATE CHRONOLOGICAL SPLITS
# --------------------------------------------------

test_split = int(len(X) * 0.80)

X_train = X.iloc[:test_split]
y_train = y.iloc[:test_split]

validation_split = int(len(X_train) * 0.80)

X_validation = X_train.iloc[validation_split:]
y_validation = y_train.iloc[validation_split:]

X_test = X.iloc[test_split:]
y_test = y.iloc[test_split:]

print("Validation rows:", len(X_validation))
print("Final test rows:", len(X_test))

# --------------------------------------------------
# 4. LOAD TUNED PIPELINES
# --------------------------------------------------

models = {
    "Random Forest": MODEL_DIR / "random_forest_tuned_candidate.joblib",
    "Logistic Regression": MODEL_DIR / "logistic_regression_tuned_candidate.joblib"
}

results = []

# --------------------------------------------------
# 5. VALIDATION THRESHOLD AND TEST EVALUATION
# --------------------------------------------------

for name, model_path in models.items():

    print("\n" + "=" * 55)
    print("Evaluating:", name)

    model = joblib.load(model_path)

    # Predict probabilities on validation data
    val_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = 0.50
    best_f1 = -1

    # Find threshold that maximizes validation F1
    for threshold in [i / 100 for i in range(10, 91)]:

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_validation,
            val_predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print("Selected threshold:", best_threshold)
    print("Validation F1-score:", round(best_f1, 4))

    # Evaluate on final test data
    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "Model": name,
        "Threshold": best_threshold,
        "Validation_F1": best_f1,
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, test_predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test, test_probabilities
        ),
        "PR_AUC": average_precision_score(
            y_test, test_probabilities
        ),
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    }

    results.append(metrics)

    print("Test precision:", round(metrics["Precision"], 4))
    print("Test recall:", round(metrics["Recall"], 4))
    print("Test F1-score:", round(metrics["F1_Score"], 4))
    print("Test PR-AUC:", round(metrics["PR_AUC"], 4))
    print("Confusion matrix:")
    print([[tn, fp], [fn, tp]])


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = REPORT_DIR / "tuned_model_evaluation.csv"

results_df.to_csv(output_file, index=False)

print("\nFINAL TUNED MODEL EVALUATION")
print(results_df.to_string(index=False))

print("\nResults saved at:", output_file)
