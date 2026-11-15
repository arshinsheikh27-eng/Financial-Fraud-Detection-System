
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "model_ready"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load existing data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Chronological validation split within training data
split_index = int(len(X_train) * 0.80)

X_subtrain = X_train.iloc[:split_index]
X_validation = X_train.iloc[split_index:]

y_subtrain = y_train.iloc[:split_index]
y_validation = y_train.iloc[split_index:]

print("Subtraining rows:", len(X_subtrain))
print("Validation rows:", len(X_validation))
print("Final testing rows:", len(X_test))

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():

    print("\nEvaluating:", name)

    # Train only on subtraining data
    model.fit(X_subtrain, y_subtrain)

    # Validation probabilities
    val_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # Find threshold with best validation F1
    best_threshold = 0.50
    best_f1 = -1

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

    print("Selected validation threshold:", best_threshold)
    print("Validation F1:", round(best_f1, 4))

    # Evaluate once on untouched test data
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
        "Accuracy": accuracy_score(
            y_test, test_predictions
        ),
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

    # Save this validation-trained model
    filename = name.lower().replace(" ", "_")
    joblib.dump(
        model,
        MODEL_DIR / f"{filename}_validation.joblib"
    )

    print("Test precision:", round(metrics["Precision"], 4))
    print("Test recall:", round(metrics["Recall"], 4))
    print("Test F1:", round(metrics["F1_Score"], 4))
    print("Test confusion matrix:")
    print([[tn, fp], [fn, tp]])

# Save comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "threshold_evaluation.csv",
    index=False
)

print("\nFINAL THRESHOLD EVALUATION")
print(results_df.to_string(index=False))

print("\nResults saved in reports/threshold_evaluation.csv")
