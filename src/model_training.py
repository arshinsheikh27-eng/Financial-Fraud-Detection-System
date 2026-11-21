
import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "model_ready"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load preprocessed data
# --------------------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

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

# --------------------------------------------------
# 4. Train and evaluate baseline models
# --------------------------------------------------

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    # Predict class labels
    predictions = model.predict(X_test)

    # Predict fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(
            y_test, probabilities
        )
    }

    results.append(metrics)

    # Save individual trained model
    model_filename = (
        name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR / model_filename
    )

    print("Accuracy:", round(metrics["Accuracy"], 4))
    print("Precision:", round(metrics["Precision"], 4))
    print("Recall:", round(metrics["Recall"], 4))
    print("F1 Score:", round(metrics["F1_Score"], 4))
    print("ROC-AUC:", round(metrics["ROC_AUC"], 4))
    print("PR-AUC:", round(metrics["PR_AUC"], 4))

# --------------------------------------------------
# 5. Save model comparison results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n--------------------------------")
print("MODEL TRAINING COMPLETED")
print("--------------------------------")

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nTrained models saved in:", MODEL_DIR)
print("Comparison report saved in:", REPORT_DIR)
