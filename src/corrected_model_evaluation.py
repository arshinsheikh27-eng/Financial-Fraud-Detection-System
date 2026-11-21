
import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_engineered_fraud_data.csv"
)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD FEATURE-ENGINEERED DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)

# Parse dates before chronological sorting
df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

if df["Transaction_Date"].isna().any():
    raise ValueError("Invalid transaction dates found.")

# Sort transactions chronologically
df = df.sort_values(
    by="Transaction_Date"
).reset_index(drop=True)


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

target = "Fraudulent"

excluded_columns = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Suspicious_Keyword"
]

X = df.drop(
    columns=excluded_columns + [target]
)

y = df[target]

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

if X.isnull().any().any():
    print("Missing feature values found. They will be handled by imputers.")

print("Model input shape:", X.shape)


# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# --------------------------------------------------

# First 80% for model development
# Last 20% for final testing

test_split_index = int(len(X) * 0.80)

X_train = X.iloc[:test_split_index]
X_test = X.iloc[test_split_index:]

y_train = y.iloc[:test_split_index]
y_test = y.iloc[test_split_index:]


# --------------------------------------------------
# 5. CHRONOLOGICAL SUBTRAIN / VALIDATION SPLIT
# --------------------------------------------------

# 80% of training data for fitting
# 20% for threshold selection

validation_split_index = int(len(X_train) * 0.80)

X_subtrain = X_train.iloc[:validation_split_index]
X_validation = X_train.iloc[validation_split_index:]

y_subtrain = y_train.iloc[:validation_split_index]
y_validation = y_train.iloc[validation_split_index:]


print("\nDATA SPLIT SUMMARY")
print("Subtraining rows:", len(X_subtrain))
print("Validation rows:", len(X_validation))
print("Final testing rows:", len(X_test))


# --------------------------------------------------
# 6. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# --------------------------------------------------
# 7. CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# --------------------------------------------------
# 8. DEFINE MODELS
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
# 9. TRAIN, SELECT THRESHOLD AND EVALUATE
# --------------------------------------------------

results = []

for name, classifier in models.items():

    print("\n" + "=" * 55)
    print("Evaluating:", name)

    # Each model gets its own complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Fit preprocessing and model ONLY on subtraining data
    pipeline.fit(X_subtrain, y_subtrain)

    # Select threshold using validation data
    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = 0.50
    best_f1 = -1

    for threshold in [i / 100 for i in range(10, 91)]:

        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print("Selected threshold:", best_threshold)
    print("Validation F1-score:", round(best_f1, 4))

    # Final test evaluation
    # Test data is not used for fitting or threshold selection

    test_probabilities = pipeline.predict_proba(
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

    # Save complete pipeline, including fitted preprocessing
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        pipeline,
        MODEL_DIR / f"{filename}_corrected_validation_pipeline.joblib"
    )

    print("Test precision:", round(metrics["Precision"], 4))
    print("Test recall:", round(metrics["Recall"], 4))
    print("Test F1-score:", round(metrics["F1_Score"], 4))
    print("Test confusion matrix:")
    print([[tn, fp], [fn, tp]])


# --------------------------------------------------
# 10. SAVE CORRECTED RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = REPORT_DIR / "corrected_threshold_evaluation.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 65)
print("CORRECTED THRESHOLD EVALUATION RESULTS")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nResults saved at:", output_file)
print("Corrected validation pipelines saved in models/")
