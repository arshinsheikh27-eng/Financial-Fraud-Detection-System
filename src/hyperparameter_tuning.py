
import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
# 2. LOAD AND SORT DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

if df["Transaction_Date"].isna().any():
    raise ValueError("Invalid transaction dates found.")

df = df.sort_values(
    "Transaction_Date"
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


# --------------------------------------------------
# 4. CHRONOLOGICAL DATA SPLIT
# --------------------------------------------------

test_split = int(len(X) * 0.80)

X_train = X.iloc[:test_split]
y_train = y.iloc[:test_split]

# Keep validation and final test separate
validation_split = int(len(X_train) * 0.80)

X_subtrain = X_train.iloc[:validation_split]
y_subtrain = y_train.iloc[:validation_split]

X_validation = X_train.iloc[validation_split:]
y_validation = y_train.iloc[validation_split:]

X_test = X.iloc[test_split:]
y_test = y.iloc[test_split:]

print("Subtraining rows:", len(X_subtrain))
print("Validation rows:", len(X_validation))
print("Final test rows:", len(X_test))


# --------------------------------------------------
# 5. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# --------------------------------------------------
# 6. CREATE PREPROCESSING PIPELINE
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
# 7. CREATE CROSS-VALIDATION STRATEGY
# --------------------------------------------------

time_cv = TimeSeriesSplit(n_splits=4)


# --------------------------------------------------
# 8. DEFINE MODEL PIPELINES
# --------------------------------------------------

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )
        )
    ]
)

lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# 9. DEFINE HYPERPARAMETER SEARCH SPACES
# --------------------------------------------------

rf_parameters = {
    "classifier__n_estimators": [200, 300],
    "classifier__max_depth": [8, 12, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__max_features": ["sqrt", 0.8]
}

lr_parameters = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}


# --------------------------------------------------
# 10. TUNE RANDOM FOREST
# --------------------------------------------------

print("\nStarting Random Forest tuning...")

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_parameters,
    n_iter=20,
    scoring="average_precision",
    cv=time_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

rf_search.fit(X_subtrain, y_subtrain)

print("\nRANDOM FOREST RESULTS")
print("Best parameters:", rf_search.best_params_)
print("Best CV Average Precision:", rf_search.best_score_)

joblib.dump(
    rf_search.best_estimator_,
    MODEL_DIR / "random_forest_tuned_candidate.joblib"
)


# --------------------------------------------------
# 11. TUNE LOGISTIC REGRESSION
# --------------------------------------------------

print("\nStarting Logistic Regression tuning...")

lr_search = RandomizedSearchCV(
    estimator=lr_pipeline,
    param_distributions=lr_parameters,
    n_iter=5,
    scoring="average_precision",
    cv=time_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

lr_search.fit(X_subtrain, y_subtrain)

print("\nLOGISTIC REGRESSION RESULTS")
print("Best parameters:", lr_search.best_params_)
print("Best CV Average Precision:", lr_search.best_score_)

joblib.dump(
    lr_search.best_estimator_,
    MODEL_DIR / "logistic_regression_tuned_candidate.joblib"
)


# --------------------------------------------------
# 12. SAVE TUNING COMPARISON
# --------------------------------------------------

results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Best_CV_PR_AUC": rf_search.best_score_,
        "Best_Parameters": str(rf_search.best_params_)
    },
    {
        "Model": "Logistic Regression",
        "Best_CV_PR_AUC": lr_search.best_score_,
        "Best_Parameters": str(lr_search.best_params_)
    }
])

output_file = REPORT_DIR / "hyperparameter_tuning_results.csv"

results.to_csv(output_file, index=False)

print("\nHYPERPARAMETER TUNING COMPARISON")
print(results.to_string(index=False))

print("\nResults saved at:", output_file)
print("\nTuned candidate pipelines saved successfully.")
