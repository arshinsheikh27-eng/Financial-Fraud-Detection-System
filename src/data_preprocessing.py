
import pandas as pd
import numpy as np
import json
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "feature_engineered_fraud_data.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "model_ready"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load and verify dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

if df.shape != (5000, 21):
    raise ValueError(
        f"Unexpected dataset shape: {df.shape}. "
        "Please verify the input file."
    )

if df["Transaction_ID"].duplicated().any():
    raise ValueError("Duplicate Transaction_ID values found.")

if not set(df["Fraudulent"].unique()).issubset({0, 1}):
    raise ValueError("Invalid values found in Fraudulent target.")

# --------------------------------------------------
# 3. Validate transaction dates
# --------------------------------------------------

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

if df["Transaction_Date"].isna().any():
    raise ValueError("Invalid transaction dates found.")

# Sort chronologically: older transactions first
df = df.sort_values(
    "Transaction_Date"
).reset_index(drop=True)

# --------------------------------------------------
# 4. Separate target and input features
# --------------------------------------------------

TARGET = "Fraudulent"

# Columns excluded from model inputs
DROP_COLUMNS = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Suspicious_Keyword"
]

X = df.drop(columns=DROP_COLUMNS + [TARGET])
y = df[TARGET].astype(int)

# --------------------------------------------------
# 5. Identify numerical and categorical columns
# --------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

if len(numerical_columns) + len(categorical_columns) != X.shape[1]:
    raise ValueError("Some feature columns have unsupported data types.")

print("\nNumerical features:", numerical_columns)
print("\nCategorical features:", categorical_columns)

# --------------------------------------------------
# 6. Create preprocessing pipelines
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ],
    remainder="drop"
)

# --------------------------------------------------
# 7. Chronological 80:20 train-test split
# --------------------------------------------------

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTraining fraud percentage:",
      round(y_train.mean() * 100, 2), "%")

print("Testing fraud percentage:",
      round(y_test.mean() * 100, 2), "%")

# Check that both sets contain genuine and fraudulent cases
if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "One of the chronological splits contains only one class. "
        "Review the time-based split before proceeding."
    )

# --------------------------------------------------
# 8. Fit only on training data
# --------------------------------------------------

X_train_transformed = preprocessor.fit_transform(X_train)

# Transform test data using training-fitted preprocessing
X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_final = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)

X_test_final = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

# --------------------------------------------------
# 9. Final validation
# --------------------------------------------------

if X_train_final.isna().any().any():
    raise ValueError("Missing values remain in training data.")

if X_test_final.isna().any().any():
    raise ValueError("Missing values remain in testing data.")

if not np.isfinite(X_train_final.to_numpy()).all():
    raise ValueError("Non-finite values found in training data.")

if not np.isfinite(X_test_final.to_numpy()).all():
    raise ValueError("Non-finite values found in testing data.")

# --------------------------------------------------
# 10. Save separate outputs
# --------------------------------------------------

X_train_final.to_csv(
    OUTPUT_DIR / "X_train.csv", index=False
)

X_test_final.to_csv(
    OUTPUT_DIR / "X_test.csv", index=False
)

y_train.to_frame().to_csv(
    OUTPUT_DIR / "y_train.csv", index=False
)

y_test.to_frame().to_csv(
    OUTPUT_DIR / "y_test.csv", index=False
)

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessing_pipeline.joblib"
)

metadata = {
    "original_rows": len(df),
    "original_columns": len(df.columns),
    "input_features_before_encoding": X.shape[1],
    "numerical_columns": numerical_columns,
    "categorical_columns": categorical_columns,
    "encoded_feature_count": len(feature_names),
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "training_start": str(
        df["Transaction_Date"].iloc[0]
    ),
    "training_end": str(
        df["Transaction_Date"].iloc[split_index - 1]
    ),
    "testing_start": str(
        df["Transaction_Date"].iloc[split_index]
    ),
    "testing_end": str(
        df["Transaction_Date"].iloc[-1]
    )
}

with open(
    OUTPUT_DIR / "preprocessing_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)

print("\n--------------------------------")
print("PREPROCESSING COMPLETED")
print("--------------------------------")

print("Final training shape:", X_train_final.shape)
print("Final testing shape:", X_test_final.shape)

print("\nAll missing values handled.")
print("All transformed values validated.")
print("Preprocessing pipeline saved.")
print("Train and test files saved.")
print("\nOutput folder:", OUTPUT_DIR)
