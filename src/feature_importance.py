
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import permutation_importance


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

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "random_forest_corrected_validation_pipeline.joblib"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
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
    by="Transaction_Date"
).reset_index(drop=True)


# --------------------------------------------------
# 3. PREPARE FEATURES AND TARGET
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
# 4. RECREATE THE VALIDATION SPLIT
# --------------------------------------------------

train_split_index = int(len(X) * 0.80)

X_train = X.iloc[:train_split_index]
y_train = y.iloc[:train_split_index]

validation_split_index = int(len(X_train) * 0.80)

X_validation = X_train.iloc[validation_split_index:]
y_validation = y_train.iloc[validation_split_index:]

print("Validation rows:", len(X_validation))


# --------------------------------------------------
# 5. LOAD TRAINED RANDOM FOREST PIPELINE
# --------------------------------------------------

model = joblib.load(MODEL_FILE)

print("\nRandom Forest pipeline loaded successfully.")


# --------------------------------------------------
# 6. CALCULATE PERMUTATION IMPORTANCE
# --------------------------------------------------

print("\nCalculating feature importance...")

importance = permutation_importance(
    model,
    X_validation,
    y_validation,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 7. CREATE IMPORTANCE REPORT
# --------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": X_validation.columns,
    "Importance_Mean": importance.importances_mean,
    "Importance_Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance_Mean",
    ascending=False
).reset_index(drop=True)

output_file = REPORT_DIR / "feature_importance.csv"

importance_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\nFEATURE IMPORTANCE RESULTS")
print(importance_df.to_string(index=False))

print("\nReport saved at:", output_file)


# --------------------------------------------------
# 9. CREATE FEATURE IMPORTANCE GRAPH
# --------------------------------------------------

plot_df = importance_df.head(12).sort_values(
    by="Importance_Mean",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_df["Feature"],
    plot_df["Importance_Mean"],
    xerr=plot_df["Importance_Std"],
    capsize=3
)

plt.xlabel("Decrease in Average Precision")
plt.ylabel("Features")
plt.title("Random Forest - Permutation Feature Importance")
plt.tight_layout()

graph_file = REPORT_DIR / "feature_importance.png"

plt.savefig(
    graph_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGraph saved at:", graph_file)
