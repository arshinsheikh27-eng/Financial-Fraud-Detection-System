
import pandas as pd
from pathlib import Path

# File paths
input_path = Path("data/raw/financial_fraud_detection_dataset.csv")
output_path = Path("data/processed/cleaned_fraud_data.csv")

# Load original dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# 1. Remove exact duplicate rows
initial_rows = len(df)
df = df.drop_duplicates()
print("Exact duplicate rows removed:", initial_rows - len(df))

# 2. Remove unnecessary spaces from text columns
text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].str.strip()

# 3. Convert transaction date
df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

invalid_dates = df["Transaction_Date"].isna().sum()
print("Invalid dates:", invalid_dates)

# 4. Check required numerical columns
numeric_columns = [
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Check invalid numerical values
print("\nNegative transaction amounts:",
      (df["Transaction_Amount"] < 0).sum())

print("Negative average spending:",
      (df["Average_Spend"] < 0).sum())

print("Negative previous transactions:",
      (df["Previous_Transactions"] < 0).sum())

print("Negative account age:",
      (df["Account_Age_Days"] < 0).sum())

# 6. Check target labels
print("\nFraud target values:")
print(df["Fraudulent"].value_counts(dropna=False))

# 7. Check duplicate transaction IDs
print("\nDuplicate transaction IDs:",
      df["Transaction_ID"].duplicated().sum())

# 8. Final quality checks
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Do not silently delete records if unexpected problems occur
if df.isnull().any().any():
    raise ValueError("Missing values found. Investigate before proceeding.")

if invalid_dates > 0:
    raise ValueError("Invalid dates found. Investigate before proceeding.")

if not set(df["Fraudulent"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected fraud target labels found.")

if df["Transaction_ID"].duplicated().any():
    raise ValueError("Duplicate transaction IDs found.")

if (df[numeric_columns] < 0).any().any():
    raise ValueError("Negative numerical values found. Investigate first.")

# 9. Save cleaned dataset
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\n========== CLEANING SUMMARY ==========")
print("Final dataset shape:", df.shape)
print("Saved file:", output_path)
print("Cleaning completed successfully!")
