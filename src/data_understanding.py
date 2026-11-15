
import pandas as pd
from pathlib import Path

# 1. Define dataset path
DATA_PATH = Path("data/raw/financial_fraud_detection_dataset.csv")

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

# 3. Dataset dimensions
print("\nDataset Shape:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# 4. Display first five records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# 5. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 6. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 7. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 8. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate Rows:", df.duplicated().sum())

# 9. Unique values
print("\n========== UNIQUE VALUES ==========")
print(df.nunique())

# 10. Fraud distribution
print("\n========== FRAUD DISTRIBUTION ==========")
print(df["Fraudulent"].value_counts())

print("\nFraud Percentage:")
print((df["Fraudulent"].value_counts(normalize=True) * 100).round(2))

# 11. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all").T)

# 12. Save report
Path("reports").mkdir(exist_ok=True)

with open("reports/data_quality_report.txt", "w", encoding="utf-8") as file:
    file.write("DATASET QUALITY REPORT\n")
    file.write("=" * 40 + "\n")
    file.write(f"Dataset Shape: {df.shape}\n\n")
    file.write("Missing Values:\n")
    file.write(df.isnull().sum().to_string())
    file.write("\n\nDuplicate Rows: ")
    file.write(str(df.duplicated().sum()))
    file.write("\n\nFraud Distribution:\n")
    file.write(df["Fraudulent"].value_counts().to_string())
    file.write("\n\nData Types:\n")
    file.write(df.dtypes.to_string())

print("\nReport saved successfully in reports/data_quality_report.txt")
print("\nDataset inspection completed!")
