
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/financial_fraud_detection_dataset.csv")

print("\n========== DATE VALIDATION ==========")

# Compare both possible date interpretations
dates_dayfirst = pd.to_datetime(
    df["Transaction_Date"], dayfirst=True, errors="coerce"
)

dates_monthfirst = pd.to_datetime(
    df["Transaction_Date"], dayfirst=False, errors="coerce"
)

print("Invalid dates (day-first):", dates_dayfirst.isna().sum())
print("Invalid dates (month-first):", dates_monthfirst.isna().sum())

print("\nFirst 10 original dates:")
print(df["Transaction_Date"].head(10).to_list())

print("\nDay-first date range:")
print(dates_dayfirst.min(), "to", dates_dayfirst.max())

print("\nMonth-first date range:")
print(dates_monthfirst.min(), "to", dates_monthfirst.max())

print("\n========== NUMERICAL VALIDATION ==========")

numeric_columns = [
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days"
]

print(df[numeric_columns].describe().T)

print("\n========== CATEGORICAL VALUES ==========")

categorical_columns = [
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "Suspicious_Keyword"
]

for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\n========== FRAUD RATE BY FEATURE ==========")

for col in ["Is_International", "Suspicious_Keyword"]:
    print(f"\nFraud rate by {col}:")
    print(
        df.groupby(col)["Fraudulent"]
        .agg(["count", "mean"])
        .assign(fraud_percentage=lambda x: x["mean"] * 100)
    )

print("\n========== CUSTOMER AND ID CHECK ==========")

print("Duplicate Transaction IDs:", df["Transaction_ID"].duplicated().sum())
print("Unique Customers:", df["Customer_ID"].nunique())

print("\nValidation completed!")

