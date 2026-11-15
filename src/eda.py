
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_fraud_data.csv")

# Convert date column
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])

# Create output folder
output_dir = Path("reports/eda")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("========== EDA SUMMARY ==========")

# 1. Fraud distribution
print("\nFraud distribution:")
print(df["Fraudulent"].value_counts())

print("\nFraud percentage:")
print((df["Fraudulent"].value_counts(normalize=True) * 100).round(2))

# 2. Fraud distribution chart
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Fraudulent")
plt.title("Genuine vs Fraud Transactions")
plt.xlabel("Fraud Label (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig(output_dir / "fraud_distribution.png", bbox_inches="tight")
plt.close()

# 3. Transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Transaction_Amount", hue="Fraudulent",
             bins=40, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig(output_dir / "transaction_amount.png", bbox_inches="tight")
plt.close()

# 4. Fraud rate by categorical features
categorical_features = [
    "Merchant_Category",
    "Payment_Method",
    "Device_Type",
    "Location",
    "Is_International",
    "Suspicious_Keyword"
]

for col in categorical_features:
    fraud_rates = (
        df.groupby(col)["Fraudulent"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print(f"\nFraud percentage by {col}:")
    print(fraud_rates.round(2))

    plt.figure(figsize=(9, 5))
    fraud_rates.plot(kind="bar")
    plt.title(f"Fraud Rate by {col}")
    plt.ylabel("Fraud Percentage")
    plt.xlabel(col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"fraud_by_{col}.png", bbox_inches="tight")
    plt.close()

# 5. Numerical feature comparison
numerical_features = [
    "Transaction_Amount",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days"
]

for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Fraudulent", y=col)
    plt.title(f"{col} by Fraud Label")
    plt.savefig(output_dir / f"{col}_boxplot.png", bbox_inches="tight")
    plt.close()

# 6. Monthly fraud trend
df["Month"] = df["Transaction_Date"].dt.to_period("M").astype(str)

monthly_fraud = (
    df.groupby("Month")["Fraudulent"]
    .agg(["sum", "count"])
)

monthly_fraud["Fraud_Rate"] = (
    monthly_fraud["sum"] / monthly_fraud["count"] * 100
)

print("\nMonthly fraud analysis:")
print(monthly_fraud.round(2))

plt.figure(figsize=(11, 5))
plt.plot(monthly_fraud.index, monthly_fraud["sum"], marker="o")
plt.title("Monthly Fraud Transactions")
plt.xlabel("Month")
plt.ylabel("Number of Fraud Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "monthly_fraud_trend.png", bbox_inches="tight")
plt.close()

# 7. Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", bbox_inches="tight")
plt.close()

print("\nEDA completed successfully!")
print("All graphs saved in:", output_dir)
