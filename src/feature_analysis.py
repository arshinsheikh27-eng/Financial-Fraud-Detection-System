
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_engineered_fraud_data.csv"
)

REPORT_DIR = BASE_DIR / "reports" / "feature_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

if df["Transaction_Date"].isna().any():
    raise ValueError("Invalid transaction dates found.")

# Sort chronologically
df = df.sort_values("Transaction_Date").reset_index(drop=True)

# Use only the training period; preserve final test period
train_size = int(len(df) * 0.80)
df = df.iloc[:train_size].copy()

print("Training-period rows:", len(df))


# --------------------------------------------------
# 1. FRAUD RATE BY TRANSACTION HOUR
# --------------------------------------------------

hour_analysis = (
    df.groupby("Transaction_Hour")
    .agg(
        Total_Transactions=("Fraudulent", "count"),
        Fraud_Transactions=("Fraudulent", "sum"),
        Fraud_Rate=("Fraudulent", "mean")
    )
    .reset_index()
)

hour_analysis["Fraud_Rate_Percent"] = (
    hour_analysis["Fraud_Rate"] * 100
).round(2)

hour_analysis.to_csv(
    REPORT_DIR / "fraud_rate_by_hour.csv",
    index=False
)

print("\nFRAUD RATE BY TRANSACTION HOUR")
print(
    hour_analysis[
        [
            "Transaction_Hour",
            "Total_Transactions",
            "Fraud_Transactions",
            "Fraud_Rate_Percent"
        ]
    ].to_string(index=False)
)

plt.figure(figsize=(12, 6))

plt.bar(
    hour_analysis["Transaction_Hour"],
    hour_analysis["Fraud_Rate_Percent"]
)

plt.xlabel("Transaction Hour")
plt.ylabel("Fraud Rate (%)")
plt.title("Fraud Rate by Transaction Hour")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraud_rate_by_hour.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. FRAUD RATE BY INTERNATIONAL STATUS
# --------------------------------------------------

international_analysis = (
    df.groupby("Is_International")
    .agg(
        Total_Transactions=("Fraudulent", "count"),
        Fraud_Transactions=("Fraudulent", "sum"),
        Fraud_Rate=("Fraudulent", "mean")
    )
    .reset_index()
)

international_analysis["Transaction_Type"] = (
    international_analysis["Is_International"]
    .map({0: "Domestic", 1: "International"})
)

international_analysis["Fraud_Rate_Percent"] = (
    international_analysis["Fraud_Rate"] * 100
).round(2)

international_analysis.to_csv(
    REPORT_DIR / "fraud_rate_by_international.csv",
    index=False
)

print("\nFRAUD RATE BY TRANSACTION TYPE")
print(
    international_analysis[
        [
            "Transaction_Type",
            "Total_Transactions",
            "Fraud_Transactions",
            "Fraud_Rate_Percent"
        ]
    ].to_string(index=False)
)

plt.figure(figsize=(8, 5))

plt.bar(
    international_analysis["Transaction_Type"],
    international_analysis["Fraud_Rate_Percent"]
)

plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.title("Fraud Rate: Domestic vs International")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraud_rate_by_international.png",
    dpi=300
)

plt.close()

print("\nFeature analysis completed successfully.")
print("Reports saved in:", REPORT_DIR)
