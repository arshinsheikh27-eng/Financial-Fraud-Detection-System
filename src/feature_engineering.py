
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "processed" / "cleaned_fraud_data.csv"
output_file = BASE_DIR / "data" / "processed" / "feature_engineered_fraud_data.csv"

# Load cleaned dataset
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# Convert transaction date
df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%Y-%m-%d %H:%M:%S"
)

# Extract date and time features
df["Transaction_Hour"] = df["Transaction_Date"].dt.hour

df["Transaction_Day"] = df["Transaction_Date"].dt.day

df["Transaction_Month"] = df["Transaction_Date"].dt.month

df["Day_of_Week"] = df["Transaction_Date"].dt.dayofweek

# 1 = Weekend, 0 = Weekday
df["Is_Weekend"] = (
    df["Day_of_Week"] >= 5
).astype(int)

# Compare transaction amount with average spending
df["Amount_to_Average_Ratio"] = (
    df["Transaction_Amount"] /
    df["Average_Spend"].replace(0, pd.NA)
)

df["Amount_Difference"] = (
    df["Transaction_Amount"] - df["Average_Spend"]
)

# Save feature-engineered dataset
df.to_csv(output_file, index=False)

print("\nFeature engineering completed!")

print("New dataset shape:", df.shape)

print("\nNew features:")
print([
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Day_of_Week",
    "Is_Weekend",
    "Amount_to_Average_Ratio",
    "Amount_Difference"
])

print("\nSaved file:", output_file)

