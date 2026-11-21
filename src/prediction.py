
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" /
    "random_forest_tuned_candidate.joblib"
)

# Fraud classification threshold selected on validation data
THRESHOLD = 0.19


# --------------------------------------------------
# 2. LOAD TRAINED MODEL PIPELINE
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully!")


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

def prepare_features(transaction):

    df = pd.DataFrame([transaction])

    # Convert transaction date
    df["Transaction_Date"] = pd.to_datetime(
        df["Transaction_Date"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise"
    )

    # Extract date and time features
    df["Transaction_Hour"] = (
        df["Transaction_Date"].dt.hour
    )

    df["Transaction_Day"] = (
        df["Transaction_Date"].dt.day
    )

    df["Transaction_Month"] = (
        df["Transaction_Date"].dt.month
    )

    df["Day_of_Week"] = (
        df["Transaction_Date"].dt.dayofweek
    )

    df["Is_Weekend"] = (
        df["Day_of_Week"] >= 5
    ).astype(int)

    # Amount comparison features
    average_spend = pd.to_numeric(
        df["Average_Spend"], errors="raise"
    )

    transaction_amount = pd.to_numeric(
        df["Transaction_Amount"], errors="raise"
    )

    df["Amount_to_Average_Ratio"] = (
        transaction_amount /
        average_spend.replace(0, float("nan"))
    )

    df["Amount_Difference"] = (
        transaction_amount - average_spend
    )

    return df


# --------------------------------------------------
# 4. PREDICT TRANSACTION
# --------------------------------------------------

def predict_transaction(transaction):

    # Generate required features
    df = prepare_features(transaction)

    # Select exactly the model input columns
    features = [
        "Transaction_Amount",
        "Is_International",
        "Previous_Transactions",
        "Average_Spend",
        "Account_Age_Days",
        "Transaction_Hour",
        "Transaction_Day",
        "Transaction_Month",
        "Day_of_Week",
        "Is_Weekend",
        "Amount_to_Average_Ratio",
        "Amount_Difference",
        "Merchant_Category",
        "Payment_Method",
        "Device_Type",
        "Location"
    ]

    X = df[features]

    # Predict fraud probability
    fraud_probability = model.predict_proba(X)[0][1]

    # Apply selected threshold
    prediction = (
        1 if fraud_probability >= THRESHOLD else 0
    )

    # Prepare result
    result = {
        "Fraud_Probability": round(
            float(fraud_probability) * 100, 2
        ),
        "Prediction": (
            "Fraudulent" if prediction == 1 else "Genuine"
        ),
        "Suggested_Action": (
            "Flag for Review" if prediction == 1
            else "Proceed"
        )
    }

    return result


# --------------------------------------------------
# 5. TEST WITH SAMPLE TRANSACTION
# --------------------------------------------------

if __name__ == "__main__":

    sample_transaction = {
    "Transaction_Date": "2024-02-20 14:30:00",
    "Transaction_Amount": 50.00,
    "Merchant_Category": "Grocery",
    "Payment_Method": "UPI",
    "Device_Type": "Mobile",
    "Location": "Delhi",
    "Is_International": 0,
    "Previous_Transactions": 150,
    "Average_Spend": 100.00,
    "Account_Age_Days": 1500
    }

    result = predict_transaction(sample_transaction)

    print("\nTRANSACTION PREDICTION")
    print("-----------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
