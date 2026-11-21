# Financial Fraud Detection — VS Code project

## Dashboard redesign
The Streamlit overview has been redesigned to follow the supplied dark fraud-analytics reference: dark navy workspace, compact left navigation, KPI cards, fraud-distribution donut, system integration status, hourly fraud trend, and model-comparison chart. The existing prediction, batch prediction, investigation history, and model/feature-analysis pages remain available.

## Run on Windows
Open this project folder in VS Code Terminal (the folder containing `dashboard`, `backend`, `models`, and `requirements-app.txt`), then run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-app.txt
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

If you already created a virtual environment, activate it and run the install and launch commands only. Do not add a period after `dashboard/app.py`.

## Optional FastAPI backend
Open a second terminal in the same project folder, activate the same `.venv`, and run:

```powershell
python -m uvicorn backend.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`.

## Data and interpretation notes
- Overview feature charts use the two aggregate CSV files in `reports/feature_analysis/`.
- The international and domestic rows in the supplied CSV sum to 4,000 transactions. These are source-sample aggregates, not live predictions.
- Live KPI counters and Investigation History are based only on predictions made through this app and stored in local SQLite.
- Batch prediction requires a transaction-level CSV with these columns: `Transaction_Date`, `Transaction_Amount`, `Merchant_Category`, `Payment_Method`, `Device_Type`, `Location`, `Is_International`, `Previous_Transactions`, `Average_Spend`, `Account_Age_Days`.
- Model output is a risk estimate. A flagged transaction requires human investigation and is not proof of fraud.

## Model compatibility
The saved joblib model is version-sensitive. The supplied requirements pin scikit-learn to 1.7.2 and Python 3.11 is recommended. If model loading fails, verify that the correct virtual environment is active and that `pip show scikit-learn` reports 1.7.2.
