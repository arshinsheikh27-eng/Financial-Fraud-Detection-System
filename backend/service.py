from pathlib import Path
import sqlite3
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'fraud_predictions.db'
MODEL_PATH = ROOT / 'models' / 'random_forest_tuned_candidate.joblib'
THRESHOLD = 0.19

FEATURES = [
    'Transaction_Amount','Is_International','Previous_Transactions','Average_Spend',
    'Account_Age_Days','Transaction_Hour','Transaction_Day','Transaction_Month',
    'Day_of_Week','Is_Weekend','Amount_to_Average_Ratio','Amount_Difference',
    'Merchant_Category','Payment_Method','Device_Type','Location'
]

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')
        _model = joblib.load(MODEL_PATH)
    return _model

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            transaction_date TEXT, amount REAL, merchant_category TEXT, payment_method TEXT,
            device_type TEXT, location TEXT, is_international INTEGER,
            fraud_probability REAL, prediction TEXT, suggested_action TEXT)''')

def prepare_features(transaction):
    df = pd.DataFrame([transaction])
    dt = pd.to_datetime(df['Transaction_Date'], errors='raise')
    df['Transaction_Hour'] = dt.dt.hour
    df['Transaction_Day'] = dt.dt.day
    df['Transaction_Month'] = dt.dt.month
    df['Day_of_Week'] = dt.dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    avg = pd.to_numeric(df['Average_Spend'], errors='raise').replace(0, float('nan'))
    amount = pd.to_numeric(df['Transaction_Amount'], errors='raise')
    df['Amount_to_Average_Ratio'] = (amount / avg).fillna(0)
    df['Amount_Difference'] = amount - avg
    return df[FEATURES]

def predict_transaction(transaction, save=True):
    required = ['Transaction_Date','Transaction_Amount','Merchant_Category','Payment_Method',
                'Device_Type','Location','Is_International','Previous_Transactions',
                'Average_Spend','Account_Age_Days']
    missing = [x for x in required if x not in transaction or transaction[x] in (None, '')]
    if missing:
        raise ValueError('Missing required fields: ' + ', '.join(missing))
    X = prepare_features(transaction)
    probability = float(get_model().predict_proba(X)[0][1])
    label = 'Fraudulent' if probability >= THRESHOLD else 'Genuine'
    result = {'Fraud_Probability': round(probability*100, 2), 'Prediction': label,
              'Suggested_Action': 'Flag for Review' if label == 'Fraudulent' else 'Proceed',
              'Threshold': THRESHOLD}
    if save:
        init_db()
        with sqlite3.connect(DB_PATH) as con:
            con.execute('''INSERT INTO predictions(transaction_date,amount,merchant_category,payment_method,
                device_type,location,is_international,fraud_probability,prediction,suggested_action)
                VALUES(?,?,?,?,?,?,?,?,?,?)''', (str(transaction['Transaction_Date']), float(transaction['Transaction_Amount']),
                str(transaction['Merchant_Category']), str(transaction['Payment_Method']), str(transaction['Device_Type']),
                str(transaction['Location']), int(transaction['Is_International']), result['Fraud_Probability'],
                label, result['Suggested_Action']))
    return result

def get_history(limit=500):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', con, params=(int(limit),))
