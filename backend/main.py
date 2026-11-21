from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from backend.service import predict_transaction, get_history, init_db

app = FastAPI(title='Financial Fraud Detection API', version='1.0.0',
              description='Fraud-risk scoring API. Predictions are risk indicators, not confirmed fraud findings.')

class Transaction(BaseModel):
    Transaction_Date: str
    Transaction_Amount: float = Field(gt=0)
    Merchant_Category: str
    Payment_Method: str
    Device_Type: str
    Location: str
    Is_International: Literal[0,1]
    Previous_Transactions: int = Field(ge=0)
    Average_Spend: float = Field(gt=0)
    Account_Age_Days: int = Field(ge=0)

@app.on_event('startup')
def startup(): init_db()

@app.get('/')
def root(): return {'message':'Financial Fraud Detection API is running','docs':'/docs'}

@app.post('/predict')
def predict(tx: Transaction):
    try: return predict_transaction(tx.model_dump())
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

@app.get('/history')
def history(limit: int = 500):
    try: return get_history(limit).to_dict(orient='records')
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
