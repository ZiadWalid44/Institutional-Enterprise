from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import io
app = FastAPI(
    title="FraudShield AI - Inference API",
    description="Intelligent Real-time and Batch Fraud Risk Assessment Engine",
    version="1.0.0"
)

MODEL_PATH = "fraud_model_xgboost.pkl"
try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {str(e)}")

MODEL_VERSION = "xgboost_v1.0"
DECISION_THRESHOLD = 0.30  
def get_risk_band(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    elif probability <= 0.70:
        return "Medium"
    else:
        return "High"
class TransactionPayload(BaseModel):
    step: int = Field(..., example=1, description="Simulation step (1 step = 1 hour)")
    type: str = Field(..., example="TRANSFER", description="TRANSFER or CASH_OUT")
    amount: float = Field(..., gt=0, example=181.0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, example=181.0, description="Initial balance of origin")
    newbalanceOrig: float = Field(..., ge=0, example=0.0, description="New balance of origin")
    oldbalanceDest: float = Field(..., ge=0, example=0.0, description="Initial balance of destination")
    newbalanceDest: float = Field(..., ge=0, example=0.0, description="New balance of destination")
@app.post("/predict", tags=["Inference"])
def predict_single_transaction(payload: TransactionPayload):
    input_data = pd.DataFrame([payload.model_dump()])
    
    prob = float(model_pipeline.predict_proba(input_data)[:, 1][0])
    prediction = int(prob >= DECISION_THRESHOLD)
    risk_band = get_risk_band(prob)
    
    return {
        "status": "success",
        "model_version": MODEL_VERSION,
        "fraud_probability": round(prob, 4),
        "prediction": prediction,
        "is_fraud": bool(prediction),
        "risk_band": risk_band,
        "threshold_used": DECISION_THRESHOLD
    }
@app.post("/predict-batch", tags=["Inference"])
async def predict_batch_transactions(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    contents = await file.read()
    try:
        batch_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")
    
    required_cols = [
        "step", "type", "amount", "oldbalanceOrg", 
        "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    ]
    missing_cols = [col for col in required_cols if col not in batch_df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422, 
            detail=f"Missing required columns in CSV: {missing_cols}"
        )
    
    probabilities = model_pipeline.predict_proba(batch_df[required_cols])[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    
    batch_df["fraud_probability"] = [round(float(p), 4) for p in probabilities]
    batch_df["prediction"] = predictions
    batch_df["risk_band"] = [get_risk_band(p) for p in probabilities]
    
    summary = {
        "total_records": len(batch_df),
        "high_risk_count": int((batch_df["risk_band"] == "High").sum()),
        "medium_risk_count": int((batch_df["risk_band"] == "Medium").sum()),
        "low_risk_count": int((batch_df["risk_band"] == "Low").sum()),
        "flagged_fraud_cases": int((predictions == 1).sum())
    }
    
    return {
        "status": "success",
        "model_version": MODEL_VERSION,
        "summary": summary,
        "results": batch_df[["fraud_probability", "prediction", "risk_band"]].to_dict(orient="records")
    }