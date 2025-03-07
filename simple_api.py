"""
Simple FastAPI service for churn predictions
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import yaml
import numpy as np
import pandas as pd
import uvicorn

# Load configuration
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create the FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="A simple API for predicting customer churn",
    version="1.0.0"
)

# Load the model
model_path = os.path.join(config['models']['save_path'], 'random_forest_model.joblib')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Define the input schema (this is a simplified example)
class CustomerData(BaseModel):
    # These are example fields - replace with your actual model's features
    tenure_months: int
    contract_type: str
    monthly_charges: float
    has_fiber_optic: bool

# Define the output schema
class ChurnPrediction(BaseModel):
    churn_probability: float
    churn_risk: str
    confidence: str

@app.get("/")
async def root():
    return {"message": "Churn Prediction API"}

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Service is healthy"}

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # NOTE: This is a simplified example prediction
    # In a real implementation, you would need to:
    # 1. Preprocess the input data exactly as during training
    # 2. Ensure all required features are present
    # 3. Handle categorical encoding the same way
    
    # For demonstration purposes, we're just returning a random prediction
    import random
    probability = random.random()
    
    # Determine risk level
    if probability < 0.3:
        risk = "Low"
        confidence = "High"
    elif probability < 0.7:
        risk = "Medium"
        confidence = "Medium"
    else:
        risk = "High"
        confidence = "High"
    
    return ChurnPrediction(
        churn_probability=probability,
        churn_risk=risk,
        confidence=confidence
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
