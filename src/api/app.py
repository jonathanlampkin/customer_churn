"""
FastAPI application for deploying the churn prediction model
"""

import os
import joblib
import yaml
import pandas as pd
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

# Load the best model
model_files = [f for f in os.listdir('models') if f.startswith('tuned_')]
if model_files:
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join('models', latest_model)
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
else:
    logger.error("No model found in models directory")
    model = None

# Load feature transformers
transformer_path = os.path.join('models', 'transformers.joblib')
if os.path.exists(transformer_path):
    transformers = joblib.load(transformer_path)
    logger.info(f"Loaded transformers from {transformer_path}")
else:
    logger.warning("No transformers found")
    transformers = None

class CustomerFeatures(BaseModel):
    """Pydantic model for customer data input"""
    # Define fields based on your feature set
    # For example:
    account_length: float = Field(..., description="Length of account in months")
    international_plan: str = Field(..., description="International plan (yes/no)")
    # ... add more fields based on your dataset

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""
    churn_probability: float
    churn_prediction: bool
    explanation: Dict[str, float]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {"message": "Churn Prediction API", "status": "active"}

@app.post("/predict/", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn for a single customer
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([customer.dict()])
    
    # Preprocess input (similar to the pipeline preprocessing)
    # This would apply the same transformations used during training
    
    # Make prediction
    churn_probability = model.predict_proba(input_data)[0, 1]
    churn_prediction = bool(churn_probability >= 0.5)
    
    # Generate explanation
    if hasattr(model, 'feature_importances_'):
        features = input_data.columns
        importances = model.feature_importances_
        explanation = {features[i]: float(importances[i]) for i in range(len(features))}
    else:
        explanation = {}
    
    return PredictionResponse(
        churn_probability=float(churn_probability),
        churn_prediction=churn_prediction,
        explanation=explanation
    )

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 