"""
Advanced Churn Prediction API with versioning, rate limiting, and monitoring
"""

import os
import time
import logging
import datetime
import joblib
import yaml
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import json
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize metrics
REQUEST_COUNT = Counter('churn_api_requests_total', 'Total number of requests received', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('churn_api_request_latency_seconds', 'Request latency in seconds')
PREDICTION_COUNT = Counter('churn_api_predictions_total', 'Total number of predictions', ['prediction_class'])

# Initialize API key security (for demonstration - in production, use a more robust approach)
API_KEY = os.getenv("API_KEY", "test_api_key")
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

# Create rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_log = {}
    
    def check_rate_limit(self, client_ip: str) -> bool:
        current_time = time.time()
        
        # Clean old requests
        self.request_log = {ip: times for ip, times in self.request_log.items() 
                           if any(t > current_time - 60 for t in times)}
        
        # Get requests for this IP
        if client_ip not in self.request_log:
            self.request_log[client_ip] = []
        
        ip_requests = self.request_log[client_ip]
        
        # Remove requests older than 1 minute
        ip_requests = [t for t in ip_requests if t > current_time - 60]
        
        # Check if rate limit exceeded
        if len(ip_requests) >= self.requests_per_minute:
            return False
        
        # Add current request
        ip_requests.append(current_time)
        self.request_log[client_ip] = ip_requests
        
        return True

rate_limiter = RateLimiter()

# Load available models
def get_available_models():
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('tuned_') and f.endswith('.joblib')]
    
    models = {}
    for model_file in model_files:
        # Extract version info from filename
        model_name = model_file.replace('tuned_', '').replace('.joblib', '')
        
        # Check if metadata exists
        metadata_path = os.path.join('models/metadata', f'retraining_{model_name}.yml')
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
        
        models[model_name] = {
            'file': model_file,
            'path': os.path.join(model_dir, model_file),
            'metadata': metadata
        }
    
    return models

# Get the latest model
def get_latest_model():
    models = get_available_models()
    if not models:
        raise HTTPException(status_code=500, detail="No models available")
    
    latest_model_name = sorted(models.keys())[-1]
    return models[latest_model_name]

# Model cache to avoid reloading
model_cache = {}

# Load a specific model
def load_model(model_name=None):
    models = get_available_models()
    
    if model_name is not None:
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        model_info = models[model_name]
    else:
        # Use latest model
        latest_model_name = sorted(models.keys())[-1]
        model_info = models[latest_model_name]
    
    # Check if model is in cache
    if model_info['file'] in model_cache:
        return model_cache[model_info['file']]
    
    # Load the model
    try:
        model = joblib.load(model_info['path'])
        model_cache[model_info['file']] = model
        logger.info(f"Loaded model from {model_info['path']}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Load feature transformers if available
def load_transformers():
    transformer_path = os.path.join('models', 'transformers.joblib')
    if os.path.exists(transformer_path):
        try:
            transformers = joblib.load(transformer_path)
            logger.info(f"Loaded transformers from {transformer_path}")
            return transformers
        except Exception as e:
            logger.warning(f"Error loading transformers: {e}")
    
    return None

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Advanced API for predicting customer churn with versioning and monitoring",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Get client IP for rate limiting
    client_ip = request.client.host
    
    # Check rate limit
    if not rate_limiter.check_rate_limit(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again later."}
        )
    
    # Process the request
    response = await call_next(request)
    
    # Record metrics
    request_latency = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method, 
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.observe(request_latency)
    
    return response

# Define API models
class CustomerFeatures(BaseModel):
    """Pydantic model for customer data input"""
    # This should be customized based on your specific feature set
    model_version: Optional[str] = Field(None, description="Model version to use (default: latest)")
    features: Dict[str, Any] = Field(..., description="Customer features as key-value pairs")
    
    # Validators
    @validator('features')
    def validate_features(cls, features):
        if not features:
            raise ValueError("Features dictionary cannot be empty")
        return features

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""
    model_version: str
    churn_probability: float
    churn_prediction: bool
    confidence: float
    explanation: Dict[str, float]
    prediction_id: str
    timestamp: str

# API endpoints
@app.get("/", dependencies=[Depends(get_api_key)])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Churn Prediction API", 
        "status": "active",
        "versions": list(get_available_models().keys())
    }

@app.get("/models", dependencies=[Depends(get_api_key)])
async def list_models():
    """List available model versions"""
    models = get_available_models()
    
    model_info = {}
    for name, info in models.items():
        model_details = {
            "file": info["file"]
        }
        
        if info["metadata"]:
            model_details["trained_at"] = info["metadata"].get("timestamp")
            model_details["metrics"] = info["metadata"].get("metrics")
        
        model_info[name] = model_details
    
    return {
        "models": model_info,
        "latest": sorted(models.keys())[-1] if models else None
    }

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn for a single customer
    """
    # Load specified model or latest
    try:
        model = load_model(customer.model_version)
        model_version = customer.model_version or sorted(get_available_models().keys())[-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Convert input to DataFrame
    try:
        input_data = pd.DataFrame([customer.features])
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid feature format: {str(e)}"
        )
    
    # Make prediction
    try:
        prediction_proba = model.predict_proba(input_data)[0, 1]
        prediction = bool(prediction_proba >= 0.5)
        
        # Record prediction in metrics
        PREDICTION_COUNT.labels(
            prediction_class="churn" if prediction else "no_churn"
        ).inc()
        
        # Calculate confidence
        confidence = max(prediction_proba, 1 - prediction_proba)
        
        # Generate unique prediction ID
        prediction_id = f"pred_{int(time.time())}_{hash(str(customer.features)) % 10000}"
        
        # Generate explanation
        explanation = {}
        if hasattr(model, 'feature_importances_'):
            features = input_data.columns
            importances = model.feature_importances_
            feature_importances = {features[i]: float(importances[i]) for i in range(len(features))}
            
            # Get feature values
            feature_values = {col: float(input_data[col].values[0]) 
                             if isinstance(input_data[col].values[0], (int, float)) 
                             else str(input_data[col].values[0]) 
                             for col in input_data.columns}
            
            # Sort by importance
            sorted_features = sorted(
                feature_importances.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            # Take top 10 features
            explanation = {
                f: {
                    "importance": feature_importances[f],
                    "value": feature_values[f]
                }
                for f, _ in sorted_features[:10]
            }
        
        # Log prediction
        prediction_log = {
            "prediction_id": prediction_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_version": model_version,
            "features": customer.features,
            "prediction": prediction,
            "probability": float(prediction_proba),
            "confidence": float(confidence)
        }
        
        # Save to prediction log file
        os.makedirs('logs/predictions', exist_ok=True)
        with open(f'logs/predictions/{prediction_id}.json', 'w') as f:
            json.dump(prediction_log, f)
        
        return PredictionResponse(
            model_version=model_version,
            churn_probability=float(prediction_proba),
            churn_prediction=prediction,
            confidence=float(confidence),
            explanation=explanation,
            prediction_id=prediction_id,
            timestamp=datetime.datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Load the latest model to verify it works
    try:
        model = load_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "model_loaded": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advanced_app:app", host="0.0.0.0", port=8000, reload=True) 