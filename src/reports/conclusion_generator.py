"""
Conclusion Generator for Churn Prediction Project
Generates a final report and conclusions based on the model results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
import logging
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load the project configuration."""
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_best_model():
    """Load the best model from disk."""
    # Get list of models in the models directory
    model_files = [f for f in os.listdir('models') if f.startswith('tuned_')]
    
    if not model_files:
        logger.warning("No tuned models found")
        return None
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join('models', latest_model)
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded best model from {model_path}")
        return model, latest_model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def generate_conclusion(metrics: Dict[str, float]) -> str:
    """
    Generate a comprehensive conclusion based on model metrics.
    
    Args:
        metrics: Dictionary of model performance metrics
        
    Returns:
        Conclusion text as string
    """
    config = load_config()
    best_model, model_name = load_best_model()
    
    if not metrics:
        return "Unable to generate conclusion due to missing metrics."
    
    # Format metrics for display
    metrics_text = "\n".join([f"- {k.upper()}: {v:.4f}" for k, v in metrics.items()])
    
    # Create model analysis
    model_type = model_name.replace("tuned_", "").replace(".joblib", "") if model_name else "Unknown"
    
    # Performance analysis
    if 'roc_auc' in metrics:
        if metrics['roc_auc'] > 0.9:
            performance = "excellent"
        elif metrics['roc_auc'] > 0.8:
            performance = "very good"
        elif metrics['roc_auc'] > 0.7:
            performance = "good"
        elif metrics['roc_auc'] > 0.6:
            performance = "fair"
        else:
            performance = "poor"
    else:
        performance = "undefined"
    
    # Load feature importance if available
    if best_model and hasattr(best_model, 'feature_importances_'):
        # Get feature names from interim data
        try:
            interim_path = config['data']['interim_path']
            base_path = os.path.splitext(interim_path)[0]
            X_test = pd.read_parquet(f"{base_path}_X_test.parquet")
            feature_names = X_test.columns
            
            # Get top features
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [feature_names[i] for i in indices[:5]]
            
            feature_importance_text = "Top 5 most important features:\n" + "\n".join([
                f"- {i+1}. {feature}: {importances[indices[i]]:.4f}" 
                for i, feature in enumerate(top_features)
            ])
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            feature_importance_text = "Feature importance information unavailable."
    else:
        feature_importance_text = "Feature importance information unavailable for this model type."
    
    # Generate conclusion text
    conclusion = f"""
CHURN PREDICTION PROJECT - CONCLUSION REPORT
===========================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL SUMMARY
------------
Best performing model: {model_type}
Overall performance assessment: {performance.upper()}

PERFORMANCE METRICS
------------------
{metrics_text}

KEY INSIGHTS
-----------
{feature_importance_text}

BUSINESS IMPLICATIONS
-------------------
Based on the model performance (ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}), we are able to identify 
potential churners with {performance} accuracy. With a precision of {metrics.get('precision', 'N/A'):.4f}, 
the model gives us confidence that customers identified as likely to churn are indeed high-risk cases.

The recall of {metrics.get('recall', 'N/A'):.4f} indicates that we are able to identify approximately 
{metrics.get('recall', 0)*100:.1f}% of all actual churners, allowing for targeted retention campaigns.

RECOMMENDATIONS
--------------
1. Deploy the model as part of a customer relationship management system
2. Implement targeted interventions for customers identified as high churn risks
3. Focus business strategies on addressing the factors identified in the top features
4. Continue to monitor model performance and update as new data becomes available
5. Consider A/B testing different retention strategies based on model predictions

MODEL DEPLOYMENT CONSIDERATIONS
-----------------------------
- Ensure the model is regularly retrained with fresh data
- Implement a monitoring system to track model drift
- Create an explainability layer for business users to understand predictions
- Develop a simple API for integration with existing systems
"""
    
    # Save the conclusion to a report file
    os.makedirs('reports', exist_ok=True)
    with open('reports/conclusion_report.txt', 'w') as f:
        f.write(conclusion)
    
    logger.info("Saved conclusion to reports/conclusion_report.txt")
    
    return conclusion

def main():
    """Main function for testing the conclusion generator."""
    # Sample metrics for testing
    sample_metrics = {
        'accuracy': 0.85,
        'precision': 0.78,
        'recall': 0.72,
        'f1': 0.75,
        'roc_auc': 0.88,
        'pr_auc': 0.82
    }
    
    conclusion = generate_conclusion(sample_metrics)
    print(conclusion)

if __name__ == "__main__":
    main() 