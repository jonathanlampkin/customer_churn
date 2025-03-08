"""
Generate a conclusion report for the churn prediction project
"""
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load configuration
import yaml
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

def generate_conclusion():
    """Generate a conclusion report based on model metrics"""
    
    # Load metrics
    metrics_path = config['reports']['metrics_path']
    if not os.path.exists(metrics_path):
        logger.error(f"Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Generate conclusion text
    conclusion = f"""
CHURN PREDICTION PROJECT - CONCLUSION REPORT
===========================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL SUMMARY
------------
Best performing model: {metrics.get('model_type', 'Unknown')}
Overall performance assessment: {"GOOD" if metrics.get('roc_auc', 0) > 0.75 else "FAIR"}

PERFORMANCE METRICS
------------------
- Accuracy: {metrics.get('accuracy', 0):.4f}
- ROC-AUC: {metrics.get('roc_auc', 0):.4f}

KEY INSIGHTS
-----------
Based on feature importance analysis, the top predictors of churn are:
"""

    # Add top features
    if 'feature_importance' in metrics:
        importance = metrics['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            conclusion += f"{i}. {feature}: {importance:.4f}\n"
    
    conclusion += """
BUSINESS IMPLICATIONS
-------------------
The model provides good predictive capability for identifying customers at risk of churning.
This allows for targeted retention campaigns and proactive customer service.

RECOMMENDATIONS
--------------
1. Deploy the model as part of a customer relationship management system
2. Implement targeted interventions for customers identified as high churn risks
3. Focus business strategies on addressing the factors identified in the top features
4. Continue to monitor model performance and update as new data becomes available
5. Consider A/B testing different retention strategies based on model predictions
"""

    # Save the conclusion
    report_path = os.path.join(
        os.path.dirname(config['reports']['metrics_path']),
        'conclusion_report.txt'
    )
    
    with open(report_path, 'w') as f:
        f.write(conclusion)
    
    logger.info(f"Conclusion report saved to {report_path}")
    print(conclusion)  # Also print to console

if __name__ == "__main__":
    generate_conclusion()
