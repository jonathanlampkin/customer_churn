"""
Simplified main script for Churn Prediction
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger()

# Load configuration
import yaml
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

def main():
    """Run a simplified pipeline"""
    logger.info("Starting simplified churn prediction pipeline")
    
    # Make sure output directories exist
    os.makedirs(os.path.dirname(config['data']['interim_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['data']['processed_path']), exist_ok=True)
    os.makedirs(config['models']['save_path'], exist_ok=True)
    os.makedirs(config['reports']['figures_path'], exist_ok=True)
    
    # 1. Load data
    logger.info(f"Loading data from {config['data']['raw_path']}")
    df = pd.read_csv(config['data']['raw_path'])
    logger.info(f"Data loaded: {df.shape}")
    
    # 2. Preprocess - just handle missing values
    logger.info("Preprocessing data")
    # Fill numeric columns with mean
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    # Save interim data
    df.to_parquet(config['data']['interim_path'])
    logger.info(f"Interim data saved to {config['data']['interim_path']}")
    
    # 3. Prepare for modeling
    target_col = config['training']['target_column']
    
    # Make sure target column exists
    if target_col not in df.columns:
        # Look for a column containing 'churn'
        potential_targets = [col for col in df.columns if 'churn' in col.lower()]
        if potential_targets:
            target_col = potential_targets[0]
            logger.info(f"Using '{target_col}' as target column")
        else:
            logger.error("Target column not found")
            return
    
    # 4. Simple feature preparation
    # Extract features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['training']['test_size'],
        random_state=config['models']['random_seed'],
        stratify=y
    )
    
    logger.info(f"Data split: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # 5. Train a simple model
    logger.info("Training random forest model")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=config['models']['random_seed'],
        n_jobs=-1  # Use all cores
    )
    
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    logger.info(f"Model performance: Accuracy = {accuracy:.4f}, ROC AUC = {roc_auc:.4f}")
    
    # 7. Create a simple visualization
    logger.info("Creating feature importance plot")
    plt.figure(figsize=(10, 8))
    
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Features by Importance')
    
    # Save the plot
    importance_path = os.path.join(config['reports']['figures_path'], 'feature_importance.png')
    plt.tight_layout()
    plt.savefig(importance_path)
    plt.close()
    
    logger.info(f"Feature importance plot saved to {importance_path}")
    
    # 8. Create confusion matrix
    logger.info("Creating confusion matrix")
    plt.figure(figsize=(8, 6))
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    # Save the plot
    cm_path = os.path.join(config['reports']['figures_path'], 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # 9. Save model
    model_path = os.path.join(config['models']['save_path'], 'random_forest_model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 10. Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'model_type': 'RandomForest',
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'feature_importance': {X.columns[i]: float(importances[i]) for i in indices}
    }
    
    os.makedirs(os.path.dirname(config['reports']['metrics_path']), exist_ok=True)
    with open(config['reports']['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {config['reports']['metrics_path']}")
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
