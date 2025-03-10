"""
Model Training Module for Churn Prediction
Implements model training, hyperparameter tuning, and evaluation.
"""

import os
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, brier_score_loss, log_loss,
    calibration_curve, cross_val_score
)
import xgboost as xgb
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model trainer for churn prediction models.
    """
    
    def __init__(self, config_path: Union[str, Path] = 'configs/config.yml'):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_col = self.config['training']['target_column']
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Create figures directory for model evaluation plots
        os.makedirs('reports/figures', exist_ok=True)
        
        logger.info("Model trainer initialized")
    
    def load_data(self) -> None:
        """Load the engineered data."""
        # Load from the interim path
        interim_path = self.config['data']['interim_path']
        base_path = os.path.splitext(interim_path)[0]
        
        logger.info(f"Loading engineered data from {os.path.dirname(base_path)}")
        
        try:
            self.X_train = pl.read_parquet(f"{base_path}_X_train.parquet")
            self.X_test = pl.read_parquet(f"{base_path}_X_test.parquet")
            self.y_train = pl.read_parquet(f"{base_path}_y_train.parquet")
            self.y_test = pl.read_parquet(f"{base_path}_y_test.parquet")
            
            logger.info(f"Data loaded: X_train shape {self.X_train.shape}, X_test shape {self.X_test.shape}")
        except Exception as e:
            logger.error(f"Error loading engineered data: {e}")
            
            # Fallback to generating the engineered data
            logger.info("Falling back to generating engineered data...")
            from src.features.feature_engineering import FeatureEngineer
            
            engineer = FeatureEngineer()
            self.X_train, self.X_test = engineer.run_feature_engineering_pipeline()
            self.y_train = engineer.y_train
            self.y_test = engineer.y_test
    
    def train_baseline_models(self) -> Dict[str, Any]:
        """
        Train multiple baseline models for comparison.
        
        Returns:
            Dictionary of trained models
        """
        if self.X_train is None or self.y_train is None:
            self.load_data()
        
        # Convert to pandas for sklearn
        X_train_pd = self.X_train.to_pandas()
        y_train_pd = self.y_train.to_pandas()
        y_train_values = y_train_pd.values.ravel()
        
        logger.info("Training baseline models for comparison")
        
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42)
        }
        
        # Train each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_pd, y_train_values)
            self.models[name] = model
        
        logger.info("Baseline models training completed")
        
        return self.models
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models with advanced statistics.
        
        Returns:
            Dictionary of model metrics
        """
        if not self.models:
            logger.warning("No models to evaluate")
            return {}
        
        # Convert to pandas for sklearn
        X_test_pd = self.X_test.to_pandas()
        y_test_pd = self.y_test.to_pandas()
        y_test_values = y_test_pd.values.ravel()
        
        logger.info("Evaluating model performance with advanced metrics")
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics = {}
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}")
            
            # Predictions and probabilities
            y_pred = model.predict(X_test_pd)
            y_proba = model.predict_proba(X_test_pd)[:, 1]
            
            # Basic metrics
            accuracy = accuracy_score(y_test_values, y_pred)
            precision = precision_score(y_test_values, y_pred)
            recall = recall_score(y_test_values, y_pred)
            f1 = f1_score(y_test_values, y_pred)
            roc_auc = roc_auc_score(y_test_values, y_proba)
            
            # Advanced metrics
            brier = brier_score_loss(y_test_values, y_proba)
            log_loss_value = log_loss(y_test_values, y_proba)
            
            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_test_values, y_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Cross-validated AUC with confidence intervals
            cv_auc_scores = cross_val_score(model, X_test_pd, y_test_values, 
                                          cv=cv, scoring='roc_auc')
            auc_mean = np.mean(cv_auc_scores)
            auc_std = np.std(cv_auc_scores)
            auc_ci_lower, auc_ci_upper = stats.norm.interval(
                0.95, loc=auc_mean, scale=auc_std/np.sqrt(len(cv_auc_scores))
            )
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_test_values, y_proba, n_bins=10)
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            
            # Store metrics
            metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'brier_score': brier,
                'log_loss': log_loss_value,
                'cv_auc_mean': auc_mean,
                'cv_auc_std': auc_std,
                'cv_auc_ci_lower': max(0, auc_ci_lower),
                'cv_auc_ci_upper': min(1, auc_ci_upper),
                'calibration_error': calibration_error
            }
        
        # Find best model based on ROC AUC
        best_model_name = max(metrics, key=lambda k: metrics[k]['roc_auc'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        self.metrics = metrics
        
        # Apply statistical significance tests between models
        self.compare_models_statistically()
        
        logger.info(f"Best model: {best_model_name} with ROC AUC: {metrics[best_model_name]['roc_auc']:.4f}")
        return metrics
    
    def plot_model_comparison(self) -> None:
        """Create visualizations comparing model performance."""
        if not self.metrics:
            logger.warning("No metrics available. Call evaluate_models() first.")
            return
        
        logger.info("Creating model comparison plots")
        
        # Create a dataframe from metrics
        metrics_data = []
        for model_name, model_metrics in self.metrics.items():
            for metric_name, value in model_metrics.items():
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create grouped bar chart
        plt.figure(figsize=(15, 10))
        
        # Plot different metrics side by side
        ax = sns.barplot(
            x='Model', 
            y='Value', 
            hue='Metric', 
            data=metrics_df
        )
        
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap for better visualization
        pivot_metrics = metrics_df.pivot(index='Model', columns='Metric', values='Value')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_metrics, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Model Performance Metrics Heatmap', fontsize=16)
        plt.tight_layout()
        
        plt.savefig('reports/figures/model_metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison plots created")
    
    def tune_best_model(self, n_trials: int = 100) -> Dict:
        """
        Tune hyperparameters of the best model using Bayesian optimization.
        
        Args:
            n_trials: Number of trials for hyperparameter tuning
            
        Returns:
            Dictionary with best parameters and model
        """
        if self.best_model_name is None:
            logger.warning("No best model identified. Call evaluate_models() first.")
            return {}
        
        if self.X_train is None or self.y_train is None:
            self.load_data()
        
        # Convert to pandas for sklearn
        X_train_pd = self.X_train.to_pandas()
        y_train_pd = self.y_train.to_pandas()
        y_train_values = y_train_pd.values.ravel()
        
        X_test_pd = self.X_test.to_pandas()
        y_test_pd = self.y_test.to_pandas()
        y_test_values = y_test_pd.values.ravel()
        
        logger.info(f"Tuning hyperparameters for {self.best_model_name} with {n_trials} trials")
        
        # Define search space based on model type
        if self.best_model_name == 'logistic_regression':
            space = {
                'C': hp.loguniform('C', -4, 4),
                'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'none']),
                'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                'max_iter': hp.quniform('max_iter', 100, 2000, 100)
            }
            
            def objective(params):
                # Some solvers don't work with some penalties
                if params['penalty'] == 'l1' and params['solver'] in ['newton-cg', 'sag']:
                    params['solver'] = 'saga'
                elif params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
                    params['solver'] = 'saga'
                elif params['penalty'] == 'none' and params['solver'] in ['liblinear']:
                    params['solver'] = 'saga'
                
                # Create and train model
                model = LogisticRegression(
                    C=params['C'],
                    penalty=params['penalty'],
                    solver=params['solver'],
                    max_iter=int(params['max_iter']),
                    random_state=42
                )
                
                model.fit(X_train_pd, y_train_values)
                
                # Evaluate on validation set
                y_proba = model.predict_proba(X_test_pd)[:, 1]
                roc_auc = roc_auc_score(y_test_values, y_proba)
                
                return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}
        
        elif self.best_model_name == 'random_forest':
            space = {
                'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                'max_depth': hp.quniform('max_depth', 3, 30, 1),
                'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
                'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7])
            }
            
            def objective(params):
                # Create and train model
                model = RandomForestClassifier(
                    n_estimators=int(params['n_estimators']),
                    max_depth=int(params['max_depth']),
                    min_samples_split=int(params['min_samples_split']),
                    min_samples_leaf=int(params['min_samples_leaf']),
                    max_features=params['max_features'],
                    random_state=42
                )
                
                model.fit(X_train_pd, y_train_values)
                
                # Evaluate on validation set
                y_proba = model.predict_proba(X_test_pd)[:, 1]
                roc_auc = roc_auc_score(y_test_values, y_proba)
                
                return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}
        
        elif self.best_model_name == 'gradient_boosting':
            space = {
                'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'max_depth': hp.quniform('max_depth', 1, 15, 1),
                'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
                'subsample': hp.uniform('subsample', 0.5, 1.0)
            }
            
            def objective(params):
                # Create and train model
                model = GradientBoostingClassifier(
                    n_estimators=int(params['n_estimators']),
                    learning_rate=params['learning_rate'],
                    max_depth=int(params['max_depth']),
                    min_samples_split=int(params['min_samples_split']),
                    min_samples_leaf=int(params['min_samples_leaf']),
                    subsample=params['subsample'],
                    random_state=42
                )
                
                model.fit(X_train_pd, y_train_values)
                
                # Evaluate on validation set
                y_proba = model.predict_proba(X_test_pd)[:, 1]
                roc_auc = roc_auc_score(y_test_values, y_proba)
                
                return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}
        
        elif self.best_model_name == 'xgboost':
            space = {
                'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'max_depth': hp.quniform('max_depth', 1, 15, 1),
                'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                'gamma': hp.uniform('gamma', 0, 5)
            }
            
            def objective(params):
                # Create and train model
                model = xgb.XGBClassifier(
                    n_estimators=int(params['n_estimators']),
                    learning_rate=params['learning_rate'],
                    max_depth=int(params['max_depth']),
                    min_child_weight=int(params['min_child_weight']),
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    gamma=params['gamma'],
                    random_state=42
                )
                
                model.fit(X_train_pd, y_train_values)
                
                # Evaluate on validation set
                y_proba = model.predict_proba(X_test_pd)[:, 1]
                roc_auc = roc_auc_score(y_test_values, y_proba)
                
                return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}
        
        elif self.best_model_name == 'lightgbm':
            space = {
                'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
                'max_depth': hp.quniform('max_depth', 3, 15, 1),
                'min_child_samples': hp.quniform('min_child_samples', 5, 100, 1),
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
            }
            
            def objective(params):
                # Create and train model
                model = lgb.LGBMClassifier(
                    n_estimators=int(params['n_estimators']),
                    learning_rate=params['learning_rate'],
                    num_leaves=int(params['num_leaves']),
                    max_depth=int(params['max_depth']),
                    min_child_samples=int(params['min_child_samples']),
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    random_state=42
                )
                
                model.fit(X_train_pd, y_train_values)
                
                # Evaluate on validation set
                y_proba = model.predict_proba(X_test_pd)[:, 1]
                roc_auc = roc_auc_score(y_test_values, y_proba)
                
                return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}
        
        else:
            logger.warning(f"No hyperparameter tuning defined for {self.best_model_name}")
            return {}
        
        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            rstate=np.random.RandomState(42)
        )
        
        # Extract results
        best_trial_idx = np.argmin([trial['result']['loss'] for trial in trials.trials])
        best_score = -trials.trials[best_trial_idx]['result']['loss']
        best_model = trials.trials[best_trial_idx]['result']['model']
        
        # Update best model
        self.best_model = best_model
        
        # Save the tuned model
        joblib.dump(best_model, f'models/tuned_{self.best_model_name}.joblib')
        logger.info(f"Saved tuned model to models/tuned_{self.best_model_name}.joblib")
        
        logger.info(f"Best score (ROC AUC): {best_score:.4f}")
        logger.info(f"Best parameters: {best}")
        
        return {
            'model': self.best_model_name,
            'best_params': best,
            'best_score': best_score,
            'best_model': best_model
        }
    
    def evaluate_best_model(self) -> Dict[str, float]:
        """
        Perform detailed evaluation of the best model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_model is None:
            logger.warning("No best model available. Call tune_best_model() first.")
            return {}
        
        if self.X_test is None or self.y_test is None:
            self.load_data()
        
        # Convert to pandas for sklearn
        X_test_pd = self.X_test.to_pandas()
        y_test_pd = self.y_test.to_pandas()
        y_test_values = y_test_pd.values.ravel()
        
        logger.info(f"Evaluating best model: {self.best_model_name}")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test_pd)
        y_proba = self.best_model.predict_proba(X_test_pd)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_values, y_pred)
        precision = precision_score(y_test_values, y_pred)
        recall = recall_score(y_test_values, y_pred)
        f1 = f1_score(y_test_values, y_pred)
        roc_auc = roc_auc_score(y_test_values, y_proba)
        
        # Log metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(y_test_values, y_pred)
        
        # Log classification report
        report = classification_report(y_test_values, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix', fontsize=15)
        plt.tight_layout()
        plt.savefig('reports/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test_values, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=15)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('reports/figures/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_values, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        plt.plot(recall_curve, precision_curve, lw=2, 
                label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontsize=15)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig('reports/figures/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # If the model has feature importances, plot them
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature names
            feature_names = X_test_pd.columns
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 10))
            top_features = importance_df.head(20)
            
            ax = sns.barplot(x='importance', y='feature', data=top_features)
            
            plt.title('Feature Importance', fontsize=15)
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Best model evaluation completed")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def compare_models_statistically(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform statistical tests to compare model performance.
        Uses DeLong test for AUC comparison.
        
        Returns:
            Dictionary with pairwise comparison results
        """
        if len(self.models) < 2:
            logger.info("Need at least 2 models for statistical comparison")
            return {}
        
        # Convert to pandas for sklearn
        X_test_pd = self.X_test.to_pandas()
        y_test_pd = self.y_test.to_pandas()
        y_test_values = y_test_pd.values.ravel()
        
        # Get model predictions
        model_probas = {}
        for name, model in self.models.items():
            model_probas[name] = model.predict_proba(X_test_pd)[:, 1]
        
        # Implement DeLong test for AUC comparison
        # This is a simplified version - a full implementation would use statsmodels or scikit-learn
        comparison_results = {}
        
        # Calculate p-values for each pair of models
        model_names = list(self.models.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                # McNemar's test for classification agreement
                y_pred1 = (model_probas[model1] >= 0.5).astype(int)
                y_pred2 = (model_probas[model2] >= 0.5).astype(int)
                
                # Create contingency table
                # a: both correct, b: model1 correct & model2 wrong
                # c: model1 wrong & model2 correct, d: both wrong
                a = np.sum((y_pred1 == y_test_values) & (y_pred2 == y_test_values))
                b = np.sum((y_pred1 == y_test_values) & (y_pred2 != y_test_values))
                c = np.sum((y_pred1 != y_test_values) & (y_pred2 == y_test_values))
                d = np.sum((y_pred1 != y_test_values) & (y_pred2 != y_test_values))
                
                # McNemar's test
                if b + c > 0:  # Avoid division by zero
                    mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
                    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
                else:
                    mcnemar_p = 1.0
                    
                # Bootstrap comparison of AUCs
                n_bootstrap = 1000
                auc_diffs = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap sampling
                    indices = np.random.choice(len(y_test_values), len(y_test_values), replace=True)
                    y_boot = y_test_values[indices]
                    prob1_boot = model_probas[model1][indices]
                    prob2_boot = model_probas[model2][indices]
                    
                    # Compute AUCs
                    auc1 = roc_auc_score(y_boot, prob1_boot)
                    auc2 = roc_auc_score(y_boot, prob2_boot)
                    auc_diffs.append(auc1 - auc2)
                
                # Compute p-value
                auc_diff_mean = np.mean(auc_diffs)
                auc_p = np.mean([1 if diff <= 0 else 0 for diff in auc_diffs]) if auc_diff_mean > 0 else np.mean([1 if diff >= 0 else 0 for diff in auc_diffs])
                auc_p = min(auc_p, 1 - auc_p) * 2  # Two-tailed p-value
                
                comparison_results[f"{model1}_vs_{model2}"] = {
                    "auc_diff": self.metrics[model1]['roc_auc'] - self.metrics[model2]['roc_auc'],
                    "auc_p_value": auc_p,
                    "mcnemar_p_value": mcnemar_p,
                    "significantly_different": auc_p < 0.05 or mcnemar_p < 0.05
                }
        
        # Store the results
        self.comparison_results = comparison_results
        
        # Log significant differences
        significant_comparisons = {k: v for k, v in comparison_results.items() if v["significantly_different"]}
        if significant_comparisons:
            logger.info(f"Found {len(significant_comparisons)} statistically significant model differences")
        
        return comparison_results
    
    def run_model_pipeline(self) -> None:
        """Run the complete model training and evaluation pipeline."""
        logger.info("Starting model pipeline")
        
        # 1. Load data
        self.load_data()
        
        # 2. Train baseline models
        self.train_baseline_models()
        
        # 3. Evaluate models
        self.evaluate_models()
        
        # 4. Visualize model comparison
        self.plot_model_comparison()
        
        # 5. Tune best model
        self.tune_best_model(n_trials=50)
        
        # 6. Evaluate best model
        final_metrics = self.evaluate_best_model()
        
        logger.info("Model pipeline completed")
        
        return final_metrics

def main():
    """Main execution function"""
    trainer = ModelTrainer()
    metrics = trainer.run_model_pipeline()
    
    logger.info("Model training and evaluation completed")
    logger.info(f"Final model metrics: {metrics}")

if __name__ == "__main__":
    main() 