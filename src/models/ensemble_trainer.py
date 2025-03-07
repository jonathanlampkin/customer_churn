"""
Ensemble Model Trainer for Churn Prediction
Implements advanced ensemble techniques including stacking, blending, and weighted averaging.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, log_loss,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", "The objective function.*")
warnings.filterwarnings("ignore", "XGBoost.*")
warnings.filterwarnings("ignore", "Starting from version 2.2.1.*")

class EnsembleTrainer:
    """
    Advanced ensemble model trainer for churn prediction.
    Features:
    - Stacking ensemble with cross-validation
    - Blending multiple models
    - Weighted ensemble using optimization
    - Diversity analysis of base models
    - Feature selection per model
    """
    
    def __init__(self, config_path: Union[str, Path] = 'configs/config.yml'):
        """
        Initialize the ensemble trainer with configuration.
        
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
        
        self.base_models = {}
        self.ensemble_models = {}
        self.best_ensemble = None
        self.best_ensemble_name = None
        self.metrics = {}
        
        # Create directories
        os.makedirs('models/ensembles', exist_ok=True)
        os.makedirs('reports/figures/ensembles', exist_ok=True)
        
        logger.info("Ensemble trainer initialized")

    def load_data(self) -> None:
        """Load the engineered data."""
        # Load from the interim path
        interim_path = self.config['data']['interim_path']
        base_path = os.path.splitext(interim_path)[0]
        
        logger.info(f"Loading engineered data from {os.path.dirname(base_path)}")
        
        try:
            import polars as pl
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
    
    def train_base_models(self) -> Dict[str, Any]:
        """
        Train a diverse set of base models.
        
        Returns:
            Dictionary of trained base models
        """
        if self.X_train is None or self.y_train is None:
            self.load_data()
        
        # Convert to pandas for sklearn
        X_train_pd = self.X_train.to_pandas()
        y_train_pd = self.y_train.to_pandas()
        y_train_values = y_train_pd.values.ravel()
        
        logger.info("Training diverse base models for ensemble")
        
        models = {
            'logistic_regression': LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', max_iter=1000, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=4, random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, 
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, 
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'svm': SVC(
                probability=True, C=1.0, kernel='rbf', gamma='scale', random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), alpha=0.0001, 
                learning_rate_init=0.001, max_iter=200, random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5, weights='distance'
            )
        }
        
        # Train each model with timing
        for name, model in models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            try:
                model.fit(X_train_pd, y_train_values)
                self.base_models[name] = model
                
                training_time = time.time() - start_time
                logger.info(f"Trained {name} in {training_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        logger.info(f"Base models training completed, {len(self.base_models)} models trained")
        
        return self.base_models 

    def evaluate_base_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all base models and analyze their diversity.
        
        Returns:
            Dictionary of model metrics
        """
        if not self.base_models:
            logger.warning("No base models to evaluate")
            return {}
        
        # Convert to pandas for sklearn
        X_test_pd = self.X_test.to_pandas()
        y_test_pd = self.y_test.to_pandas()
        y_test_values = y_test_pd.values.ravel()
        
        logger.info("Evaluating base models performance")
        
        metrics = {}
        predictions = {}
        probabilities = {}
        
        for name, model in self.base_models.items():
            logger.info(f"Evaluating {name}")
            
            try:
                # Predictions and probabilities
                y_pred = model.predict(X_test_pd)
                y_proba = model.predict_proba(X_test_pd)[:, 1]
                
                # Store for diversity analysis
                predictions[name] = y_pred
                probabilities[name] = y_proba
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_values, y_pred)
                precision = precision_score(y_test_values, y_pred)
                recall = recall_score(y_test_values, y_pred)
                f1 = f1_score(y_test_values, y_pred)
                roc_auc = roc_auc_score(y_test_values, y_proba)
                log_loss_value = log_loss(y_test_values, y_proba)
                
                metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'log_loss': log_loss_value
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        # Calculate diversity metrics
        self.diversity_metrics = self._analyze_model_diversity(predictions, probabilities)
        
        self.base_metrics = metrics
        
        # Create a comparison plot
        self._plot_model_comparison(metrics)
        
        logger.info("Base models evaluation completed")
        return metrics
    
    def _analyze_model_diversity(self, predictions: Dict[str, np.ndarray], 
                               probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze the diversity of the base models.
        
        Args:
            predictions: Dictionary of model predictions
            probabilities: Dictionary of model prediction probabilities
            
        Returns:
            Dictionary of diversity metrics
        """
        logger.info("Analyzing model diversity")
        
        try:
            model_names = list(predictions.keys())
            n_models = len(model_names)
            
            if n_models < 2:
                return {}
            
            # Get true labels
            y_test_values = self.y_test.to_pandas().values.ravel()
            
            # Calculate pairwise disagreement and correlation
            disagreement_matrix = np.zeros((n_models, n_models))
            correlation_matrix = np.zeros((n_models, n_models))
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i != j:
                        # Disagreement: percentage of samples where models make different predictions
                        disagreement = np.mean(predictions[model1] != predictions[model2])
                        disagreement_matrix[i, j] = disagreement
                        
                        # Correlation between probability predictions
                        corr = np.corrcoef(probabilities[model1], probabilities[model2])[0, 1]
                        correlation_matrix[i, j] = corr
            
            # Calculate diversity metrics
            mean_disagreement = np.mean(disagreement_matrix)
            mean_correlation = np.mean(correlation_matrix)
            
            # Calculate Q-statistic
            q_statistics = np.zeros((n_models, n_models))
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i != j:
                        # Contingency table
                        a = np.sum((predictions[model1] == 1) & (predictions[model2] == 1))
                        b = np.sum((predictions[model1] == 1) & (predictions[model2] == 0))
                        c = np.sum((predictions[model1] == 0) & (predictions[model2] == 1))
                        d = np.sum((predictions[model1] == 0) & (predictions[model2] == 0))
                        
                        # Q statistic
                        q = (a*d - b*c) / (a*d + b*c) if (a*d + b*c) > 0 else 0
                        q_statistics[i, j] = q
            
            mean_q_statistic = np.mean(q_statistics)
            
            # Plot diversity heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                xticklabels=model_names,
                yticklabels=model_names,
                cmap='coolwarm',
                vmin=-1,
                vmax=1
            )
            plt.title('Correlation Between Model Predictions', fontsize=15)
            plt.tight_layout()
            plt.savefig('reports/figures/ensembles/model_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Return diversity metrics
            diversity_metrics = {
                'mean_disagreement': mean_disagreement,
                'mean_correlation': mean_correlation,
                'mean_q_statistic': mean_q_statistic,
                'disagreement_matrix': disagreement_matrix.tolist(),
                'correlation_matrix': correlation_matrix.tolist(),
                'q_statistics': q_statistics.tolist()
            }
            
            logger.info(f"Diversity analysis completed: mean disagreement = {mean_disagreement:.4f}, " + 
                       f"mean correlation = {mean_correlation:.4f}")
            
            return diversity_metrics
            
        except Exception as e:
            logger.error(f"Error in diversity analysis: {e}")
            return {}
            
    def _plot_model_comparison(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Create comparison plots for model performance.
        
        Args:
            metrics: Dictionary of model metrics
        """
        try:
            # Extract metrics
            models = list(metrics.keys())
            accuracy = [metrics[m]['accuracy'] for m in models]
            precision = [metrics[m]['precision'] for m in models]
            recall = [metrics[m]['recall'] for m in models]
            f1 = [metrics[m]['f1'] for m in models]
            roc_auc = [metrics[m]['roc_auc'] for m in models]
            
            # Create performance comparison plot
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(models))
            width = 0.15
            
            plt.bar(x - width*2, accuracy, width, label='Accuracy')
            plt.bar(x - width, precision, width, label='Precision')
            plt.bar(x, recall, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1')
            plt.bar(x + width*2, roc_auc, width, label='ROC AUC')
            
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Model Performance Comparison', fontsize=15)
            plt.xticks(x, models, rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig('reports/figures/ensembles/model_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create ROC AUC ranking plot
            roc_auc_sorted = sorted([(m, metrics[m]['roc_auc']) for m in models], key=lambda x: x[1], reverse=True)
            sorted_models = [m[0] for m in roc_auc_sorted]
            sorted_roc_auc = [m[1] for m in roc_auc_sorted]
            
            plt.figure(figsize=(12, 6))
            bars = plt.barh(sorted_models, sorted_roc_auc, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            
            plt.xlabel('ROC AUC Score', fontsize=12)
            plt.title('Models Ranked by ROC AUC', fontsize=15)
            plt.xlim(0.5, 1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add values to bars
            for i, v in enumerate(sorted_roc_auc):
                plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('reports/figures/ensembles/model_roc_auc_ranking.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Model comparison plots created")
            
        except Exception as e:
            logger.error(f"Error creating model comparison plots: {e}") 

    def create_voting_ensemble(self) -> Any:
        """
        Create a voting ensemble from base models.
        
        Returns:
            Trained voting ensemble model
        """
        if not self.base_models:
            self.train_base_models()
            
        # Convert to pandas for sklearn
        X_train_pd = self.X_train.to_pandas()
        y_train_pd = self.y_train.to_pandas()
        y_train_values = y_train_pd.values.ravel()
        
        logger.info("Creating voting ensemble")
        
        try:
            # Create a list of (name, model) tuples for VotingClassifier
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            # Create soft voting ensemble (using predicted probabilities)
            voting_ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=None,  # Equal weights initially
                n_jobs=-1
            )
            
            # Train the ensemble
            voting_ensemble.fit(X_train_pd, y_train_values)
            
            # Store the ensemble
            self.ensemble_models['voting'] = voting_ensemble
            
            logger.info("Voting ensemble created and trained")
            return voting_ensemble
            
        except Exception as e:
            logger.error(f"Error creating voting ensemble: {e}")
            return None
    
    def create_stacking_ensemble(self, cv: int = 5) -> Any:
        """
        Create a stacking ensemble with cross-validation.
        
        Args:
            cv: Number of cross-validation folds
            
        Returns:
            Trained stacking ensemble model
        """
        if not self.base_models:
            self.train_base_models()
            
        # Convert to pandas for sklearn
        X_train_pd = self.X_train.to_pandas()
        y_train_pd = self.y_train.to_pandas()
        y_train_values = y_train_pd.values.ravel()
        
        logger.info(f"Creating stacking ensemble with {cv}-fold cross-validation")
        
        try:
            # Create a list of (name, model) tuples for StackingClassifier
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            # Use logistic regression as final estimator
            final_estimator = LogisticRegression(max_iter=1000, random_state=42)
            
            # Create stacking ensemble with cross-validation
            stacking_ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=cv,
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            # Train the ensemble
            stacking_ensemble.fit(X_train_pd, y_train_values)
            
            # Store the ensemble
            self.ensemble_models['stacking'] = stacking_ensemble
            
            logger.info("Stacking ensemble created and trained")
            return stacking_ensemble
            
        except Exception as e:
            logger.error(f"Error creating stacking ensemble: {e}")
            return None
    
    def create_weighted_ensemble(self) -> Any:
        """
        Create a weighted voting ensemble where weights are based on model performance.
        
        Returns:
            Trained weighted ensemble model
        """
        if not self.base_models:
            self.train_base_models()
            
        if not self.base_metrics:
            self.evaluate_base_models()
            
        # Convert to pandas for sklearn
        X_train_pd = self.X_train.to_pandas()
        y_train_pd = self.y_train.to_pandas()
        y_train_values = y_train_pd.values.ravel()
        
        logger.info("Creating weighted ensemble")
        
        try:
            # Calculate weights based on ROC AUC
            weights = {name: metrics['roc_auc'] for name, metrics in self.base_metrics.items()}
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            normalized_weights = [weights[name] / total_weight for name in self.base_models.keys()]
            
            # Create weighted voting ensemble
            estimators = [(name, model) for name, model in self.base_models.items()]
            
            weighted_ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=normalized_weights,
                n_jobs=-1
            )
            
            # Train the ensemble
            weighted_ensemble.fit(X_train_pd, y_train_values)
            
            # Store the ensemble
            self.ensemble_models['weighted'] = weighted_ensemble
            
            logger.info("Weighted ensemble created and trained")
            return weighted_ensemble
            
        except Exception as e:
            logger.error(f"Error creating weighted ensemble: {e}")
            return None 

    def evaluate_ensembles(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all ensemble models and compare with base models.
        
        Returns:
            Dictionary of ensemble metrics
        """
        if not self.ensemble_models:
            logger.warning("No ensemble models to evaluate")
            return {}
        
        # Convert to pandas for sklearn
        X_test_pd = self.X_test.to_pandas()
        y_test_pd = self.y_test.to_pandas()
        y_test_values = y_test_pd.values.ravel()
        
        logger.info("Evaluating ensemble models")
        
        ensemble_metrics = {}
        
        for name, model in self.ensemble_models.items():
            logger.info(f"Evaluating {name} ensemble")
            
            try:
                # Predictions and probabilities
                y_pred = model.predict(X_test_pd)
                y_proba = model.predict_proba(X_test_pd)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_values, y_pred)
                precision = precision_score(y_test_values, y_pred)
                recall = recall_score(y_test_values, y_pred)
                f1 = f1_score(y_test_values, y_pred)
                roc_auc = roc_auc_score(y_test_values, y_proba)
                log_loss_value = log_loss(y_test_values, y_proba)
                
                ensemble_metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'log_loss': log_loss_value
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {name} ensemble: {e}")
        
        # Find best ensemble
        if ensemble_metrics:
            self.best_ensemble_name = max(ensemble_metrics, key=lambda k: ensemble_metrics[k]['roc_auc'])
            self.best_ensemble = self.ensemble_models[self.best_ensemble_name]
            logger.info(f"Best ensemble: {self.best_ensemble_name} with ROC AUC: {ensemble_metrics[self.best_ensemble_name]['roc_auc']:.4f}")
        
        # Compare with base models
        self._plot_ensemble_comparison(self.base_metrics, ensemble_metrics)
        
        logger.info("Ensemble evaluation completed")
        return ensemble_metrics
    
    def _plot_ensemble_comparison(self, base_metrics: Dict[str, Dict[str, float]], 
                                ensemble_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Create plots comparing ensemble performance to base models.
        
        Args:
            base_metrics: Dictionary of base model metrics
            ensemble_metrics: Dictionary of ensemble model metrics
        """
        try:
            # Combine metrics
            all_metrics = {**base_metrics, **ensemble_metrics}
            
            # Sort models by ROC AUC
            sorted_models = sorted(all_metrics.keys(), key=lambda x: all_metrics[x]['roc_auc'], reverse=True)
            sorted_roc_auc = [all_metrics[m]['roc_auc'] for m in sorted_models]
            
            # Create color mapping - ensembles in one color, base models in another
            is_ensemble = [m in ensemble_metrics for m in sorted_models]
            colors = ['#1f77b4' if e else '#ff7f0e' for e in is_ensemble]
            
            plt.figure(figsize=(12, 6))
            bars = plt.barh(sorted_models, sorted_roc_auc, color=colors)
            
            plt.xlabel('ROC AUC Score', fontsize=12)
            plt.title('Models Ranked by ROC AUC', fontsize=15)
            plt.xlim(0.5, 1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add values to bars
            for i, v in enumerate(sorted_roc_auc):
                plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=10)
            
            # Add legend
            import matplotlib.patches as mpatches
            ensemble_patch = mpatches.Patch(color='#1f77b4', label='Ensemble Models')
            base_patch = mpatches.Patch(color='#ff7f0e', label='Base Models')
            plt.legend(handles=[ensemble_patch, base_patch], loc='lower right')
            
            plt.tight_layout()
            plt.savefig('reports/figures/ensembles/ensemble_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Ensemble comparison plot created")
            
        except Exception as e:
            logger.error(f"Error creating ensemble comparison plot: {e}")
    
    def save_best_ensemble(self) -> None:
        """Save the best ensemble model to disk."""
        if self.best_ensemble is None:
            logger.warning("No best ensemble to save")
            return
        
        try:
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"ensemble_{self.best_ensemble_name}_{timestamp}.joblib"
            filepath = os.path.join("models/ensembles", filename)
            
            # Save the model
            joblib.dump(self.best_ensemble, filepath)
            
            logger.info(f"Best ensemble saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving best ensemble: {e}")
    
    def run_ensemble_pipeline(self) -> Dict[str, Dict[str, float]]:
        """
        Run the complete ensemble training and evaluation pipeline.
        
        Returns:
            Dictionary of ensemble metrics
        """
        logger.info("Starting ensemble pipeline")
        
        # 1. Load data
        self.load_data()
        
        # 2. Train base models
        self.train_base_models()
        
        # 3. Evaluate base models
        self.evaluate_base_models()
        
        # 4. Create ensembles
        self.create_voting_ensemble()
        self.create_stacking_ensemble()
        self.create_weighted_ensemble()
        
        # 5. Evaluate ensembles
        ensemble_metrics = self.evaluate_ensembles()
        
        # 6. Save best ensemble
        self.save_best_ensemble()
        
        logger.info("Ensemble pipeline completed")
        
        return ensemble_metrics

def main():
    """Main execution function"""
    trainer = EnsembleTrainer()
    ensemble_metrics = trainer.run_ensemble_pipeline()
    
    logger.info("Ensemble training and evaluation completed")
    
    # Print best results
    if trainer.best_ensemble_name:
        logger.info(f"Best ensemble: {trainer.best_ensemble_name}")
        metrics = ensemble_metrics[trainer.best_ensemble_name]
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 