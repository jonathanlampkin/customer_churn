"""
Model Explainer module for Churn Prediction
Provides SHAP-based explainability for trained models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import yaml
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """Class for generating model explanations using SHAP."""
    
    def __init__(self, config_path: Union[str, Path] = 'configs/config.yml'):
        """
        Initialize the model explainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.model_name = None
        self.X_test = None
        self.y_test = None
        self.explainer = None
        self.shap_values = None
        
        # Create figures directory for explanation plots
        os.makedirs('reports/figures/explanations', exist_ok=True)
        
        logger.info("Model explainer initialized")
    
    def load_model(self) -> Any:
        """
        Load the best tuned model from disk.
        
        Returns:
            Loaded model
        """
        # Get list of models in the models directory
        model_files = [f for f in os.listdir('models') if f.startswith('tuned_')]
        
        if not model_files:
            logger.warning("No tuned models found")
            return None
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join('models', latest_model)
        
        try:
            self.model = joblib.load(model_path)
            self.model_name = latest_model.replace('tuned_', '').replace('.joblib', '')
            logger.info(f"Loaded model {self.model_name} from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load test data for explanations.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        try:
            # Load the interim data
            interim_path = self.config['data']['interim_path']
            base_path = os.path.splitext(interim_path)[0]
            
            # Since we're using Polars, convert to pandas for SHAP
            import polars as pl
            X_test_pl = pl.read_parquet(f"{base_path}_X_test.parquet")
            y_test_pl = pl.read_parquet(f"{base_path}_y_test.parquet")
            
            self.X_test = X_test_pl.to_pandas()
            self.y_test = y_test_pl.to_pandas()
            
            logger.info(f"Loaded test data with {len(self.X_test)} samples")
            return self.X_test, self.y_test
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return None, None
    
    def create_explainer(self) -> Any:
        """
        Create a SHAP explainer for the model.
        
        Returns:
            SHAP explainer
        """
        if self.model is None:
            logger.error("No model loaded")
            return None
        
        if self.X_test is None:
            logger.error("No test data loaded")
            return None
        
        try:
            # Determine the type of explainer to use based on the model type
            model_type = type(self.model).__name__
            
            # Tree-based models: Use TreeExplainer for efficiency
            if any(t in model_type for t in ['RandomForest', 'GradientBoosting', 'XGB', 'LightGBM']):
                logger.info(f"Creating TreeExplainer for {model_type}")
                self.explainer = shap.TreeExplainer(self.model)
            # Linear models: Use LinearExplainer
            elif 'Linear' in model_type or 'Logistic' in model_type:
                logger.info(f"Creating LinearExplainer for {model_type}")
                self.explainer = shap.LinearExplainer(self.model, self.X_test)
            # Default to KernelExplainer for other model types
            else:
                logger.info(f"Creating KernelExplainer for {model_type}")
                # Use a small sample for background data
                background = shap.kmeans(self.X_test, 100)
                self.explainer = shap.KernelExplainer(
                    model=lambda x: self.model.predict_proba(x)[:, 1],
                    data=background
                )
            
            logger.info("SHAP explainer created successfully")
            return self.explainer
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            return None
    
    def compute_shap_values(self, sample_size: int = 1000) -> np.ndarray:
        """
        Compute SHAP values for test data.
        
        Args:
            sample_size: Number of samples to use
            
        Returns:
            Array of SHAP values
        """
        if self.explainer is None:
            logger.error("No explainer created")
            return None
        
        if self.X_test is None:
            logger.error("No test data loaded")
            return None
        
        try:
            # Limit the number of samples for computation
            n_samples = min(sample_size, self.X_test.shape[0])
            X_sample = self.X_test.iloc[:n_samples]
            
            logger.info(f"Computing SHAP values for {n_samples} samples")
            
            # Compute SHAP values
            if isinstance(self.explainer, shap.TreeExplainer):
                self.shap_values = self.explainer.shap_values(X_sample)
                
                # For multi-class models, get the SHAP values for the positive class
                if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
                    self.shap_values = self.shap_values[1]  # Assuming binary classification
            else:
                self.shap_values = self.explainer.shap_values(X_sample)
            
            logger.info("SHAP values computed successfully")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def plot_summary(self) -> None:
        """Generate and save SHAP summary plot."""
        if self.shap_values is None:
            logger.error("No SHAP values computed")
            return
        
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                self.shap_values, 
                self.X_test.iloc[:len(self.shap_values)], 
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig('reports/figures/explanations/shap_summary_bar.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(14, 10))
            shap.summary_plot(
                self.shap_values, 
                self.X_test.iloc[:len(self.shap_values)],
                show=False
            )
            plt.tight_layout()
            plt.savefig('reports/figures/explanations/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("SHAP summary plots created")
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
    
    def plot_dependence(self, feature: str) -> None:
        """
        Generate and save SHAP dependence plot for a feature.
        
        Args:
            feature: Feature name to plot
        """
        if self.shap_values is None:
            logger.error("No SHAP values computed")
            return
        
        if feature not in self.X_test.columns:
            logger.error(f"Feature {feature} not found in data")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            interaction_idx = "auto"
            shap.dependence_plot(
                feature, 
                self.shap_values, 
                self.X_test.iloc[:len(self.shap_values)],
                interaction_index=interaction_idx,
                show=False
            )
            plt.tight_layout()
            plt.savefig(f'reports/figures/explanations/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP dependence plot created for {feature}")
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot for {feature}: {e}")
    
    def create_explanation_report(self, sample_idx: int) -> Dict:
        """
        Create a comprehensive explanation report for a single sample.
        
        Args:
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary with explanation information
        """
        if self.shap_values is None or self.X_test is None:
            logger.error("No SHAP values or test data available")
            return {}
        
        if sample_idx >= len(self.shap_values):
            logger.error(f"Sample index {sample_idx} out of bounds")
            return {}
        
        try:
            # Get the sample data
            sample = self.X_test.iloc[sample_idx]
            
            # Get actual prediction
            prediction_proba = self.model.predict_proba(sample.values.reshape(1, -1))[0, 1]
            prediction = bool(prediction_proba >= 0.5)
            
            # Get SHAP values for this sample
            sample_shap_values = self.shap_values[sample_idx]
            
            # Sort features by SHAP value magnitude
            feature_importance = [(self.X_test.columns[i], float(sample_shap_values[i])) 
                                 for i in range(len(sample.values))]
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Format the explanation
            explanation = {
                "sample_id": int(sample_idx),
                "prediction": bool(prediction),
                "prediction_probability": float(prediction_proba),
                "features": dict(sample),
                "feature_importance": dict(feature_importance),
                "top_positive_features": [],
                "top_negative_features": []
            }
            
            # Get top positive and negative features
            for feature, value in feature_importance:
                if value > 0:
                    explanation["top_positive_features"].append({
                        "feature": feature,
                        "contribution": float(value),
                        "value": float(sample[feature]) if isinstance(sample[feature], (int, float)) else str(sample[feature])
                    })
                else:
                    explanation["top_negative_features"].append({
                        "feature": feature,
                        "contribution": float(value),
                        "value": float(sample[feature]) if isinstance(sample[feature], (int, float)) else str(sample[feature])
                    })
            
            # Limit to top 5 for each
            explanation["top_positive_features"] = explanation["top_positive_features"][:5]
            explanation["top_negative_features"] = explanation["top_negative_features"][:5]
            
            # Create a visual explanation for this sample
            plt.figure(figsize=(14, 8))
            
            # Plot the SHAP values as a force plot
            shap.force_plot(
                self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                sample_shap_values,
                sample,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            plt.savefig(f'reports/figures/explanations/sample_{sample_idx}_force_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a decision plot
            plt.figure(figsize=(12, 8))
            shap.decision_plot(
                self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                sample_shap_values,
                sample,
                feature_display_range=20,
                show=False
            )
            plt.tight_layout()
            plt.savefig(f'reports/figures/explanations/sample_{sample_idx}_explanation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Explanation report created for sample {sample_idx}")
            return explanation
        
        except Exception as e:
            logger.error(f"Error creating explanation report: {e}")
            return {}
    
    def run_all_explanations(self, sample_size: int = 1000) -> None:
        """
        Run all explanation methods.
        
        Args:
            sample_size: Number of samples to use for SHAP computation
        """
        # Load model and data if not already loaded
        if self.model is None:
            self.load_model()
            
        if self.X_test is None:
            self.load_data()
        
        # Create explainer and compute SHAP values
        self.create_explainer()
        self.compute_shap_values(sample_size=sample_size)
        
        # Generate plots
        self.plot_summary()
        
        # Plot dependence for top 5 features
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        top_features = [self.X_test.columns[i] for i in feature_importance.argsort()[-5:]]
        
        for feature in top_features:
            self.plot_dependence(feature)
        
        # Create explanation reports for 3 random samples
        explanations = []
        for _ in range(3):
            sample_idx = np.random.randint(0, min(len(self.shap_values), self.X_test.shape[0]))
            explanation = self.create_explanation_report(sample_idx)
            explanations.append(explanation)
        
        # Save explanations to file
        import json
        with open('reports/sample_explanations.json', 'w') as f:
            json.dump(explanations, f, indent=2)
        
        logger.info("All explanations completed")

def main():
    """Main execution function"""
    explainer = ModelExplainer()
    explainer.run_all_explanations()
    
    logger.info("Model explanation completed")

if __name__ == "__main__":
    main() 