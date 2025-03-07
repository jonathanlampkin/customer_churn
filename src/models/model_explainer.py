"""
Model Explainer module for Churn Prediction
Provides advanced model interpretability using SHAP and other techniques.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import yaml
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from sklearn.inspection import permutation_importance, partial_dependence, plot_partial_dependence
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", "The 'nopython' keyword.*")

class ModelExplainer:
    """
    Class for generating comprehensive model explanations using advanced techniques.
    Features:
    - SHAP-based local and global explanations
    - Feature importance analysis
    - Feature interaction detection
    - Partial dependence plots
    - ICE (Individual Conditional Expectation) plots
    """
    
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.permutation_importances = None
        
        # Create directories for explanation plots
        os.makedirs('reports/figures/explanations', exist_ok=True)
        os.makedirs('reports/figures/explanations/shap', exist_ok=True)
        os.makedirs('reports/figures/explanations/pdp', exist_ok=True)
        os.makedirs('reports/figures/explanations/interactions', exist_ok=True)
        
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
            X_train_pl = pl.read_parquet(f"{base_path}_X_train.parquet")
            X_test_pl = pl.read_parquet(f"{base_path}_X_test.parquet")
            y_train_pl = pl.read_parquet(f"{base_path}_y_train.parquet")
            y_test_pl = pl.read_parquet(f"{base_path}_y_test.parquet")
            
            # Convert to pandas
            self.X_train = X_train_pl.to_pandas()
            self.X_test = X_test_pl.to_pandas()
            self.y_train = y_train_pl.to_pandas()
            self.y_test = y_test_pl.to_pandas()
            
            # Extract feature names
            self.feature_names = self.X_test.columns.tolist()
            
            logger.info(f"Data loaded: X_test shape {self.X_test.shape}")
            
            return self.X_test, self.y_test
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None
    
    def create_explainer(self) -> Any:
        """
        Create a SHAP explainer based on the model type.
        
        Returns:
            SHAP explainer object
        """
        if self.model is None:
            self.load_model()
        
        if self.X_test is None:
            self.load_data()
        
        logger.info(f"Creating SHAP explainer for {self.model_name}")
        
        try:
            # Select the appropriate SHAP explainer based on model type
            if self.model_name in ['random_forest', 'gradient_boosting']:
                # Tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_name == 'xgboost':
                # XGBoost - special handling
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_name == 'lightgbm':
                # LightGBM - special handling
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For other models (like logistic regression), use KernelExplainer with background samples
                # Take a smaller sample as background for Kernel explainer (for efficiency)
                train_sample = shap.sample(self.X_train, 100)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, train_sample)
            
            logger.info("SHAP explainer created successfully")
            return self.explainer
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            return None
    
    def compute_shap_values(self, sample_size: Optional[int] = None) -> np.ndarray:
        """
        Compute SHAP values for the test data.
        
        Args:
            sample_size: Optional subset size to compute SHAP values for
                         (useful for large datasets)
        
        Returns:
            Array of SHAP values
        """
        if self.explainer is None:
            self.create_explainer()
        
        if sample_size and sample_size < self.X_test.shape[0]:
            # Take a random subset
            np.random.seed(42)
            idx = np.random.choice(self.X_test.shape[0], sample_size, replace=False)
            X_sample = self.X_test.iloc[idx]
        else:
            X_sample = self.X_test
        
        logger.info(f"Computing SHAP values for {len(X_sample)} samples")
        
        try:
            if isinstance(self.explainer, shap.TreeExplainer):
                # For tree models, directly compute SHAP values
                self.shap_values = self.explainer.shap_values(X_sample)
                
                # Handle different return formats
                if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                    # Binary classification returns a list with two elements
                    # We'll use the second (positive class) for analysis
                    self.shap_values = self.shap_values[1]
            else:
                # For other models like logistic regression
                self.shap_values = self.explainer.shap_values(X_sample)
                # For binary classification, take the positive class
                if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                    self.shap_values = self.shap_values[1]
            
            logger.info("SHAP values computed successfully")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def compute_permutation_importance(self, n_repeats: int = 10) -> pd.DataFrame:
        """
        Compute permutation feature importance.
        
        Args:
            n_repeats: Number of times to permute each feature
            
        Returns:
            DataFrame with permutation importance for each feature
        """
        if self.model is None:
            self.load_model()
            
        if self.X_test is None:
            self.load_data()
            
        logger.info(f"Computing permutation importance with {n_repeats} repeats")
        
        try:
            # Use ROC AUC as the scoring metric
            result = permutation_importance(
                self.model, self.X_test, self.y_test.values.ravel(),
                n_repeats=n_repeats,
                random_state=42,
                scoring='roc_auc'
            )
            
            # Create a DataFrame with results
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            self.permutation_importances = importance_df
            
            # Save to file
            importance_df.to_csv('reports/permutation_importance.csv', index=False)
            
            logger.info("Permutation importance computed successfully")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error computing permutation importance: {e}")
            return None
    
    def plot_summary(self) -> None:
        """Plot SHAP summary plot showing feature importance."""
        if self.shap_values is None:
            self.compute_shap_values()
        
        logger.info("Creating SHAP summary plot")
        
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                plot_type="bar",
                max_display=20,
                show=False
            )
            plt.tight_layout()
            plt.savefig('reports/figures/explanations/shap/feature_importance_summary.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create dot summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                plot_type="dot",
                max_display=20,
                show=False
            )
            plt.tight_layout()
            plt.savefig('reports/figures/explanations/shap/feature_impact_summary.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("SHAP summary plots created")
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
    
    def plot_dependence(self, feature_name: str, interaction_feature: Optional[str] = None) -> None:
        """
        Plot SHAP dependence plot showing how a feature affects predictions.
        
        Args:
            feature_name: Feature to analyze
            interaction_feature: Optional second feature to show interaction with
        """
        if self.shap_values is None:
            self.compute_shap_values()
            
        if feature_name not in self.feature_names:
            logger.warning(f"Feature {feature_name} not found")
            return
            
        logger.info(f"Creating SHAP dependence plot for {feature_name}")
        
        try:
            plt.figure(figsize=(12, 8))
            
            if interaction_feature is None:
                # Let SHAP automatically find the best interaction feature
                shap.dependence_plot(
                    feature_name, 
                    self.shap_values, 
                    self.X_test,
                    show=False
                )
            else:
                shap.dependence_plot(
                    feature_name, 
                    self.shap_values, 
                    self.X_test,
                    interaction_index=interaction_feature,
                    show=False
                )
                
            plt.tight_layout()
            output_name = f"{feature_name}"
            if interaction_feature:
                output_name += f"_with_{interaction_feature}"
                
            plt.savefig(f'reports/figures/explanations/shap/dependence_{output_name}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP dependence plot created for {feature_name}")
            
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")
    
    def plot_partial_dependence(self, feature_names: Union[str, List[str]]) -> None:
        """
        Create partial dependence plots to show how features affect predictions.
        
        Args:
            feature_names: Feature or list of features to analyze
        """
        if self.model is None:
            self.load_model()
            
        if self.X_train is None:
            self.load_data()
            
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            
        valid_features = [f for f in feature_names if f in self.feature_names]
        if not valid_features:
            logger.warning(f"None of the features {feature_names} found")
            return
            
        logger.info(f"Creating partial dependence plots for {valid_features}")
        
        try:
            for feature in valid_features:
                # Get feature index
                feature_idx = self.feature_names.index(feature)
                
                # Calculate partial dependence
                pdp_result = partial_dependence(
                    self.model, 
                    self.X_train, 
                    features=[feature_idx],
                    kind='average'
                )
                
                # Plot
                plt.figure(figsize=(10, 6))
                
                # Extract values from the result
                feature_values = pdp_result['values'][0]
                pdp_values = pdp_result['average'][0]
                
                plt.plot(feature_values, pdp_values)
                plt.xlabel(feature)
                plt.ylabel('Partial Dependence')
                plt.title(f'Partial Dependence Plot for {feature}')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'reports/figures/explanations/pdp/partial_dependence_{feature}.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            logger.info(f"Partial dependence plots created for {valid_features}")
            
        except Exception as e:
            logger.error(f"Error creating partial dependence plots: {e}")
    
    def analyze_feature_interactions(self, top_n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze feature interactions based on SHAP interaction values.
        
        Args:
            top_n: Number of top interactions to return for each feature
            
        Returns:
            Dictionary of feature interactions
        """
        if self.model is None:
            self.load_model()
            
        if self.X_test is None:
            self.load_data()
            
        if not hasattr(self, 'shap_interaction_values'):
            logger.info("Computing SHAP interaction values (this may take a while)...")
            
            try:
                # For efficiency, use a small sample of the test data
                sample_size = min(100, self.X_test.shape[0])
                np.random.seed(42)
                idx = np.random.choice(self.X_test.shape[0], sample_size, replace=False)
                X_sample = self.X_test.iloc[idx]
                
                # For tree models, we can compute interaction values
                if isinstance(self.explainer, shap.TreeExplainer):
                    self.shap_interaction_values = self.explainer.shap_interaction_values(X_sample)
                    
                    # Depending on the model, we may need to handle different formats
                    if isinstance(self.shap_interaction_values, list):
                        # For binary classification, take the positive class
                        self.shap_interaction_values = self.shap_interaction_values[1]
                else:
                    logger.warning("Interaction values only supported for tree-based models")
                    return {}
                
                logger.info("SHAP interaction values computed successfully")
                
            except Exception as e:
                logger.error(f"Error computing SHAP interaction values: {e}")
                return {}
        
        # Analyze interactions
        try:
            # Sum the absolute interaction values across all samples
            interaction_strength = np.abs(self.shap_interaction_values).sum(axis=0)
            
            # Remove the diagonal (self-interactions)
            np.fill_diagonal(interaction_strength, 0)
            
            # Create a dictionary to store the top interactions for each feature
            interactions = {}
            
            for i, feature in enumerate(self.feature_names):
                # Get all interactions with this feature
                feature_interactions = []
                
                for j, other_feature in enumerate(self.feature_names):
                    if i != j:  # Skip self-interactions
                        # Take the maximum of the interaction strength in both directions
                        strength = max(interaction_strength[i, j], interaction_strength[j, i])
                        feature_interactions.append((other_feature, strength))
                
                # Sort by interaction strength
                feature_interactions.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only the top_n interactions
                interactions[feature] = feature_interactions[:top_n]
            
            # Create a visualization of the top overall interactions
            self._plot_top_interactions(interaction_strength)
            
            logger.info(f"Feature interactions analyzed for {len(interactions)} features")
            return interactions
            
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {e}")
            return {}
    
    def _plot_top_interactions(self, interaction_matrix: np.ndarray, top_n: int = 10) -> None:
        """
        Plot the top feature interactions as a heatmap.
        
        Args:
            interaction_matrix: Matrix of interaction strengths
            top_n: Number of top features to include
        """
        try:
            # Get the sum of interaction strengths for each feature
            feature_importance = interaction_matrix.sum(axis=1)
            
            # Get the indices of the top features
            top_indices = np.argsort(feature_importance)[-top_n:]
            
            # Get the names of the top features
            top_features = [self.feature_names[i] for i in top_indices]
            
            # Extract the interaction matrix for the top features
            top_matrix = interaction_matrix[np.ix_(top_indices, top_indices)]
            
            # Create a heatmap
            plt.figure(figsize=(12, 10))
            
            # Use a sequential colormap for the heatmap
            cmap = sns.color_palette("viridis", as_cmap=True)
            
            sns.heatmap(
                top_matrix,
                xticklabels=top_features,
                yticklabels=top_features,
                cmap=cmap,
                annot=True,
                fmt=".0f",
                linewidths=0.5
            )
            
            plt.title('Top Feature Interactions', fontsize=15)
            plt.tight_layout()
            plt.savefig('reports/figures/explanations/interactions/top_interactions.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Top interactions plot created")
            
        except Exception as e:
            logger.error(f"Error plotting top interactions: {e}")
    
    def create_explanation_report(self, sample_idx: int) -> Dict[str, Any]:
        """
        Create a comprehensive explanation report for a single sample.
        
        Args:
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.compute_shap_values()
            
        if sample_idx >= len(self.X_test):
            logger.warning(f"Sample index {sample_idx} is out of range")
            return {}
            
        logger.info(f"Creating explanation report for sample {sample_idx}")
        
        try:
            # Get the sample and its SHAP values
            sample = self.X_test.iloc[sample_idx]
            sample_shap_values = self.shap_values[sample_idx]
            
            # Get the actual prediction
            y_true = self.y_test.iloc[sample_idx].values[0]
            y_pred_proba = self.model.predict_proba(sample.values.reshape(1, -1))[0, 1]
            y_pred = int(y_pred_proba >= 0.5)
            
            # Create the explanation dictionary
            explanation = {
                "sample_id": int(sample_idx),
                "prediction": {
                    "probability": float(y_pred_proba),
                    "predicted_class": int(y_pred),
                    "actual_class": int(y_true),
                    "correct": y_pred == y_true
                },
                "feature_values": sample.to_dict(),
                "top_features": []
            }
            
            # Get the feature contributions
            feature_contributions = []
            for i, feature_name in enumerate(self.feature_names):
                feature_contributions.append({
                    "feature": feature_name,
                    "value": sample[feature_name],
                    "shap_value": float(sample_shap_values[i]),
                    "abs_shap_value": float(abs(sample_shap_values[i]))
                })
            
            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: x["abs_shap_value"], reverse=True)
            
            # Add top 10 features to the explanation
            explanation["top_features"] = feature_contributions[:10]
            
            # Separate positive and negative contributions
            positive_features = [f for f in feature_contributions if f["shap_value"] > 0]
            negative_features = [f for f in feature_contributions if f["shap_value"] < 0]
            
            # Sort by SHAP value
            positive_features.sort(key=lambda x: x["shap_value"], reverse=True)
            negative_features.sort(key=lambda x: x["shap_value"])
            
            explanation["top_positive_features"] = positive_features[:5]
            explanation["top_negative_features"] = negative_features[:5]
            
            # Create visualizations for this sample
            
            # 1. Force plot 
            plt.figure(figsize=(14, 4))
            shap.force_plot(
                self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                else self.explainer.expected_value,
                sample_shap_values,
                sample,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            force_plot_path = f'reports/figures/explanations/shap/sample_{sample_idx}_force_plot.png'
            plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Decision plot
            plt.figure(figsize=(10, 16))
            shap.decision_plot(
                self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list)
                else self.explainer.expected_value,
                sample_shap_values,
                sample,
                feature_display_range=20,
                show=False
            )
            plt.tight_layout()
            decision_plot_path = f'reports/figures/explanations/shap/sample_{sample_idx}_decision_plot.png'
            plt.savefig(decision_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add paths to the explanation
            explanation["visualizations"] = {
                "force_plot": force_plot_path,
                "decision_plot": decision_plot_path
            }
            
            logger.info(f"Explanation report created for sample {sample_idx}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error creating explanation report: {e}")
            return {}
    
    def run_all_explanations(self, n_samples: int = 5) -> None:
        """
        Run all explanation methods and generate a comprehensive report.
        
        Args:
            n_samples: Number of random samples to explain in detail
        """
        # Load model and data if not already loaded
        if self.model is None:
            self.load_model()
            
        if self.X_test is None:
            self.load_data()
        
        logger.info("Running comprehensive model explanation")
        
        # 1. Create SHAP explainer and compute SHAP values
        self.create_explainer()
        self.compute_shap_values(sample_size=500)  # Limit to 500 samples for efficiency
        
        # 2. Generate permutation feature importance
        perm_importance = self.compute_permutation_importance()
        
        # 3. Create summary plots
        self.plot_summary()
        
        # 4. Create dependence plots for top 10 features
        top_features = self.permutation_importances['feature'].values[:10]
        for feature in top_features:
            self.plot_dependence(feature)
        
        # 5. Create partial dependence plots for top 5 features
        for feature in top_features[:5]:
            self.plot_partial_dependence(feature)
        
        # 6. Analyze feature interactions
        interactions = self.analyze_feature_interactions()
        
        # 7. Create detailed explanations for random samples
        explanations = []
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.X_test), min(n_samples, len(self.X_test)), replace=False)
        
        for idx in sample_indices:
            explanation = self.create_explanation_report(idx)
            explanations.append(explanation)
        
        # 8. Save all explanations to a JSON file
        import json
        with open('reports/sample_explanations.json', 'w') as f:
            json.dump(explanations, f, indent=2)
        
        logger.info(f"All explanations completed, {len(explanations)} sample explanations saved")

def main():
    """Main execution function"""
    explainer = ModelExplainer()
    explainer.run_all_explanations()
    
    logger.info("Model explanation completed")

if __name__ == "__main__":
    main() 