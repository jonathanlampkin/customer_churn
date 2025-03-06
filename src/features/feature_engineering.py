"""
Feature Engineering for Churn Prediction
Implements various feature transformations and creation methods.
"""

import os
import polars as pl
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import yaml
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for feature engineering operations on the churn dataset.
    """
    
    def __init__(self, config_path: Union[str, Path] = 'configs/config.yml'):
        """
        Initialize the feature engineer with configuration.
        
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
        self.transformers = {}
        
        # Create interim data directory if needed
        os.makedirs(os.path.dirname(self.config['data']['interim_path']), exist_ok=True)
        
        logger.info("Feature engineer initialized")
    
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Load preprocessed and split data from disk.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        from src.data.data_processor import ChurnDataProcessor
        
        logger.info("Loading preprocessed data")
        processor = ChurnDataProcessor()
        df = processor.load_data()
        processed_df = processor.preprocess_data(df)
        X_train, X_test, y_train, y_test = processor.split_data(processed_df)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Data loaded: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, method: str = 'standard', X_train: Optional[pl.DataFrame] = None, 
                       X_test: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Scale numerical features using various methods.
        
        Args:
            method: Scaling method ('standard', 'minmax', or 'yeo-johnson')
            X_train: Training features (uses self.X_train if None)
            X_test: Test features (uses self.X_test if None)
            
        Returns:
            Scaled X_train and X_test
        """
        if X_train is None:
            X_train = self.X_train
        
        if X_test is None:
            X_test = self.X_test
        
        if X_train is None or X_test is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Convert to pandas for scaling
        X_train_pd = X_train.to_pandas()
        X_test_pd = X_test.to_pandas()
        
        # Get numeric columns
        numeric_cols = [col for col in X_train_pd.columns 
                       if np.issubdtype(X_train_pd[col].dtype, np.number)]
        
        logger.info(f"Scaling {len(numeric_cols)} numeric features using {method} method")
        
        # Choose scaler based on method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'yeo-johnson':
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit scaler on training data
        X_train_scaled = X_train_pd.copy()
        X_test_scaled = X_test_pd.copy()
        
        # Apply scaling to numeric columns
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_pd[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test_pd[numeric_cols])
        
        # Store the scaler for later use
        self.transformers[f'scaler_{method}'] = (scaler, numeric_cols)
        
        # Convert back to polars
        X_train_scaled_pl = pl.from_pandas(X_train_scaled)
        X_test_scaled_pl = pl.from_pandas(X_test_scaled)
        
        logger.info(f"Scaling complete using {method} method")
        
        return X_train_scaled_pl, X_test_scaled_pl
    
    def run_feature_engineering_pipeline(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Run the complete feature engineering pipeline.
        
        Returns:
            X_train and X_test with engineered features
        """
        # Make sure data is loaded
        if self.X_train is None or self.X_test is None:
            self.load_data()
        
        logger.info("Starting feature engineering pipeline")
        
        # 1. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(method='standard')
        
        # 2. Select important features
        # Use SelectKBest to get the top features
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Convert to pandas for sklearn
        X_train_pd = X_train_scaled.to_pandas()
        y_train_pd = self.y_train.to_pandas()
        X_test_pd = X_test_scaled.to_pandas()
        
        # Get numeric columns
        numeric_cols = [col for col in X_train_pd.columns 
                      if np.issubdtype(X_train_pd[col].dtype, np.number)]
        
        # Select top features
        selector = SelectKBest(f_classif, k=min(40, len(numeric_cols)))
        selector.fit(X_train_pd[numeric_cols], y_train_pd)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [numeric_cols[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_features)} important features")
        
        # Store the feature selector
        self.transformers['feature_selector'] = (selector, numeric_cols)
        
        # 3. Add cluster features
        # Create a temporary dataset with only the selected features for clustering
        X_train_selected = X_train_pd[selected_features]
        X_test_selected = X_test_pd[selected_features]
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        
        # Get cluster assignments
        X_train_pd['cluster'] = kmeans.fit_predict(X_train_selected)
        X_test_pd['cluster'] = kmeans.predict(X_test_selected)
        
        # Calculate distances to cluster centers
        train_distances = kmeans.transform(X_train_selected)
        test_distances = kmeans.transform(X_test_selected)
        
        # Add distance features
        for i in range(5):
            X_train_pd[f'distance_to_cluster_{i}'] = train_distances[:, i]
            X_test_pd[f'distance_to_cluster_{i}'] = test_distances[:, i]
        
        logger.info("Added cluster-based features")
        
        # Store the clustering model
        self.transformers['kmeans'] = (kmeans, selected_features)
        
        # 4. Reduce dimensions using PCA for feature extraction
        pca = PCA(n_components=5, random_state=42)
        pca_features_train = pca.fit_transform(X_train_selected)
        pca_features_test = pca.transform(X_test_selected)
        
        # Add PCA features
        for i in range(5):
            X_train_pd[f'pca_component_{i+1}'] = pca_features_train[:, i]
            X_test_pd[f'pca_component_{i+1}'] = pca_features_test[:, i]
        
        logger.info("Added PCA components as features")
        
        # Store the PCA transformer
        self.transformers['pca'] = (pca, selected_features)
        
        # Convert back to polars
        X_train_engineered = pl.from_pandas(X_train_pd)
        X_test_engineered = pl.from_pandas(X_test_pd)
        
        # Save the engineered datasets
        interim_path = self.config['data']['interim_path']
        base_path = os.path.splitext(interim_path)[0]
        
        X_train_engineered.write_parquet(f"{base_path}_X_train.parquet")
        X_test_engineered.write_parquet(f"{base_path}_X_test.parquet")
        self.y_train.write_parquet(f"{base_path}_y_train.parquet")
        self.y_test.write_parquet(f"{base_path}_y_test.parquet")
        
        logger.info(f"Saved engineered datasets to {os.path.dirname(interim_path)}")
        logger.info("Feature engineering pipeline completed")
        
        return X_train_engineered, X_test_engineered

def main():
    """Main execution function"""
    engineer = FeatureEngineer()
    X_train, X_test = engineer.run_feature_engineering_pipeline()
    
    logger.info(f"Engineered datasets: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

if __name__ == "__main__":
    main() 