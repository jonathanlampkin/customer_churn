"""
Integration tests for model training and inference pipeline
"""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
import joblib
from pathlib import Path

from src.data.data_processor import ChurnDataProcessor
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer

class TestModelIntegration:
    """Integration tests for the entire model pipeline"""
    
    @pytest.fixture
    def setup_test_environment(self):
        """Set up a test environment with temporary directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directories
            data_dir = os.path.join(temp_dir, 'data')
            model_dir = os.path.join(temp_dir, 'models')
            report_dir = os.path.join(temp_dir, 'reports')
            
            os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'interim'), exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(os.path.join(report_dir, 'figures'), exist_ok=True)
            
            # Create test config
            config = {
                'data': {
                    'raw_path': os.path.join(data_dir, 'raw', 'test_data.arff'),
                    'processed_path': os.path.join(data_dir, 'processed', 'test_processed.parquet'),
                    'interim_path': os.path.join(data_dir, 'interim', 'test_interim.parquet')
                },
                'models': {
                    'random_seed': 42,
                    'test_size': 0.2,
                    'validation_size': 0.25,
                    'cv_folds': 5
                },
                'training': {
                    'target_column': 'churn'
                }
            }
            
            config_path = os.path.join(temp_dir, 'test_config.yml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Create test data
            np.random.seed(42)
            n_samples = 500
            
            # Generate synthetic features
            X = pd.DataFrame({
                'feature1': np.random.normal(0, 1, n_samples),
                'feature2': np.random.normal(0, 1, n_samples),
                'feature3': np.random.normal(0, 1, n_samples),
                'feature4': np.random.choice(['A', 'B', 'C'], n_samples),
                'feature5': np.random.choice([True, False], n_samples),
                'feature6': np.random.uniform(0, 100, n_samples),
                'feature7': np.random.uniform(0, 1, n_samples),
                'feature8': np.random.poisson(5, n_samples)
            })
            
            # Generate target with some patterns
            y_prob = 1 / (1 + np.exp(-(
                -1.5 + 
                0.5 * X['feature1'] + 
                -0.8 * X['feature2'] + 
                0.3 * X['feature3'] + 
                0.7 * (X['feature4'] == 'A') +
                0.4 * X['feature5'].astype(int) +
                0.01 * X['feature6'] +
                -1.2 * X['feature7'] +
                0.1 * X['feature8']
            )))
            
            y = pd.DataFrame({
                'churn': np.random.binomial(1, y_prob)
            })
            
            # Combine X and y
            data = pd.concat([X, y], axis=1)
            
            # Save as ARFF format
            from scipy.io import arff
            import tempfile
            
            # We'll create a CSV first, then convert to ARFF
            csv_path = os.path.join(data_dir, 'raw', 'test_data.csv')
            data.to_csv(csv_path, index=False)
            
            # Create a simple ARFF header
            arff_content = ["@RELATION test_data\n"]
            
            # Add attributes
            for col in X.columns:
                if data[col].dtype == 'float64' or data[col].dtype == 'int64':
                    arff_content.append(f"@ATTRIBUTE {col} NUMERIC")
                elif data[col].dtype == 'bool':
                    arff_content.append(f"@ATTRIBUTE {col} {{True, False}}")
                else:
                    unique_values = data[col].unique()
                    values_str = ', '.join(map(str, unique_values))
                    arff_content.append(f"@ATTRIBUTE {col} {{{values_str}}}")
            
            # Add target
            arff_content.append(f"@ATTRIBUTE churn {{0, 1}}\n")
            
            # Add data section header
            arff_content.append("@DATA")
            
            # Write the ARFF file
            with open(config['data']['raw_path'], 'w') as f:
                f.write('\n'.join(arff_content) + '\n')
                
                # Write data rows
                for _, row in data.iterrows():
                    values = []
                    for val in row:
                        if isinstance(val, str):
                            values.append(f"'{val}'")
                        else:
                            values.append(str(val))
                    f.write(','.join(values) + '\n')
                
            return config_path, temp_dir
    
    def test_end_to_end_pipeline(self, setup_test_environment, monkeypatch):
        """Test the entire pipeline from data processing to model evaluation"""
        config_path, temp_dir = setup_test_environment
        
        # Patch the file paths to use our temporary environment
        def mock_open(*args, **kwargs):
            # If trying to open the default config, return our test config
            if args[0] == 'configs/config.yml' and 'r' in kwargs.get('mode', 'r'):
                return open(config_path, 'r')
            return open(*args, **kwargs)
        
        # Apply the patch
        monkeypatch.setattr('builtins.open', mock_open)
        
        # 1. Run data processing
        processor = ChurnDataProcessor(config_path=config_path)
        df = processor.load_data()
        processed_df = processor.preprocess_data(df)
        X_train, X_test, y_train, y_test = processor.split_data(processed_df)
        
        # Assert the data splits are correct
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        # 2. Run feature engineering
        engineer = FeatureEngineer(config_path=config_path)
        engineer.X_train = X_train
        engineer.X_test = X_test
        engineer.y_train = y_train
        engineer.y_test = y_test
        
        X_train_eng, X_test_eng = engineer.run_feature_engineering_pipeline()
        
        # Assert feature engineering produced results
        assert X_train_eng is not None
        assert X_test_eng is not None
        assert X_train_eng.shape[1] >= X_train.shape[1]  # Should have added features
        
        # 3. Run model training (with reduced parameters for faster tests)
        trainer = ModelTrainer(config_path=config_path)
        trainer.X_train = X_train_eng
        trainer.X_test = X_test_eng
        trainer.y_train = y_train
        trainer.y_test = y_test
        
        # Train simplified models for testing
        trainer.train_baseline_models()
        
        # Check that models were trained
        assert len(trainer.models) > 0
        
        # Evaluate models
        metrics = trainer.evaluate_models()
        
        # Assert metrics were produced for each model
        assert metrics is not None
        assert len(metrics) == len(trainer.models)
        
        # Check that we got a best model
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None
        
        # 4. Test prediction on the best model
        # Take a sample from test data
        import polars as pl
        X_test_pd = X_test_eng.to_pandas()
        sample = X_test_pd.iloc[[0]]
        
        # Make a prediction
        prediction = trainer.best_model.predict_proba(sample)[0, 1]
        
        # Check prediction is valid
        assert 0 <= prediction <= 1
        
        # Make sure we can save and load the model
        model_path = os.path.join(temp_dir, 'models', 'test_model.joblib')
        joblib.dump(trainer.best_model, model_path)
        
        # Load the model back
        loaded_model = joblib.load(model_path)
        
        # Make prediction with loaded model
        loaded_prediction = loaded_model.predict_proba(sample)[0, 1]
        
        # Check predictions match
        assert prediction == loaded_prediction
        
        # 5. Test model evaluation
        eval_metrics = trainer.evaluate_best_model()
        
        # Check evaluation produced metrics
        assert eval_metrics is not None
        assert 'accuracy' in eval_metrics
        assert 'roc_auc' in eval_metrics
        
        # Verify all metrics are in valid range
        for metric, value in eval_metrics.items():
            assert 0 <= value <= 1 