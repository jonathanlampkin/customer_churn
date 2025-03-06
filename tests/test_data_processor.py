"""
Tests for the data processor module
"""

import os
import pytest
import polars as pl
import numpy as np
import tempfile
import yaml
from src.data.data_processor import ChurnDataProcessor

class TestChurnDataProcessor:
    """Test cases for ChurnDataProcessor class"""
    
    @pytest.fixture
    def setup_config(self):
        """Create a temporary config file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as temp_file:
            # Create test config
            config = {
                'data': {
                    'raw_path': 'path/to/test_data.arff',
                    'processed_path': 'path/to/processed_data.parquet',
                    'interim_path': 'path/to/interim_data.parquet'
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
            
            yaml.dump(config, temp_file)
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def mock_processor(self, setup_config, monkeypatch):
        """Create a processor with mocked data loading"""
        
        # Mock the arff.loadarff function
        def mock_loadarff(file_path):
            import pandas as pd
            from scipy.io import arff
            
            # Create mock data
            data = pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100),
                'feature3': np.random.rand(100),
                'churn': np.random.choice([0, 1], size=100)
            })
            
            # Mock meta
            meta = {
                'feature1': {'type': 'numeric'},
                'feature2': {'type': 'numeric'},
                'feature3': {'type': 'numeric'},
                'churn': {'type': 'nominal', 'values': ['0', '1']}
            }
            
            # Convert to format expected by arff loader
            return data.to_records(index=False), meta
        
        # Apply the mock
        monkeypatch.setattr('scipy.io.arff.loadarff', mock_loadarff)
        
        # Create processor
        processor = ChurnDataProcessor(config_path=setup_config)
        
        return processor
    
    def test_load_data(self, mock_processor):
        """Test data loading"""
        df = mock_processor.load_data()
        
        # Check dataframe type and shape
        assert isinstance(df, pl.DataFrame)
        assert df.height == 100
        assert df.width == 4
        assert 'churn' in df.columns
    
    def test_preprocess_data(self, mock_processor):
        """Test data preprocessing"""
        # Load data
        df = mock_processor.load_data()
        
        # Preprocess
        processed_df = mock_processor.preprocess_data(df)
        
        # Check results
        assert isinstance(processed_df, pl.DataFrame)
        assert processed_df.height == df.height
        assert processed_df.null_count().sum() == 0  # No nulls
    
    def test_split_data(self, mock_processor):
        """Test data splitting"""
        # Load and preprocess data
        df = mock_processor.load_data()
        processed_df = mock_processor.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = mock_processor.split_data(processed_df)
        
        # Check splits
        test_size = mock_processor.config['models']['test_size']
        expected_test_size = int(processed_df.height * test_size)
        expected_train_size = processed_df.height - expected_test_size
        
        assert X_test.height == expected_test_size
        assert X_train.height == expected_train_size
        assert y_test.height == expected_test_size
        assert y_train.height == expected_train_size
        
        # Check target column is not in X
        assert mock_processor.config['training']['target_column'] not in X_train.columns
        assert mock_processor.config['training']['target_column'] not in X_test.columns
        
        # Check target column is in y
        assert mock_processor.config['training']['target_column'] in y_train.columns
        assert mock_processor.config['training']['target_column'] in y_test.columns 