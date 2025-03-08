"""
Complete Machine Learning Pipeline for Churn Prediction
Logs and visualizes each step of the process
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import json

# Set up logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_directories():
    """Create all required directories for the project"""
    dirs = [
        'data/raw',
        'data/interim',
        'data/processed',
        'models',
        'reports/figures',
        'reports/metrics',
        'logs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")

def run_data_processing():
    """Run the data processing pipeline"""
    logger.info("=" * 80)
    logger.info("STARTING DATA PROCESSING")
    logger.info("=" * 80)
    
    from src.data.data_processor import ChurnDataProcessor
    
    # Start timer
    start_time = time.time()
    
    # Initialize processor
    processor = ChurnDataProcessor()
    
    # Load data
    logger.info("Loading raw data...")
    df = processor.load_data()
    
    # Examine data quality
    logger.info("Examining data quality...")
    quality_report = processor.examine_nulls(df)
    processor.identify_exclude_columns(df)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    processed_df = processor.preprocess_data(df)
    
    # Split data
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = processor.split_data(processed_df)
    
    # Log completion
    processing_time = time.time() - start_time
    logger.info(f"Data processing completed in {processing_time:.2f} seconds")
    
    # Save intermediate data for visualization
    X_train.write_csv('data/interim/X_train_sample.csv', n_rows=1000)
    
    return X_train, X_test, y_train, y_test

def run_feature_engineering(X_train, X_test, y_train, y_test):
    """Run the feature engineering pipeline"""
    logger.info("=" * 80)
    logger.info("STARTING FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    from src.features.feature_engineering import FeatureEngineer
    
    # Start timer
    start_time = time.time()
    
    # Initialize engineer
    engineer = FeatureEngineer()
    engineer.set_data(X_train, X_test, y_train, y_test)
    
    # Run the feature engineering pipeline
    logger.info("Running feature engineering pipeline...")
    X_train_eng, X_test_eng = engineer.run_feature_engineering_pipeline()
    
    # Log completion
    engineering_time = time.time() - start_time
    logger.info(f"Feature engineering completed in {engineering_time:.2f} seconds")
    
    return X_train_eng, X_test_eng, y_train, y_test

def run_model_training(X_train, X_test, y_train, y_test):
    """Run the model training pipeline"""
    logger.info("=" * 80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 80)
    
    from src.models.model_trainer import ModelTrainer
    
    # Start timer
    start_time = time.time()
    
    # Initialize trainer
    trainer = ModelTrainer()
    trainer.set_data(X_train, X_test, y_train, y_test)
    
    # Compare different approaches to class imbalance
    logger.info("Comparing class imbalance handling approaches...")
    imbalance_results = trainer.compare_class_imbalance_approaches()
    
    with open('reports/metrics/imbalance_comparison.json', 'w') as f:
        json.dump(imbalance_results, f, indent=4)
    
    # Train and compare models
    logger.info("Training and comparing models...")
    comparison_results = trainer.train_and_compare_models()
    
    with open('reports/metrics/model_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    # Get holistic model metrics (including interpretability, complexity)
    logger.info("Evaluating models holistically...")
    holistic_results = trainer.evaluate_models_holistically()
    
    with open('reports/metrics/holistic_evaluation.json', 'w') as f:
        json.dump(holistic_results, f, indent=4)
    
    # Select best model based on multiple criteria
    logger.info("Selecting best model...")
    best_model_name, selection_criteria = trainer.select_best_model(holistic_results)
    
    logger.info(f"Selected model: {best_model_name}")
    logger.info(f"Selection criteria: {selection_criteria}")
    
    # Fine-tune best model
    logger.info(f"Fine-tuning {best_model_name}...")
    tuning_results = trainer.tune_best_model()
    
    with open('reports/metrics/hyperparameter_tuning.json', 'w') as f:
        json.dump(tuning_results, f, indent=4)
    
    # Train final model
    logger.info("Training final model...")
    final_metrics = trainer.train_final_model()
    
    with open('reports/metrics/final_model_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # Log completion
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return trainer.best_model, final_metrics

def run_model_explanation(model, X_test, y_test):
    """Run the model explanation pipeline"""
    logger.info("=" * 80)
    logger.info("STARTING MODEL EXPLANATION")
    logger.info("=" * 80)
    
    from src.models.model_explainer import ModelExplainer
    
    # Start timer
    start_time = time.time()
    
    # Initialize explainer
    explainer = ModelExplainer()
    explainer.set_model_and_data(model, X_test, y_test)
    
    # Generate explanations
    logger.info("Generating global feature importance...")
    importance_df = explainer.global_feature_importance()
    importance_df.to_csv('reports/metrics/feature_importance.csv', index=False)
    
    logger.info("Generating SHAP explanations...")
    shap_values = explainer.calculate_shap_values()
    
    logger.info("Generating sample explanations...")
    sample_explanations = explainer.generate_sample_explanations(10)
    
    with open('reports/metrics/sample_explanations.json', 'w') as f:
        json.dump(sample_explanations, f, indent=4)
    
    # Log completion
    explanation_time = time.time() - start_time
    logger.info(f"Model explanation completed in {explanation_time:.2f} seconds")

def main():
    """Run the complete pipeline"""
    logger.info("=" * 80)
    logger.info("STARTING COMPLETE ML PIPELINE")
    logger.info("=" * 80)
    
    # Create directories
    create_directories()
    
    # Start timer for the whole pipeline
    pipeline_start_time = time.time()
    
    try:
        # Run data processing
        X_train, X_test, y_train, y_test = run_data_processing()
        
        # Run feature engineering
        X_train_eng, X_test_eng, y_train, y_test = run_feature_engineering(X_train, X_test, y_train, y_test)
        
        # Run model training
        best_model, final_metrics = run_model_training(X_train_eng, X_test_eng, y_train, y_test)
        
        # Run model explanation
        run_model_explanation(best_model, X_test_eng, y_test)
        
        # Log completion
        pipeline_time = time.time() - pipeline_start_time
        logger.info("=" * 80)
        logger.info(f"COMPLETE PIPELINE FINISHED in {pipeline_time:.2f} seconds")
        logger.info("=" * 80)
        logger.info(f"Final model performance: {final_metrics}")
        
        # Generate dashboard visuals
        logger.info("Generating dashboard visuals...")
        from src.dashboard.create_demo_visuals import create_all_visuals
        create_all_visuals()
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 