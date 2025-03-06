"""
Main execution script for the Churn Prediction Project.
This script orchestrates the entire end-to-end pipeline.
"""

import os
import logging
import time
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('churn_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_project_structure():
    """Create the project directory structure and config file."""
    from project_structure import create_project_structure
    logger.info("Creating project structure...")
    create_project_structure()

def run_data_processing():
    """Run data processing and preprocessing."""
    from src.data.data_processor import ChurnDataProcessor
    
    logger.info("Starting data processing...")
    processor = ChurnDataProcessor()
    df = processor.load_data()
    processed_df = processor.preprocess_data(df)
    X_train, X_test, y_train, y_test = processor.split_data(processed_df)
    
    logger.info("Data processing completed")
    return X_train, X_test, y_train, y_test

def run_eda():
    """Run exploratory data analysis."""
    from src.visualization.eda import ChurnEDA
    
    logger.info("Starting exploratory data analysis...")
    eda = ChurnEDA()
    eda.run_all_analyses()
    
    logger.info("EDA completed")

def run_feature_engineering():
    """Run feature engineering pipeline."""
    from src.features.feature_engineering import FeatureEngineer
    
    logger.info("Starting feature engineering...")
    engineer = FeatureEngineer()
    X_train, X_test = engineer.run_feature_engineering_pipeline()
    
    logger.info("Feature engineering completed")
    return X_train, X_test

def run_model_training():
    """Run model training and evaluation."""
    from src.models.model_trainer import ModelTrainer
    
    logger.info("Starting model training and evaluation...")
    trainer = ModelTrainer()
    metrics = trainer.run_model_pipeline()
    
    logger.info("Model training and evaluation completed")
    return metrics

def generate_report(metrics):
    """Generate final report with conclusions."""
    from src.reports.conclusion_generator import generate_conclusion
    
    logger.info("Generating final report...")
    conclusion = generate_conclusion(metrics)
    
    logger.info("Report generation completed")
    return conclusion

def main():
    """Main execution function that runs the entire pipeline."""
    start_time = time.time()
    
    logger.info("Starting Churn Prediction Pipeline")
    
    # 1. Create project structure (if not exists)
    if not os.path.exists('configs/config.yml'):
        create_project_structure()
    
    # 2. Run data processing
    run_data_processing()
    
    # 3. Run exploratory data analysis
    run_eda()
    
    # 4. Run feature engineering
    run_feature_engineering()
    
    # 5. Run model training and evaluation
    metrics = run_model_training()
    
    # 6. Generate conclusion report
    conclusion = generate_report(metrics)
    
    # Print conclusion
    logger.info("\n" + "="*80)
    logger.info("CHURN PREDICTION PROJECT CONCLUSION")
    logger.info("="*80)
    logger.info(conclusion)
    logger.info("="*80)
    
    execution_time = time.time() - start_time
    logger.info(f"Pipeline completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    main() 