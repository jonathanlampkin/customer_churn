"""
Create directory structure for the churn prediction project
"""
import os
import sys

def create_project_structure():
    """Creates the directory structure for the project"""
    
    # Base directories
    directories = [
        'data/raw',
        'data/processed',
        'data/interim',
        'models',
        'notebooks',
        'reports/figures',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'configs'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create initial config file
    config_content = """
    # Churn Prediction Project Configuration
    
    # Data paths
    data:
      raw_path: "/mnt/hdd/data/churn/mobile_66kx66_nonull.arff"
      processed_path: "data/processed/churn_processed.parquet"
      interim_path: "data/interim/churn_interim.parquet"
    
    # Model parameters
    models:
      random_seed: 42
      test_size: 0.2
      validation_size: 0.25  # % of training data
      cv_folds: 5
    
    # Training parameters
    training:
      target_column: "churn (target)"
    """
    
    with open('configs/config.yml', 'w') as f:
        f.write(config_content)
    
    print("Project structure created successfully")

if __name__ == "__main__":
    create_project_structure() 