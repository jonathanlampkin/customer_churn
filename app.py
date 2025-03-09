"""
Customer Churn Prediction Dashboard
Main entry point for the Streamlit application
"""
import sys
import os
import streamlit as st
import json

# Add the src directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DO NOT set page config here since it's set in the dashboard file
# Let the dashboard handle all Streamlit setup

# Create necessary directories
os.makedirs('reports/metrics', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/interim', exist_ok=True)

# Check if we have the necessary data files
required_files = [
    'reports/metrics/final_model_metrics.json',
    'reports/metrics/feature_importance.csv',
    'reports/metrics/sample_explanations.json'
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    # This check for missing files should use other methods
    # to avoid using st. commands before the page config
    missing_file_str = ', '.join(missing_files)
    
    if __name__ == "__main__":
        # Instead of importing a specific function, just run the dashboard file
        import runpy
        # This will execute the dashboard module as __main__
        runpy.run_module("src.dashboard.app", run_name="__main__")
else:
    if __name__ == "__main__":
        # Instead of importing a specific function, just run the dashboard file
        import runpy
        # This will execute the dashboard module as __main__
        runpy.run_module("src.dashboard.app", run_name="__main__") 