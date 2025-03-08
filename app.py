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

# Configure the page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.error("Required data files are missing. The ML pipeline needs to be run first.")
    st.error(f"Missing files: {', '.join(missing_files)}")
    
    if st.button("Run Pipeline Now"):
        import subprocess
        try:
            with st.spinner("Running ML pipeline..."):
                result = subprocess.run(["python", "src/run_pipeline.py"], 
                                      capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Pipeline completed successfully!")
                st.experimental_rerun()
            else:
                st.error("Pipeline failed. Check the error below:")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")
    
    st.stop()  # Don't continue with the app if files are missing

if __name__ == "__main__":
    # Instead of importing a specific function, just run the dashboard file
    import runpy
    # This will execute the dashboard module as __main__
    runpy.run_module("src.dashboard.app", run_name="__main__") 