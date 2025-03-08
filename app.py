"""
Customer Churn Prediction Dashboard
Main entry point for the Streamlit application
"""
import sys
import os
import streamlit as st

# Add the src directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure the page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if __name__ == "__main__":
    # Instead of importing a specific function, just run the dashboard file
    import runpy
    # This will execute the dashboard module as __main__
    runpy.run_module("src.dashboard.app", run_name="__main__") 