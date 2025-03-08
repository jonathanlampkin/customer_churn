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
    # Import and run the dashboard directly
    from src.dashboard.app import main
    main() 