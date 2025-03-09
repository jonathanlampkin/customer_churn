"""
Customer Churn Prediction Dashboard - Streamlit Entry Point
"""
import streamlit as st

# Set page config as the first command
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then run the actual app
import main  # This will run your main.py file 