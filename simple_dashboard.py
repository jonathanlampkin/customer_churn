"""
Simple Streamlit dashboard for churn prediction results
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import joblib
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load configuration
import yaml
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Title and introduction
st.title("Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard presents the results of our churn prediction model.
""")

# Load metrics
metrics_path = config['reports']['metrics_path']
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Display metrics in a nice format
    st.header("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    
    with col2:
        st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
    
    with col3:
        st.metric("Model Type", metrics['model_type'])
    
    # Display visualizations
    st.header("Visualizations")
    
    # Feature importance
    importance_path = os.path.join(config['reports']['figures_path'], 'feature_importance.png')
    if os.path.exists(importance_path):
        st.subheader("Feature Importance")
        st.image(importance_path, caption="Top 20 features by importance")
    
    # Confusion matrix
    cm_path = os.path.join(config['reports']['figures_path'], 'confusion_matrix.png')
    if os.path.exists(cm_path):
        st.subheader("Confusion Matrix")
        st.image(cm_path, caption="Confusion matrix on test data")
    
    # Target distribution
    target_dist_path = os.path.join(config['reports']['figures_path'], 'target_distribution.png')
    if os.path.exists(target_dist_path):
        st.subheader("Target Distribution")
        st.image(target_dist_path, caption="Distribution of the target variable")
    
    # Feature importance table
    if 'feature_importance' in metrics:
        st.header("Feature Importance Details")
        
        # Convert to DataFrame for display
        importance_df = pd.DataFrame([
            {"Feature": feature, "Importance": importance}
            for feature, importance in metrics['feature_importance'].items()
        ])
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Display table
        st.dataframe(importance_df)
else:
    st.error("Metrics file not found. Please run the model training first.")

# Optional: Create a simple prediction form
st.header("Make a Prediction")
st.markdown("Enter customer data to predict churn probability:")

# Load the model
model_path = os.path.join(config['models']['save_path'], 'random_forest_model.joblib')
if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    # Simple form with dummy features (you'd need to adapt this to your actual features)
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.slider("Customer Tenure (months)", 0, 60, 24)
        feature2 = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    with col2:
        feature3 = st.slider("Monthly Charges ($)", 0, 200, 70)
        feature4 = st.checkbox("Fiber Optic Internet")
    
    # Make a prediction button
    if st.button("Predict Churn"):
        # This is just a placeholder since we don't know your actual features
        st.info("This is a demonstration - the prediction is random since we don't know the actual model features.")
        
        # Generate a random prediction for demonstration
        import random
        churn_prob = random.random()
        
        # Display result
        st.subheader("Prediction Result:")
        
        if churn_prob > 0.5:
            st.error(f"High churn risk: {churn_prob:.2%} probability")
        else:
            st.success(f"Low churn risk: {churn_prob:.2%} probability")
else:
    st.error("Model file not found. Please run the model training first.")

# Footer
st.markdown("---")
st.markdown("© 2025 Churn Prediction Project")
