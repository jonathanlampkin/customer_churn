"""
Streamlit dashboard for monitoring churn prediction models
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yaml
import joblib
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model_trainer import ModelTrainer
from src.models.model_explainer import ModelExplainer

# Load configuration
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Set page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Define functions for dashboard components
def load_model_metadata():
    """Load model metadata from the most recent training run"""
    metadata_dir = 'models/metadata'
    if not os.path.exists(metadata_dir):
        return None
    
    metadata_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('.yml')])
    if not metadata_files:
        return None
    
    latest_metadata_file = os.path.join(metadata_dir, metadata_files[-1])
    with open(latest_metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    return metadata

def load_model_history():
    """Load history of model performance from all training runs"""
    metadata_dir = 'models/metadata'
    if not os.path.exists(metadata_dir):
        return None
    
    metadata_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('.yml')])
    if not metadata_files:
        return None
    
    history = []
    for file in metadata_files:
        with open(os.path.join(metadata_dir, file), 'r') as f:
            metadata = yaml.safe_load(f)
            
            # Extract timestamp and metrics
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            metrics = metadata['metrics']
            
            history.append({
                'timestamp': timestamp,
                **metrics
            })
    
    return pd.DataFrame(history)

def load_explanations():
    """Load sample explanations"""
    explanation_path = 'reports/sample_explanations.json'
    if not os.path.exists(explanation_path):
        return None
    
    with open(explanation_path, 'r') as f:
        explanations = json.load(f)
    
    return explanations

# Dashboard title
st.title("Churn Prediction Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Model Performance", "Model Explanations", "Prediction Simulator"])

# Main content
if page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Load model metadata
    metadata = load_model_metadata()
    history = load_model_history()
    
    if metadata and history is not None:
        # Display latest metrics
        st.subheader("Latest Model Performance")
        
        latest = history.iloc[-1]
        latest_metrics = {k: v for k, v in latest.items() if k != 'timestamp'}
        
        # Create metric display
        cols = st.columns(len(latest_metrics))
        for i, (metric, value) in enumerate(latest_metrics.items()):
            with cols[i]:
                delta = None
                if len(history) > 1:
                    previous = history.iloc[-2][metric]
                    delta = value - previous
                st.metric(label=metric.upper(), value=f"{value:.4f}", delta=f"{delta:.4f}" if delta is not None else None)
        
        # Plot metrics over time
        st.subheader("Performance Trends")
        
        # Select metrics to display
        metrics_to_plot = st.multiselect(
            "Select metrics to display", 
            options=[col for col in history.columns if col != 'timestamp'],
            default=['accuracy', 'roc_auc', 'precision', 'recall']
        )
        
        if metrics_to_plot:
            fig = go.Figure()
            for metric in metrics_to_plot:
                fig.add_trace(go.Scatter(
                    x=history['timestamp'],
                    y=history[metric],
                    mode='lines+markers',
                    name=metric
                ))
            
            fig.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Training Date",
                yaxis_title="Score",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model training history available. Run the pipeline to generate metrics.")
    
    # Display static visualizations
    st.subheader("Model Evaluation Visualizations")
    
    confusion_matrix_path = 'reports/figures/confusion_matrix.png'
    roc_curve_path = 'reports/figures/roc_curve.png'
    pr_curve_path = 'reports/figures/precision_recall_curve.png'
    
    cols = st.columns(3)
    
    if os.path.exists(confusion_matrix_path):
        with cols[0]:
            st.image(confusion_matrix_path, caption="Confusion Matrix")
    
    if os.path.exists(roc_curve_path):
        with cols[1]:
            st.image(roc_curve_path, caption="ROC Curve")
    
    if os.path.exists(pr_curve_path):
        with cols[2]:
            st.image(pr_curve_path, caption="Precision-Recall Curve")
    
    # Feature importance
    feature_importance_path = 'reports/figures/feature_importance.png'
    if os.path.exists(feature_importance_path):
        st.subheader("Feature Importance")
        st.image(feature_importance_path, caption="Top Features by Importance")

elif page == "Model Explanations":
    st.header("Model Explanations")
    
    # Load sample explanations
    explanations = load_explanations()
    
    if explanations:
        st.subheader("Sample Predictions Explained")
        
        # Select a sample to explain
        selected_sample = st.selectbox(
            "Select a sample to explain",
            options=range(len(explanations)),
            format_func=lambda x: f"Sample {explanations[x]['sample_id']} - Predicted: {explanations[x]['prediction']:.2f}, Actual: {explanations[x]['actual_class']}"
        )
        
        explanation = explanations[selected_sample]
        
        cols = st.columns(2)
        
        with cols[0]:
            st.subheader("Prediction Details")
            st.write(f"**Prediction Score:** {explanation['prediction']:.4f}")
            st.write(f"**Actual Class:** {explanation['actual_class']}")
            
            # Display top features
            st.subheader("Top Contributing Factors")
            
            # Positive factors
            st.write("**Factors increasing churn risk:**")
            for factor in explanation['top_positive_features']:
                st.write(f"- {factor['feature']}: {factor['value']} (impact: +{factor['impact']:.4f})")
            
            # Negative factors
            st.write("**Factors decreasing churn risk:**")
            for factor in explanation['top_negative_features']:
                st.write(f"- {factor['feature']}: {factor['value']} (impact: {factor['impact']:.4f})")
        
        with cols[1]:
            # Display SHAP explanation plot
            explanation_plot_path = f"reports/figures/explanations/sample_{explanation['sample_id']}_explanation.png"
            if os.path.exists(explanation_plot_path):
                st.image(explanation_plot_path, caption="SHAP Force Plot")
        
        # Display SHAP summary and dependence plots
        st.subheader("Global Model Explanations")
        
        summary_plot_path = "reports/figures/explanations/summary_plot.png"
        if os.path.exists(summary_plot_path):
            st.image(summary_plot_path, caption="SHAP Summary Plot")
        
        # Dependence plots
        st.subheader("Feature Dependence Plots")
        
        # Find all dependence plots
        dependence_plots = [f for f in os.listdir('reports/figures/explanations') if f.startswith('dependence_')]
        
        if dependence_plots:
            selected_plot = st.selectbox(
                "Select a feature to analyze",
                options=dependence_plots,
                format_func=lambda x: x.replace('dependence_', '').replace('.png', '')
            )
            
            plot_path = f"reports/figures/explanations/{selected_plot}"
            st.image(plot_path)
    else:
        st.info("No model explanations available. Run the explainer to generate explanations.")
        
        if st.button("Generate Explanations"):
            st.write("Generating explanations... This may take a moment.")
            
            with st.spinner("Running model explainer..."):
                explainer = ModelExplainer()
                explainer.run_all_explanations(sample_size=100)
            
            st.success("Explanations generated! Refresh the page to view them.")

elif page == "Prediction Simulator":
    st.header("Churn Prediction Simulator")
    
    # Load the model
    model_files = [f for f in os.listdir('models') if f.startswith('tuned_')]
    
    if model_files:
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join('models', latest_model)
        model = joblib.load(model_path)
        
        # Load a sample record from test data for simulation
        try:
            import polars as pl
            
            # Load model and test data
            interim_path = config['data']['interim_path']
            base_path = os.path.splitext(interim_path)[0]
            X_test = pl.read_parquet(f"{base_path}_X_test.parquet").to_pandas()
            
            if not X_test.empty:
                st.subheader("Enter Customer Information")
                
                # Get a sample record to initialize
                sample_idx = 0
                sample_record = X_test.iloc[sample_idx].to_dict()
                
                # Get feature names from test data
                feature_names = X_test.columns.tolist()
                
                # Simulate input form
                st.write("Enter customer details to predict churn probability:")
                
                # Display key features for input
                top_features = ['feature1', 'feature2', 'feature3']  # Replace with actual top features
                
                # Create an editable form
                with st.form("prediction_form"):
                    # Create input fields for key features
                    input_values = {}
                    
                    for feature in feature_names[:10]:  # Limit to 10 features for simplicity
                        default_value = sample_record[feature]
                        
                        # Determine input type based on value
                        if isinstance(default_value, (int, float)):
                            input_values[feature] = st.number_input(
                                f"{feature}:", 
                                value=float(default_value),
                                format="%.2f" if isinstance(default_value, float) else "%d"
                            )
                        elif isinstance(default_value, str):
                            input_values[feature] = st.text_input(f"{feature}:", value=default_value)
                        else:
                            input_values[feature] = st.text_input(f"{feature}:", value=str(default_value))
                    
                    # Submit button
                    submitted = st.form_submit_button("Predict Churn")
                    
                    if submitted:
                        # Create input dataframe
                        input_df = pd.DataFrame([input_values])
                        
                        # Make prediction
                        prediction = model.predict_proba(input_df)[0, 1]
                        churn_result = "Likely to Churn" if prediction >= 0.5 else "Not Likely to Churn"
                        
                        # Display result
                        st.subheader("Prediction Result")
                        
                        # Show prediction gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prediction,
                            title = {'text': "Churn Probability"},
                            gauge = {
                                'axis': {'range': [0, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 0.3], 'color': "green"},
                                    {'range': [0.3, 0.7], 'color': "yellow"},
                                    {'range': [0.7, 1], 'color': "red"}
                                ]
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**Prediction:** {churn_result} ({prediction:.2%} probability)")
                        
                        # Generate explanation if available
                        if hasattr(model, 'feature_importances_'):
                            # Get feature importances
                            importances = model.feature_importances_
                            feature_importance = pd.DataFrame({
                                'feature': input_df.columns,
                                'importance': importances,
                                'value': [input_df[col].values[0] for col in input_df.columns]
                            }).sort_values('importance', ascending=False)
                            
                            st.subheader("Top Contributing Factors")
                            
                            # Show top factors
                            st.write("Features with the highest impact on this prediction:")
                            for _, row in feature_importance.head(5).iterrows():
                                st.write(f"- {row['feature']}: {row['value']} (importance: {row['importance']:.4f})")
            else:
                st.error("Test data is empty. Please run the pipeline to generate test data.")
        except Exception as e:
            st.error(f"Error loading test data: {e}")
    else:
        st.error("No trained model found. Please run the pipeline to train a model.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard provides insights into the churn prediction model's "
    "performance and explanations for its predictions."
)
st.sidebar.markdown("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M")) 