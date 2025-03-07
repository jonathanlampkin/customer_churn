"""
Streamlit dashboard for visualizing churn prediction results
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yaml
import joblib
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model_explainer import ModelExplainer
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E88E5;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 500;
        color: #333;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-importance-item {
        padding: 5px;
        margin: 2px 0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

def load_config(config_path: str = 'configs/config.yml') -> dict:
    """Load project configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_metrics():
    """Load model metrics from the evaluation results"""
    try:
        metrics_path = "reports/model_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model metrics: {e}")
        return None

def load_feature_importance():
    """Load feature importance data"""
    try:
        importance_path = "reports/feature_importance.csv"
        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
            return importance_df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")
        return None

def load_explanation_samples():
    """Load sample explanations"""
    try:
        explanation_path = "reports/sample_explanations.json"
        if os.path.exists(explanation_path):
            with open(explanation_path, 'r') as f:
                explanations = json.load(f)
            return explanations
        else:
            return None
    except Exception as e:
        st.error(f"Error loading explanations: {e}")
        return None

def display_model_metrics(metrics):
    """Display model performance metrics"""
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    
    if not metrics:
        st.warning("No model metrics available.")
        return
    
    best_model = metrics.get('best_model', '')
    best_metrics = metrics.get('metrics', {})
    
    if not best_metrics:
        st.warning("No metrics available for the best model.")
        return
    
    # Create a row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
    
    with col2:
        st.metric("ROC AUC", f"{best_metrics.get('roc_auc', 0):.4f}")
    
    with col3:
        st.metric("Precision", f"{best_metrics.get('precision', 0):.4f}")
    
    with col4:
        st.metric("Recall", f"{best_metrics.get('recall', 0):.4f}")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    all_models = metrics.get('metrics', {})
    if all_models:
        # Prepare data for plotting
        model_names = list(all_models.keys())
        accuracy = [all_models[m].get('accuracy', 0) for m in model_names]
        roc_auc = [all_models[m].get('roc_auc', 0) for m in model_names]
        precision = [all_models[m].get('precision', 0) for m in model_names]
        recall = [all_models[m].get('recall', 0) for m in model_names]
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'Precision': precision,
            'Recall': recall
        })
        
        # Plot using plotly
        fig = go.Figure()
        
        fig.add_trace(go.Bar(x=model_names, y=accuracy, name='Accuracy', marker_color='#1f77b4'))
        fig.add_trace(go.Bar(x=model_names, y=roc_auc, name='ROC AUC', marker_color='#ff7f0e'))
        fig.add_trace(go.Bar(x=model_names, y=precision, name='Precision', marker_color='#2ca02c'))
        fig.add_trace(go.Bar(x=model_names, y=recall, name='Recall', marker_color='#d62728'))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display comparison table
        st.dataframe(comparison_df.set_index('Model').style.format("{:.4f}"), use_container_width=True)

def display_feature_importance(importance_df):
    """Display feature importance visualization"""
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    
    if importance_df is None:
        st.warning("No feature importance data available.")
        return
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Display top 15 features
    top_features = importance_df.head(15)
    
    fig = px.bar(
        top_features, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Top 15 Features by Importance',
        color='Importance',
        color_continuous_scale='viridis',
    )
    
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature impact explanation
    st.subheader("Feature Impact Analysis")
    st.write("""
    The chart above shows the most important features in predicting customer churn. 
    Features at the top have the strongest influence on the model's predictions.
    """)
    
    # Show importance table with descriptions
    feature_descriptions = {
        # Add your feature descriptions here
        # Example: "feature_name": "Description of what this feature means"
    }
    
    # Add descriptions if available
    if feature_descriptions:
        importance_df['Description'] = importance_df['Feature'].map(
            lambda x: feature_descriptions.get(x, "No description available")
        )
    
    st.dataframe(
        importance_df[['Feature', 'Importance'] + (['Description'] if feature_descriptions else [])].head(15),
        use_container_width=True
    )

def display_sample_explanations(explanations):
    """Display sample prediction explanations"""
    st.markdown('<div class="section-header">Prediction Explanations</div>', unsafe_allow_html=True)
    
    if not explanations:
        st.warning("No sample explanations available.")
        return
    
    st.write("These examples show how the model makes predictions for specific customers.")
    
    # Select which sample to display
    sample_idx = st.selectbox("Select a sample to explain:", 
                             range(len(explanations)), 
                             format_func=lambda i: f"Sample {i+1} - " + 
                             ("Churned" if explanations[i]['prediction']['actual_class'] == 1 else "Did not churn"))
    
    if sample_idx is not None and 0 <= sample_idx < len(explanations):
        explanation = explanations[sample_idx]
        
        # Sample information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction")
            
            actual = explanation['prediction']['actual_class']
            predicted = explanation['prediction']['predicted_class']
            probability = explanation['prediction']['probability']
            correct = explanation['prediction']['correct']
            
            st.markdown(f"""
            - **Actual outcome**: {'Churned' if actual == 1 else 'Did not churn'}
            - **Predicted outcome**: {'Churned' if predicted == 1 else 'Did not churn'} (Confidence: {probability:.2%})
            - **Prediction was**: {'Correct ✓' if correct else 'Incorrect ✗'}
            """)
        
        with col2:
            st.subheader("Key Factors")
            
            # Display top positive and negative features
            pos_features = explanation.get('top_positive_features', [])
            neg_features = explanation.get('top_negative_features', [])
            
            if pos_features:
                st.markdown("**Factors increasing churn probability:**")
                for i, feature in enumerate(pos_features[:3]):
                    st.markdown(f"""
                    <div class="feature-importance-item" style="background-color: rgba(255,152,150,{min(abs(feature['shap_value'])*5, 0.9)})">
                        {i+1}. {feature['feature']}: {feature['value']}
                    </div>
                    """, unsafe_allow_html=True)
            
            if neg_features:
                st.markdown("**Factors decreasing churn probability:**")
                for i, feature in enumerate(neg_features[:3]):
                    st.markdown(f"""
                    <div class="feature-importance-item" style="background-color: rgba(152,223,138,{min(abs(feature['shap_value'])*5, 0.9)})">
                        {i+1}. {feature['feature']}: {feature['value']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Visualization
        st.subheader("Visualization of Feature Impacts")
        
        # Force plot image
        if 'visualizations' in explanation and 'force_plot' in explanation['visualizations']:
            force_plot_path = explanation['visualizations']['force_plot']
            if os.path.exists(force_plot_path):
                st.image(force_plot_path, caption="Force Plot - Shows how each feature contributes to the prediction")
        
        # Decision plot
        if 'visualizations' in explanation and 'decision_plot' in explanation['visualizations']:
            decision_plot_path = explanation['visualizations']['decision_plot']
            if os.path.exists(decision_plot_path):
                st.image(decision_plot_path, caption="Decision Plot - Shows the path to the final prediction")

def main():
    # Header
    st.markdown('<div class="main-header">Churn Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This dashboard provides insights into customer churn prediction, showing model performance,
    feature importance, and example explanations of predictions. Use the sidebar to navigate.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Performance", "Feature Importance", "Prediction Explanations"])
    
    # Load data
    metrics = load_model_metrics()
    importance_df = load_feature_importance()
    explanations = load_explanation_samples()
    
    # Display based on selection
    if page == "Model Performance":
        display_model_metrics(metrics)
    
    elif page == "Feature Importance":
        display_feature_importance(importance_df)
    
    elif page == "Prediction Explanations":
        display_sample_explanations(explanations)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard visualizes the results of a machine learning model "
        "trained to predict customer churn."
    )

if __name__ == "__main__":
    main() 