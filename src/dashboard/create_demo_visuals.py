"""
Create comprehensive visualizations for the churn dashboard.
This script generates all visualization files needed for the dashboard.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up visualization directories
os.makedirs('reports/figures/demo', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def create_force_plots():
    """Create SHAP force plots for dashboard"""
    logger.info("Creating SHAP force plots...")
    
    # Create high churn risk example
    plt.figure(figsize=(14, 3))
    features = ['Monthly Charges', 'Tenure', 'Contract Type', 'Online Security', 'Tech Support']
    values = [0.231, 0.187, 0.142, -0.118, -0.097]
    colors = ['#ff5757', '#ff5757', '#ff5757', '#5757ff', '#5757ff']
    
    plt.barh(features, values, color=colors)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.title('Feature Impact on Prediction (High Churn Risk)')
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.tight_layout()
    plt.savefig('reports/figures/demo/demo_force_plot_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create low churn risk example
    plt.figure(figsize=(14, 3))
    features = ['Tenure', 'Contract Type', 'Online Security', 'Monthly Charges', 'Tech Support']
    values = [-0.204, -0.132, -0.119, 0.087, 0.053]
    colors = ['#5757ff', '#5757ff', '#5757ff', '#ff5757', '#ff5757']
    
    plt.barh(features, values, color=colors)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.title('Feature Impact on Prediction (Low Churn Risk)')
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.tight_layout()
    plt.savefig('reports/figures/demo/demo_force_plot_2.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_decision_plots():
    """Create SHAP decision plots for dashboard"""
    logger.info("Creating decision plots...")
    
    # Create a simplified decision plot
    plt.figure(figsize=(10, 12))
    
    # Sample data
    features = ['Monthly Charges', 'Tenure', 'Contract Type', 'Online Security', 
                'Tech Support', 'Internet Service', 'Payment Method',
                'Paperless Billing', 'Multiple Lines', 'Online Backup']
    
    # Create a base value
    base_value = 0.3
    
    # Add paths
    x_values = np.linspace(0, 1, 10)
    
    # Path for a high churn customer
    high_churn_path = base_value + np.cumsum(np.array([0.05, 0.12, 0.08, 0.02, 0.03, 0.05, 0.04, 0.02, 0.01, 0.03]))
    
    # Path for a low churn customer
    low_churn_path = base_value - np.cumsum(np.array([0.03, 0.10, 0.06, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01]))
    
    # Plotting
    plt.plot(high_churn_path, range(10), 'r-', linewidth=2.5, label='High Churn Risk')
    plt.plot(low_churn_path, range(10), 'b-', linewidth=2.5, label='Low Churn Risk')
    
    # Add vertical line at base value
    plt.axvline(x=base_value, color='gray', linestyle='--', alpha=0.7)
    
    # Add feature names
    for i, feature in enumerate(features):
        plt.text(0.02, i, feature, fontsize=11, verticalalignment='center')
    
    plt.yticks([])
    plt.xlabel('Prediction Value')
    plt.title('Decision Plot - Path to Final Prediction')
    plt.legend(loc='upper right')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('reports/figures/demo/demo_decision_plot_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second decision plot with different values
    plt.figure(figsize=(10, 12))
    
    # Path for a medium churn customer
    med_churn_path = base_value + np.cumsum(np.array([0.02, -0.03, 0.05, -0.02, 0.04, 0.01, -0.03, 0.04, 0.02, -0.01]))
    
    # Plotting
    plt.plot(high_churn_path, range(10), 'r-', linewidth=2.5, alpha=0.5, label='High Churn Risk')
    plt.plot(med_churn_path, range(10), 'g-', linewidth=2.5, label='Medium Churn Risk')
    plt.plot(low_churn_path, range(10), 'b-', linewidth=2.5, alpha=0.5, label='Low Churn Risk')
    
    # Add vertical line at base value
    plt.axvline(x=base_value, color='gray', linestyle='--', alpha=0.7)
    
    # Add feature names
    for i, feature in enumerate(features):
        plt.text(0.02, i, feature, fontsize=11, verticalalignment='center')
    
    plt.yticks([])
    plt.xlabel('Prediction Value')
    plt.title('Decision Plot - Comparing Different Churn Risks')
    plt.legend(loc='upper right')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('reports/figures/demo/demo_decision_plot_2.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_plots():
    """Create model comparison visualizations"""
    logger.info("Creating model comparison plots...")
    
    # Performance comparison data
    models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    
    # Sample performance data
    data = np.array([
        [0.82, 0.76, 0.71, 0.73, 0.83, 0.81],  # Logistic Regression
        [0.86, 0.79, 0.75, 0.77, 0.88, 0.85],  # Random Forest
        [0.90, 0.83, 0.78, 0.81, 0.91, 0.89],  # XGBoost
        [0.89, 0.82, 0.77, 0.80, 0.90, 0.88],  # LightGBM
        [0.91, 0.86, 0.82, 0.84, 0.93, 0.90]   # Stacking Ensemble
    ])
    
    # Create performance heatmap
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        data, 
        annot=True, 
        cmap='viridis', 
        xticklabels=metrics, 
        yticklabels=models,
        fmt='.2f',
        cbar_kws={'label': 'Score'}
    )
    
    plt.title('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    plt.savefig('reports/figures/demo/model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive Plotly version
    fig = px.imshow(
        data,
        labels=dict(x="Metric", y="Model", color="Score"),
        x=metrics,
        y=models,
        color_continuous_scale='viridis',
        text_auto='.2f',
        aspect="auto"
    )
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metric",
        yaxis_title="Model"
    )
    
    fig.write_html('reports/figures/demo/model_comparison_interactive.html')

def create_eda_visualizations():
    """Create EDA visualizations"""
    logger.info("Creating EDA visualizations...")
    
    # Generate sample churn data for visualization
    n_samples = 1000
    
    # Generate correlated features
    np.random.seed(42)
    tenure = np.random.gamma(2, 10, n_samples)  # Skewed distribution for tenure
    monthly_charges = 50 + 50 * np.random.beta(2, 5, n_samples)  # Distribution of monthly charges
    
    # Generate correlated contract_type
    contract_probs = np.exp(-tenure/50)  # Higher tenure = lower probability of month-to-month
    contract_type = np.random.binomial(1, contract_probs)
    
    # Generate churn based on features
    logits = -0.5 - 0.1 * tenure + 0.05 * monthly_charges + 2 * contract_type
    churn_prob = 1 / (1 + np.exp(-logits))
    churn = np.random.binomial(1, churn_prob)
    
    # Create feature distributions
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(tenure, kde=True)
    plt.title('Distribution of Tenure')
    plt.xlabel('Months')
    
    plt.subplot(2, 2, 2)
    sns.histplot(monthly_charges, kde=True)
    plt.title('Distribution of Monthly Charges')
    plt.xlabel('Amount ($)')
    
    plt.subplot(2, 2, 3)
    sns.countplot(x=contract_type, palette='viridis')
    plt.title('Contract Type')
    plt.xticks([0, 1], ['Long-term', 'Month-to-month'])
    
    plt.subplot(2, 2, 4)
    sns.countplot(x=churn, palette='viridis')
    plt.title('Churn Distribution')
    plt.xticks([0, 1], ['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig('reports/figures/demo/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create churn correlation matrix
    data = {
        'Tenure': tenure,
        'Monthly_Charges': monthly_charges,
        'Contract_Type': contract_type,
        'Churn': churn
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(
        corr, 
        annot=True, 
        mask=mask, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1,
        fmt='.2f',
        linewidths=1
    )
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    plt.savefig('reports/figures/demo/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create class imbalance pie chart
    plt.figure(figsize=(10, 7))
    churn_counts = df['Churn'].value_counts()
    colors = ['#4CAF50', '#FF5252']
    explode = (0, 0.1)
    
    plt.pie(
        churn_counts, 
        explode=explode, 
        labels=['No Churn', 'Churn'],
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        textprops={'fontsize': 14}
    )
    
    plt.axis('equal')
    plt.title('Class Distribution (Imbalanced)', fontsize=16)
    
    plt.savefig('reports/figures/demo/class_imbalance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plots():
    """Create feature importance visualizations"""
    logger.info("Creating feature importance plots...")
    
    # Sample feature importance data
    features = [
        'Monthly Charges', 'Tenure', 'Contract Type', 
        'Online Security', 'Tech Support', 'Internet Service',
        'Payment Method', 'Paperless Billing', 'Multiple Lines',
        'Online Backup'
    ]
    
    importances = [0.23, 0.19, 0.15, 0.11, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02]
    
    # Basic bar chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    bars = plt.barh(features, importances, color=colors)
    plt.xlabel('Importance')
    plt.title('Feature Importance', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add importance values
    for i, v in enumerate(importances):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig('reports/figures/demo/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interactive Plotly version with tooltip details
    feature_descriptions = {
        'Monthly Charges': 'Amount charged to the customer monthly',
        'Tenure': 'Number of months the customer has stayed with the company',
        'Contract Type': 'The type of contract (month-to-month, one year, two year)',
        'Online Security': 'Whether the customer has online security service',
        'Tech Support': 'Whether the customer has tech support service',
        'Internet Service': 'Type of internet service (DSL, Fiber optic, None)',
        'Payment Method': 'The customer\'s payment method',
        'Paperless Billing': 'Whether the customer has paperless billing',
        'Multiple Lines': 'Whether the customer has multiple phone lines',
        'Online Backup': 'Whether the customer has online backup service'
    }
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='viridis',
                colorbar=dict(title='Importance')
            ),
            text=[f"{v:.2f}" for v in importances],
            textposition='auto',
            hovertext=[f"{f}: {feature_descriptions[f]}" for f in features],
            hoverinfo='text'
        )
    )
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600
    )
    
    fig.write_html('reports/figures/demo/feature_importance_interactive.html')

def create_hyperparameter_tuning_plots():
    """Create hyperparameter tuning visualizations"""
    logger.info("Creating hyperparameter tuning plots...")
    
    # Generate sample hyperparameter tuning data
    n_trials = 50
    
    # Trial results (increasing score with noise)
    trials = list(range(1, n_trials + 1))
    
    # Start with a baseline, gradually improve, then plateau with diminishing returns
    baseline = 0.75
    scores = [baseline]
    
    for i in range(1, n_trials):
        if i < 10:
            # Quick improvement phase
            improvement = 0.01 + 0.005 * np.random.random()
        elif i < 30:
            # Slower improvement phase
            improvement = 0.003 + 0.002 * np.random.random() 
        else:
            # Plateau phase
            improvement = 0.0005 + 0.001 * np.random.random()
            
        # Add some noise
        noise = 0.003 * np.random.normal()
        new_score = min(0.95, scores[-1] + improvement + noise)
        scores.append(new_score)
    
    # Create learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(trials, scores, marker='o', linestyle='-', markersize=4, color='#1f77b4')
    
    # Mark the best score
    best_trial = np.argmax(scores) + 1
    best_score = max(scores)
    plt.axhline(y=best_score, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=best_trial, color='r', linestyle='--', alpha=0.5)
    
    # Add best score annotation
    plt.annotate(
        f'Best score: {best_score:.3f} (trial {best_trial})',
        xy=(best_trial, best_score),
        xytext=(best_trial + 5, best_score - 0.02),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
        fontsize=12
    )
    
    plt.title('Hyperparameter Optimization Progress', fontsize=16)
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Score (AUC)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/demo/hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create parameter importance plot
    param_names = ['learning_rate', 'max_depth', 'num_leaves', 'min_child_weight', 
                   'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
    param_importance = [0.35, 0.25, 0.15, 0.08, 0.07, 0.05, 0.03, 0.02]
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.barh(param_names, param_importance, color=plt.cm.viridis(np.linspace(0, 1, len(param_names))))
    plt.xlabel('Importance')
    plt.title('Hyperparameter Importance', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add importance values
    for i, v in enumerate(param_importance):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig('reports/figures/demo/parameter_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive parallel coordinates plot for hyperparameters
    import random
    
    # Generate sample hyperparameter combinations
    n_samples = 20
    hyperparams = []
    
    for i in range(n_samples):
        params = {
            'learning_rate': random.uniform(0.01, 0.3),
            'max_depth': random.randint(3, 10),
            'num_leaves': random.randint(20, 100),
            'min_child_weight': random.uniform(0.5, 10.0),
            'score': random.uniform(0.8, 0.95)
        }
        hyperparams.append(params)
    
    df_params = pd.DataFrame(hyperparams)
    
    # Sort by score
    df_params = df_params.sort_values('score', ascending=False)
    
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        df_params, 
        color="score",
        labels={
            "learning_rate": "Learning Rate",
            "max_depth": "Max Depth",
            "num_leaves": "Num Leaves",
            "min_child_weight": "Min Child Weight",
            "score": "Score"
        },
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Hyperparameter Combinations by Performance"
    )
    
    fig.write_html('reports/figures/demo/hyperparameter_parallel_coords.html')

def create_all_visuals():
    """Create all visualizations"""
    logger.info("Creating all dashboard visualizations...")
    
    # Create directories if they don't exist
    os.makedirs('reports/figures/demo', exist_ok=True)
    
    # Create all types of visualizations
    create_force_plots()
    create_decision_plots()
    create_model_comparison_plots()
    create_eda_visualizations()
    create_feature_importance_plots()
    create_hyperparameter_tuning_plots()
    
    logger.info("All visualizations created successfully!")

if __name__ == "__main__":
    create_all_visuals()
    logger.info("Demo visualizations created successfully!") 