"""
Advanced Exploratory Data Analysis for Churn Prediction
Uses cutting-edge visualization and statistical techniques to explore the patterns in churn data.
"""

import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from scipy import stats
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
from typing import Tuple, List, Dict, Union, Optional
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# For reproducibility
np.random.seed(42)

# Improve plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
sns.set_palette("viridis")
warnings.filterwarnings('ignore')

# Create figures directory
os.makedirs('reports/figures/eda', exist_ok=True)

def load_config(config_path: str = 'configs/config.yml') -> dict:
    """Load project configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_arff_data(file_path: str) -> pl.DataFrame:
    """
    Load ARFF data file and convert to Polars DataFrame.
    
    Args:
        file_path: Path to the ARFF file
        
    Returns:
        Polars DataFrame containing the data
    """
    logger.info(f"Loading data from {file_path}...")
    data, meta = arff.loadarff(file_path)
    
    # First convert to pandas (arff loader returns pandas-compatible format)
    import pandas as pd
    pdf = pd.DataFrame(data)
    
    # Convert byte strings to regular strings (common in ARFF files)
    string_columns = [col for col in pdf.columns if pdf[col].dtype == object]
    for col in string_columns:
        pdf[col] = pdf[col].str.decode('utf-8')
    
    # Convert to polars
    df = pl.from_pandas(pdf)
    
    logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    return df

def explore_data(df: pl.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df: Input DataFrame
    """
    print("\n==== Basic Dataset Information ====")
    print(f"Number of samples: {df.height}")
    print(f"Number of features: {df.width}")
    
    # Display data types
    print("\n==== Data Types ====")
    print(df.schema)
    
    # Summary statistics
    print("\n==== Summary Statistics ====")
    print(df.describe())
    
    print("\n==== Missing Values Analysis ====")
    missing = df.null_count()
    missing_df = pl.DataFrame({
        'Column': missing.keys(),
        'Missing Count': [v for v in missing.values()]
    })
    missing_df = missing_df.with_columns(
        (pl.col('Missing Count') / df.height * 100).alias('Missing Percentage')
    )
    print(missing_df.filter(pl.col('Missing Count') > 0))
    
    # Identify the target variable (assuming it's called 'churn' or similar)
    potential_target_cols = [col for col in df.columns if 'churn' in col.lower()]
    if potential_target_cols:
        target_col = potential_target_cols[0]
        print(f"\n==== Target Variable: {target_col} ====")
        
        # Get value counts
        value_counts = df.group_by(target_col).agg(
            pl.count().alias('count')
        ).with_columns(
            (pl.col('count') / df.height * 100).alias('percentage')
        )
        print(value_counts)
        
        # For visualization, we need to convert to pandas
        pdf = df.to_pandas()
        
        # Create a visualization of class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=target_col, data=pdf)
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Count')
        plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
    else:
        print("\nCould not automatically identify target variable.")

def visualize_correlations(df: pl.DataFrame) -> None:
    """
    Create a correlation heatmap for numerical features.
    
    Args:
        df: Input DataFrame
    """
    # Select only numeric columns - convert to pandas for correlation calculation
    numeric_cols = [name for name, dtype in df.schema.items() 
                   if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    
    # For correlation visualization, convert to pandas
    pdf = df.select(numeric_cols).to_pandas()
    
    # Compute correlations
    plt.figure(figsize=(14, 10))
    correlation_matrix = pdf.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

def analyze_categorical_features(df: pl.DataFrame) -> None:
    """
    Analyze categorical features and their relationship with the target.
    
    Args:
        df: Input DataFrame
    """
    # Identify categorical columns (Utf8 or categorical in Polars)
    categorical_cols = [name for name, dtype in df.schema.items() 
                       if dtype in [pl.Utf8, pl.Categorical]]
    
    # Identify potential target column
    potential_target_cols = [col for col in df.columns if 'churn' in col.lower()]
    if not potential_target_cols:
        print("Cannot find target column for categorical analysis")
        return
        
    target_col = potential_target_cols[0]
    
    # Convert to pandas for easier plotting
    pdf = df.to_pandas()
    
    # Plot categorical features vs target
    for i, col in enumerate(categorical_cols):
        if col == target_col:
            continue
            
        plt.figure(figsize=(12, 6))
        
        # Create a cross-tabulation and normalize
        crosstab = pd.crosstab(pdf[col], pdf[target_col], normalize='index') * 100
        
        # Plot
        crosstab.plot(kind='bar', stacked=False)
        plt.title(f'{col} vs {target_col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel(f'Percentage of {target_col}', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title=target_col)
        plt.tight_layout()
        plt.savefig(f'categorical_{col}_vs_target.png', dpi=300, bbox_inches='tight')
        
        # Limit to 5 plots to avoid overwhelming output
        if i >= 4:
            print(f"Only showing first 5 categorical features out of {len(categorical_cols)}")
            break

def statistical_analysis(df: pl.DataFrame, target_col: str) -> Dict[str, Dict[str, float]]:
    """
    Perform statistical tests and analysis on features.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary of statistical test results
    """
    logger.info("Performing statistical analysis on features")
    
    # Convert to pandas for statistical tests
    pdf = df.to_pandas()
    
    # Identify numeric and categorical columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    categorical_cols = [col for col in df.columns if col not in numeric_cols and col != target_col]
    
    results = {}
    
    # For numeric features: Mann-Whitney U test for binary target
    if len(pdf[target_col].unique()) == 2:
        target_values = pdf[target_col].unique()
        
        for col in numeric_cols:
            if col == target_col:
                continue
                
            group1 = pdf[pdf[target_col] == target_values[0]][col].dropna()
            group2 = pdf[pdf[target_col] == target_values[1]][col].dropna()
            
            try:
                # Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(group1, group2)
                
                # Cohen's d for effect size
                d = (group1.mean() - group2.mean()) / np.sqrt((group1.std()**2 + group2.std()**2) / 2)
                
                results[col] = {
                    'test': 'mannwhitneyu',
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': d,
                    'effect_strength': 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'
                }
            except Exception as e:
                logger.warning(f"Could not perform Mann-Whitney U test for {col}: {e}")
    
    # For categorical features: Chi-square test
    for col in categorical_cols:
        try:
            # Create contingency table
            contingency = pd.crosstab(pdf[col], pdf[target_col])
            
            # Chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            # Cramer's V for effect size
            n = contingency.sum().sum()
            phi2 = chi2 / n
            r, k = contingency.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramers_v = np.sqrt(phi2corr / min(kcorr-1, rcorr-1))
            
            results[col] = {
                'test': 'chi2',
                'chi2_statistic': chi2,
                'p_value': p,
                'significant': p < 0.05,
                'effect_size': cramers_v,
                'effect_strength': 'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large'
            }
        except Exception as e:
            logger.warning(f"Could not perform Chi-square test for {col}: {e}")
    
    # Create a table of significant features
    significant_features = {k: v for k, v in results.items() if v['significant']}
    
    if significant_features:
        logger.info(f"Found {len(significant_features)} statistically significant features")
        
        # Display top features by effect size
        top_features = sorted(significant_features.items(), key=lambda x: x[1]['effect_size'], reverse=True)
        
        for feature, stats_dict in top_features[:10]:
            logger.info(f"{feature}: p-value={stats_dict['p_value']:.4f}, effect size={stats_dict['effect_size']:.4f} ({stats_dict['effect_strength']})")
    else:
        logger.info("No statistically significant features found")
    
    return results

def compute_feature_importance(df: pl.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Compute feature importance using information theoretic measures.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        
    Returns:
        DataFrame with feature importance scores
    """
    logger.info("Computing feature importance measures")
    
    # Convert to pandas
    pdf = df.to_pandas()
    
    # Extract X and y
    X = pdf.drop(columns=[target_col])
    y = pdf[target_col]
    
    # Identify numeric columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        logger.warning("No numeric features found for importance calculation")
        return pd.DataFrame()
    
    # Calculate mutual information
    X_numeric = X[numeric_cols]
    
    try:
        # Handle NaN values
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': numeric_cols,
            'Mutual_Information': mi_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Mutual_Information', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Mutual_Information', y='Feature', data=importance_df.head(20))
        plt.title('Feature Importance by Mutual Information', fontsize=15)
        plt.tight_layout()
        plt.savefig('reports/figures/eda/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Top 5 features by mutual information: {', '.join(importance_df['Feature'].head(5).tolist())}")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error computing feature importance: {e}")
        return pd.DataFrame()

def dimensionality_reduction(df: pl.DataFrame, target_col: str) -> Dict[str, np.ndarray]:
    """
    Apply dimensionality reduction techniques for visualization.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary of embeddings from different techniques
    """
    logger.info("Applying dimensionality reduction techniques")
    
    # Convert to pandas
    pdf = df.to_pandas()
    
    # Extract numeric features and target
    numeric_cols = pdf.select_dtypes(include=['number']).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if not numeric_cols:
        logger.warning("No numeric features found for dimensionality reduction")
        return {}
    
    X = pdf[numeric_cols]
    y = pdf[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    embeddings = {}
    
    # PCA
    try:
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X_scaled)
        embeddings['pca'] = pca_result
        
        # Plot PCA
        pca_df = pd.DataFrame({
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1],
            'target': y
        })
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PCA1', y='PCA2', hue='target', data=pca_df, palette='viridis', alpha=0.7)
        plt.title('PCA Visualization', fontsize=15)
        plt.tight_layout()
        plt.savefig('reports/figures/eda/pca_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Explained variance
        exp_var = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance: {exp_var[0]:.4f}, {exp_var[1]:.4f} (total: {sum(exp_var):.4f})")
        
    except Exception as e:
        logger.error(f"Error in PCA: {e}")
    
    # t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        tsne_result = tsne.fit_transform(X_scaled)
        embeddings['tsne'] = tsne_result
        
        # Plot t-SNE
        tsne_df = pd.DataFrame({
            'TSNE1': tsne_result[:, 0],
            'TSNE2': tsne_result[:, 1],
            'target': y
        })
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='target', data=tsne_df, palette='viridis', alpha=0.7)
        plt.title('t-SNE Visualization', fontsize=15)
        plt.tight_layout()
        plt.savefig('reports/figures/eda/tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in t-SNE: {e}")
    
    # UMAP
    try:
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(X_scaled)
        embeddings['umap'] = umap_result
        
        # Plot UMAP
        umap_df = pd.DataFrame({
            'UMAP1': umap_result[:, 0],
            'UMAP2': umap_result[:, 1],
            'target': y
        })
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='target', data=umap_df, palette='viridis', alpha=0.7)
        plt.title('UMAP Visualization', fontsize=15)
        plt.tight_layout()
        plt.savefig('reports/figures/eda/umap_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in UMAP: {e}")
    
    return embeddings

def visualize_feature_distributions(df: pl.DataFrame, target_col: str, top_n: int = 10) -> None:
    """
    Create advanced visualizations for feature distributions by target class.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        top_n: Number of top features to visualize
    """
    logger.info(f"Visualizing distributions for top {top_n} features")
    
    # Get feature importance to select top features
    importance_df = compute_feature_importance(df, target_col)
    
    if importance_df.empty:
        logger.warning("No feature importance scores available")
        return
    
    # Select top N features
    top_features = importance_df['Feature'].head(top_n).tolist()
    
    # Convert to pandas
    pdf = df.to_pandas()
    
    # Create a grid of distribution plots
    n_cols = 2
    n_rows = (len(top_features) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        if i < len(axes):
            # Kernel Density Estimation plot by target
            sns.kdeplot(
                data=pdf, x=feature, hue=target_col, 
                fill=True, common_norm=False, 
                alpha=0.5, linewidth=1,
                ax=axes[i]
            )
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].grid(alpha=0.3)
    
    # Remove any unused axes
    for i in range(len(top_features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('reports/figures/eda/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create violin plots for top 5 features
    for feature in top_features[:5]:
        plt.figure(figsize=(10, 6))
        
        sns.violinplot(x=target_col, y=feature, data=pdf, palette='viridis')
        
        plt.title(f'Violin Plot: {feature} by {target_col}', fontsize=13)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'reports/figures/eda/violin_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()

def correlation_analysis(df: pl.DataFrame, target_col: str) -> None:
    """
    Perform advanced correlation analysis.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
    """
    logger.info("Performing correlation analysis")
    
    # Get numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if not numeric_cols:
        logger.warning("No numeric features found for correlation analysis")
        return
    
    # Convert to pandas
    pdf = df.to_pandas()[numeric_cols + [target_col]]
    
    # Calculate correlations
    corr_pearson = pdf.corr(method='pearson')
    corr_spearman = pdf.corr(method='spearman')
    
    # Plot heatmap for Pearson correlation
    plt.figure(figsize=(14, 12))
    
    # Use a mask to hide the upper triangle
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_pearson, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        square=True, linewidths=.5, annot=False,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Pearson Correlation Matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig('reports/figures/eda/correlation_pearson.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot heatmap for Spearman correlation
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(
        corr_spearman, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        square=True, linewidths=.5, annot=False,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Spearman Correlation Matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig('reports/figures/eda/correlation_spearman.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find features most correlated with target
    target_corr = corr_pearson[target_col].drop(target_col).sort_values(ascending=False)
    
    # Plot top positive and negative correlations with target
    plt.figure(figsize=(12, 8))
    
    target_corr.head(10).append(target_corr.tail(10)).plot(kind='barh', color=plt.cm.RdYlGn(np.linspace(0, 1, 20)))
    
    plt.title(f'Top Correlations with {target_col}', fontsize=15)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/eda/target_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Top 5 positively correlated features: {', '.join(target_corr.head(5).index.tolist())}")
    logger.info(f"Top 5 negatively correlated features: {', '.join(target_corr.tail(5).index.tolist())}")

def run_complete_eda(data_path: str) -> None:
    """
    Run the complete EDA pipeline.
    
    Args:
        data_path: Path to the data file
    """
    logger.info("Starting comprehensive EDA pipeline")
    
    # Load the data
    df = load_arff_data(data_path)
    
    # Identify the target variable
    target_cols = [col for col in df.columns if 'churn' in col.lower()]
    
    if not target_cols:
        logger.error("Could not identify the target variable")
        return
    
    target_col = target_cols[0]
    logger.info(f"Identified target variable: {target_col}")
    
    # Basic exploration
    explore_data(df)
    
    # Statistical analysis
    stats_results = statistical_analysis(df, target_col)
    
    # Feature importance
    importance_df = compute_feature_importance(df, target_col)
    
    # Dimensionality reduction
    embeddings = dimensionality_reduction(df, target_col)
    
    # Visualize feature distributions
    visualize_feature_distributions(df, target_col)
    
    # Correlation analysis
    correlation_analysis(df, target_col)
    
    logger.info("EDA pipeline completed")

def main():
    """Main execution function"""
    try:
        # Load configuration
        config = load_config()
        
        # Get data path
        data_path = config['data']['raw_path']
        
        # Run the complete EDA
        run_complete_eda(data_path)
        
    except Exception as e:
        logger.error(f"Error in EDA: {e}")

if __name__ == "__main__":
    main()


