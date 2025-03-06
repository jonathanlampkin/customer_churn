"""
Synthetic Data Generator for Churn Modeling
This script creates a large synthetic dataset based on the patterns in the original data.
"""

import os
import polars as pl
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def load_original_data(file_path: str) -> pl.DataFrame:
    """Load the original ARFF dataset."""
    print(f"Loading original data from {file_path}...")
    data, meta = arff.loadarff(file_path)
    
    # First convert to pandas
    import pandas as pd
    pdf = pd.DataFrame(data)
    
    # Convert byte strings to regular strings
    string_columns = [col for col in pdf.columns if pdf[col].dtype == object]
    for col in string_columns:
        pdf[col] = pdf[col].str.decode('utf-8')
    
    # Convert to polars
    df = pl.from_pandas(pdf)
    
    print(f"Original dataset loaded with shape: {df.shape}")
    return df

def analyze_data_patterns(df: pl.DataFrame) -> dict:
    """
    Analyze patterns in the data to guide synthetic data generation.
    Returns a dictionary of data characteristics.
    """
    patterns = {}
    
    # Get column types
    patterns['schema'] = df.schema
    
    # Get numerical column statistics
    numeric_cols = [name for name, dtype in df.schema.items() 
                   if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    
    patterns['numeric_stats'] = {}
    for col in numeric_cols:
        patterns['numeric_stats'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'q25': df[col].quantile(0.25),
            'q50': df[col].quantile(0.50),
            'q75': df[col].quantile(0.75)
        }
    
    # Get categorical column statistics
    categorical_cols = [name for name, dtype in df.schema.items() 
                       if dtype in [pl.Utf8, pl.Categorical]]
    
    patterns['categorical_stats'] = {}
    for col in categorical_cols:
        # Get value counts as a dictionary
        value_counts = df.group_by(col).agg(pl.count()).sort('count', descending=True)
        total = df.height
        
        values = value_counts[col].to_list()
        counts = value_counts['count'].to_list()
        probabilities = [count/total for count in counts]
        
        patterns['categorical_stats'][col] = {
            'values': values,
            'probabilities': probabilities
        }
    
    # Identify target column
    potential_target_cols = [col for col in df.columns if 'churn' in col.lower()]
    if potential_target_cols:
        patterns['target_col'] = potential_target_cols[0]
    else:
        # Try to guess based on categorical columns with few values
        binary_cats = [col for col in categorical_cols 
                     if len(patterns['categorical_stats'][col]['values']) == 2]
        if binary_cats:
            patterns['target_col'] = binary_cats[0]
        else:
            patterns['target_col'] = None
    
    return patterns

def generate_synthetic_data(original_df: pl.DataFrame, patterns: dict, multiplier: int = 10) -> pl.DataFrame:
    """
    Generate synthetic data based on patterns in the original data.
    
    Args:
        original_df: Original Polars DataFrame
        patterns: Dictionary of data patterns from analyze_data_patterns
        multiplier: How many times larger to make the synthetic dataset
    
    Returns:
        Synthetic Polars DataFrame
    """
    n_samples = original_df.height * multiplier
    print(f"Generating {n_samples} synthetic samples...")
    
    # For capturing the generated data
    synthetic_data = {}
    
    # Generate categorical features first
    for col, stats in patterns['categorical_stats'].items():
        values = stats['values']
        probs = stats['probabilities']
        synthetic_data[col] = np.random.choice(values, size=n_samples, p=probs)
    
    # Handle target variable specially if it exists
    target_col = patterns.get('target_col')
    if target_col and target_col in patterns['categorical_stats']:
        # Keep the general class distribution but ensure some realistic patterns
        pass  # The target generation is handled with categorical features
    
    # For numerical columns, we want to preserve relationships
    # Convert original data to pandas for easier ML modeling
    numeric_cols = list(patterns['numeric_stats'].keys())
    
    if len(numeric_cols) > 1:  # Only if we have multiple numeric columns
        # We'll use PCA to understand the feature space
        pdf = original_df.select(numeric_cols).to_pandas()
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pdf)
        
        # Apply PCA to understand the principal components
        n_components = min(len(numeric_cols), 10)  # Max 10 components
        pca = PCA(n_components=n_components)
        pca.fit(scaled_data)
        
        # Generate synthetic samples in the PCA space
        synthetic_pca = np.random.multivariate_normal(
            mean=np.zeros(n_components),
            cov=np.diag(pca.explained_variance_),
            size=n_samples
        )
        
        # Transform back to original feature space
        synthetic_scaled = synthetic_pca @ pca.components_
        synthetic_numeric = scaler.inverse_transform(synthetic_scaled)
        
        # Store the values
        for i, col in enumerate(numeric_cols):
            synthetic_data[col] = synthetic_numeric[:, i]
            
            # Apply min/max bounds from original data to avoid unrealistic values
            min_val = patterns['numeric_stats'][col]['min']
            max_val = patterns['numeric_stats'][col]['max']
            synthetic_data[col] = np.clip(synthetic_data[col], min_val, max_val)
    else:
        # If only one or zero numeric columns, generate independently
        for col, stats in patterns['numeric_stats'].items():
            mean = stats['mean']
            std = stats['std']
            min_val = stats['min']
            max_val = stats['max']
            
            # Generate with normal distribution around mean
            synthetic_data[col] = np.random.normal(mean, std, size=n_samples)
            
            # Clip to original min/max
            synthetic_data[col] = np.clip(synthetic_data[col], min_val, max_val)
    
    # Create a Polars DataFrame
    synthetic_df = pl.DataFrame(synthetic_data)
    
    # Make sure data types match the original
    for col, dtype in patterns['schema'].items():
        if col in synthetic_df.columns:
            synthetic_df = synthetic_df.with_columns(
                pl.col(col).cast(dtype)
            )
    
    print(f"Synthetic dataset generated with shape: {synthetic_df.shape}")
    return synthetic_df

def visualize_comparison(original_df: pl.DataFrame, synthetic_df: pl.DataFrame, 
                         patterns: dict, n_samples: int = 10000) -> None:
    """
    Visualize a comparison between original and synthetic data.
    
    Args:
        original_df: Original Polars DataFrame
        synthetic_df: Synthetic Polars DataFrame
        patterns: Dictionary of data patterns
        n_samples: Number of samples to use for visualization
    """
    # Identify numeric columns for scatter plots
    numeric_cols = list(patterns['numeric_stats'].keys())
    
    if len(numeric_cols) >= 2:
        # Sample data for visualization
        original_sample = original_df.sample(n=min(n_samples, original_df.height))
        synthetic_sample = synthetic_df.sample(n=min(n_samples, synthetic_df.height))
        
        # Convert to pandas for easier plotting
        original_pd = original_sample.select(numeric_cols).to_pandas()
        synthetic_pd = synthetic_sample.select(numeric_cols).to_pandas()
        
        # Add a dataset label
        original_pd['dataset'] = 'Original'
        synthetic_pd['dataset'] = 'Synthetic'
        
        # Combine for plotting
        combined = pd.concat([original_pd, synthetic_pd])
        
        # Create scatter plot matrix for first few numeric features
        plot_cols = numeric_cols[:4]  # Limit to first 4 features
        
        plt.figure(figsize=(15, 15))
        sns.pairplot(combined, vars=plot_cols, hue='dataset', 
                    plot_kws={'alpha': 0.5, 's': 10})
        plt.suptitle('Comparison of Original vs Synthetic Data', y=1.02, fontsize=16)
        plt.savefig('original_vs_synthetic_scatter.png', dpi=300, bbox_inches='tight')
    
    # Compare distributions for numeric columns
    for i, col in enumerate(numeric_cols[:5]):  # Limit to first 5
        plt.figure(figsize=(12, 6))
        
        # Convert to pandas for plotting
        original_vals = original_df[col].to_pandas()
        synthetic_vals = synthetic_df[col].to_pandas()
        
        plt.hist(original_vals, alpha=0.5, label='Original', bins=30)
        plt.hist(synthetic_vals, alpha=0.5, label='Synthetic', bins=30)
        
        plt.title(f'Distribution Comparison for {col}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'distribution_compare_{col}.png', dpi=300, bbox_inches='tight')
    
    # Compare categorical distributions
    categorical_cols = list(patterns['categorical_stats'].keys())
    target_col = patterns.get('target_col')
    
    # If target exists, compare target distribution
    if target_col:
        plt.figure(figsize=(10, 6))
        
        # Get original target distribution
        orig_target_counts = original_df.group_by(target_col).agg(
            pl.count().alias('count')
        ).with_columns(
            (pl.col('count') / original_df.height * 100).alias('percentage')
        ).sort(target_col)
        
        # Get synthetic target distribution
        synth_target_counts = synthetic_df.group_by(target_col).agg(
            pl.count().alias('count')
        ).with_columns(
            (pl.col('count') / synthetic_df.height * 100).alias('percentage')
        ).sort(target_col)
        
        # Plot as bar chart
        targets = orig_target_counts[target_col].to_list()
        orig_pcts = orig_target_counts['percentage'].to_list()
        synth_pcts = synth_target_counts['percentage'].to_list()
        
        x = np.arange(len(targets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, orig_pcts, width, label='Original')
        ax.bar(x + width/2, synth_pcts, width, label='Synthetic')
        
        ax.set_ylabel('Percentage')
        ax.set_title(f'{target_col} Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('target_distribution_comparison.png', dpi=300, bbox_inches='tight')

def save_synthetic_data(df: pl.DataFrame, output_path: str) -> None:
    """Save the synthetic dataset."""
    # Save as Parquet for efficiency
    parquet_path = output_path.replace('.arff', '.parquet')
    df.write_parquet(parquet_path)
    print(f"Synthetic dataset saved to {parquet_path}")
    
    # Convert to pandas and save as CSV for compatibility
    csv_path = output_path.replace('.arff', '.csv')
    df.write_csv(csv_path)
    print(f"Synthetic dataset also saved as CSV to {csv_path}")

def main():
    """Main execution function."""
    # Original data path
    original_path = "/mnt/hdd/data/churn/mobile_66kx66_nonull.arff"
    
    # Load original data
    original_df = load_original_data(original_path)
    
    # Analyze patterns
    patterns = analyze_data_patterns(original_df)
    
    # Generate synthetic data (10x the original size)
    synthetic_df = generate_synthetic_data(original_df, patterns, multiplier=10)
    
    # Visualize comparison
    visualize_comparison(original_df, synthetic_df, patterns)
    
    # Save synthetic data
    output_path = "/mnt/hdd/data/churn/synthetic_churn_data.parquet"
    save_synthetic_data(synthetic_df, output_path)
    
    print("Synthetic data generation completed!")

if __name__ == "__main__":
    main() 