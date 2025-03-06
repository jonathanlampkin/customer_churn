"""
Exploratory Data Analysis for Churn Modeling
This script loads the ARFF dataset and performs initial exploratory analysis.
Using Polars for high-performance data processing.
"""

import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import warnings
from typing import Tuple

# For reproducibility
np.random.seed(42)

# Improve plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
warnings.filterwarnings('ignore')

def load_arff_data(file_path: str) -> pl.DataFrame:
    """
    Load ARFF data file and convert to Polars DataFrame.
    
    Args:
        file_path: Path to the ARFF file
        
    Returns:
        Polars DataFrame containing the data
    """
    print(f"Loading data from {file_path}...")
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
    
    print(f"Dataset loaded successfully with shape: {df.shape}")
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

def main():
    """Main execution function."""
    # Load the dataset
    file_path = "/mnt/hdd/data/churn/mobile_66kx66_nonull.arff"
    df = load_arff_data(file_path)
    
    # Perform exploratory analysis
    explore_data(df)
    visualize_correlations(df)
    analyze_categorical_features(df)
    
    print("\nExploratory Data Analysis completed. Check the output directory for visualizations.")

if __name__ == "__main__":
    main()

