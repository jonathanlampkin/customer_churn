"""
Minimal EDA script that avoids memory issues
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Create output directory
os.makedirs('/mnt/hdd/churn_project/reports/figures', exist_ok=True)

# File paths
data_file = "/mnt/hdd/churn_project/data/raw/churn_data.csv"
output_dir = "/mnt/hdd/churn_project/reports/figures"

def run_minimal_eda():
    """Run basic EDA on the dataset"""
    logger.info(f"Loading data from {data_file}")
    
    # Read just the column names and a few rows first
    df_sample = pd.read_csv(data_file, nrows=5)
    columns = df_sample.columns.tolist()
    
    logger.info(f"Dataset has {len(columns)} columns")
    
    # Look for potential target column
    target_col = None
    for col in columns:
        if 'churn' in col.lower():
            target_col = col
            break
    
    if not target_col:
        logger.warning("Could not identify target column")
        # Just use the last column as target
        target_col = columns[-1]
        
    logger.info(f"Using '{target_col}' as target column")
    
    # Now let's analyze the target distribution
    logger.info("Analyzing target distribution")
    
    # Read in chunks to count target values
    target_counts = {}
    chunk_size = 1000
    
    for chunk in pd.read_csv(data_file, chunksize=chunk_size):
        # Count values in this chunk
        chunk_counts = chunk[target_col].value_counts().to_dict()
        
        # Add to overall counts
        for value, count in chunk_counts.items():
            if value in target_counts:
                target_counts[value] += count
            else:
                target_counts[value] = count
    
    # Create a simple bar chart
    plt.figure(figsize=(8, 6))
    values = list(target_counts.keys())
    counts = list(target_counts.values())
    
    plt.bar(values, counts)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    
    # Save the figure
    output_file = os.path.join(output_dir, 'target_distribution.png')
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Target distribution plot saved to {output_file}")
    
    # Print basic statistics
    total = sum(counts)
    percentages = {value: (count/total)*100 for value, count in target_counts.items()}
    
    logger.info(f"Target Distribution Statistics:")
    for value, count in target_counts.items():
        logger.info(f"  {value}: {count} records ({percentages[value]:.2f}%)")
    
    return target_col

# Run the EDA
try:
    logger.info("Starting minimal EDA")
    target_column = run_minimal_eda()
    logger.info("Minimal EDA completed successfully")
except Exception as e:
    logger.error(f"EDA failed: {e}")
