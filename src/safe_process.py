"""
Ultra-safe processing of ARFF file with minimal dependencies
"""
import os
import json
import numpy as np
import pandas as pd
import gc  # Garbage collection

# Create output directories
os.makedirs('reports/metrics', exist_ok=True)

# Safe ARFF loading
print("Safely loading ARFF data...")
DATA_PATH = "/mnt/hdd/Documents/mobile_churn_66kx66_numeric_nonull"

# Try with scipy.io.arff
try:
    # Load line by line to avoid memory issues
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
    
    # Parse header to find data start
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == '@data':
            data_start = i + 1
            break
    
    # Extract header info
    header_lines = lines[:data_start-1]
    attribute_names = []
    for line in header_lines:
        if line.lower().startswith('@attribute'):
            parts = line.split()
            if len(parts) > 1:
                attribute_names.append(parts[1].strip())
    
    # Process 1000 rows at a time
    chunk_size = 1000
    all_data = []
    
    for i in range(data_start, len(lines), chunk_size):
        chunk = lines[i:i+chunk_size]
        chunk_data = []
        for line in chunk:
            if line.strip() and not line.startswith('%'):
                values = line.strip().split(',')
                chunk_data.append(values)
        all_data.extend(chunk_data)
        
        # Force garbage collection
        gc.collect()
        print(f"Processed {len(all_data)} rows...")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=attribute_names)
    print(f"Created DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Convert to proper types
    for column in df.columns:
        if df[column].dtype == object:
            # Try to convert to numeric if possible
            try:
                df[column] = pd.to_numeric(df[column])
            except (ValueError, TypeError):
                # If not convertible to numeric, keep as string
                pass
                
    # Find target column
    target_column = None
    for col in df.columns:
        if 'churn' in col.lower():
            target_column = col
            break
    
    if not target_column:
        target_column = df.columns[-1]
        print(f"No column with 'churn' in name found, using last column: {target_column}")
    else:
        print(f"Using '{target_column}' as target column")
    
    # Process the data and train models
    # ... (implement the actual model training and metric generation)
    
except Exception as e:
    print(f"Error during processing: {str(e)}")
    import traceback
    traceback.print_exc()