"""
Minimal test script - just reads a small portion of data
"""
import os
import pandas as pd
import sys

print("Starting minimal test...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

try:
    # Try to read just the first few rows of the file
    file_path = "/mnt/hdd/Documents/mobile_churn_66kx66_numeric_nonull"
    print(f"Attempting to read a small portion of: {file_path}")
    
    # Try CSV approach first
    if file_path.endswith('.csv'):
        # Read only first 5 rows
        df = pd.read_csv(file_path, nrows=5)
    else:
        # For ARFF files, use a different approach
        from scipy.io import arff
        
        # Create a temporary file to hold a small sample
        with open(file_path, 'rb') as f:
            # Read the header and a small amount of data
            sample = f.read(10000)  # Just read first 10KB
            
        with open('temp_sample.arff', 'wb') as f:
            f.write(sample)
            
        # Try to parse this small sample
        try:
            data, meta = arff.loadarff('temp_sample.arff')
            print("Successfully parsed ARFF header")
            print(f"Metadata: {list(meta.keys())[:5]}")
        except Exception as e:
            print(f"Error parsing ARFF sample: {e}")
            
            # Alternative approach - read as text
            with open(file_path, 'r') as f:
                lines = [next(f) for _ in range(10)]
                print("First 10 lines of file:")
                for line in lines:
                    print(f"  {line.strip()}")
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
