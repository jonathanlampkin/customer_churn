"""
Convert data file to CSV in small chunks to avoid memory issues
"""
import os
import pandas as pd
import numpy as np
from scipy.io import arff
import logging

# Basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

input_file = "/mnt/hdd/Documents/mobile_churn_66kx66_numeric_nonull"
output_file = "/mnt/hdd/churn_project/data/raw/churn_data.csv"

# Make sure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def process_in_chunks():
    """Process the file in chunks to avoid memory issues"""
    
    # First determine if it's an ARFF or CSV
    if input_file.endswith('.csv'):
        logger.info("Processing CSV file in chunks")
        
        # Process CSV in chunks
        chunk_size = 1000
        chunks_processed = 0
        
        # Get header first
        header = pd.read_csv(input_file, nrows=0).columns.tolist()
        
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            if chunks_processed == 0:
                # First chunk - write with header
                chunk.to_csv(output_file, index=False, mode='w')
            else:
                # Append without header
                chunk.to_csv(output_file, index=False, mode='a', header=False)
                
            chunks_processed += 1
            logger.info(f"Processed chunk {chunks_processed} - {chunk_size * chunks_processed} records")
            
        logger.info(f"CSV conversion complete - {chunk_size * chunks_processed} total records")
        return
    
    # For ARFF files, we need a different approach
    logger.info("Processing ARFF file")
    
    # Try to extract the attributes from the header
    attributes = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip().lower().startswith('@attribute'):
                parts = line.strip().split(' ', 2)
                if len(parts) >= 2:
                    attr_name = parts[1].strip("'\" ")
                    attributes.append(attr_name)
            elif line.strip().lower().startswith('@data'):
                # We've reached the data section
                break
    
    if not attributes:
        logger.error("Could not extract attributes from ARFF header")
        return
        
    logger.info(f"Found {len(attributes)} attributes in ARFF header")
    
    # Now process the data section line by line
    data_started = False
    chunk_size = 1000
    current_chunk = []
    chunks_processed = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            if not data_started:
                if line.strip().lower() == '@data':
                    data_started = True
                    logger.info("Found data section, starting processing")
                continue
                
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('%'):
                continue
                
            # Add this line to current chunk
            current_chunk.append(line.strip().split(','))
            
            # If chunk is full, process it
            if len(current_chunk) >= chunk_size:
                try:
                    # Convert to DataFrame
                    chunk_df = pd.DataFrame(current_chunk, columns=attributes)
                    
                    # Write to CSV
                    mode = 'w' if chunks_processed == 0 else 'a'
                    header = True if chunks_processed == 0 else False
                    chunk_df.to_csv(output_file, index=False, mode=mode, header=header)
                    
                    chunks_processed += 1
                    logger.info(f"Processed chunk {chunks_processed} - {chunk_size * chunks_processed} records")
                    
                    # Clear chunk
                    current_chunk = []
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    return
    
    # Process any remaining records
    if current_chunk:
        try:
            chunk_df = pd.DataFrame(current_chunk, columns=attributes)
            mode = 'w' if chunks_processed == 0 else 'a'
            header = True if chunks_processed == 0 else False
            chunk_df.to_csv(output_file, index=False, mode=mode, header=header)
            chunks_processed += 1
            logger.info(f"Processed final chunk - total {(chunk_size * (chunks_processed-1)) + len(current_chunk)} records")
        except Exception as e:
            logger.error(f"Error processing final chunk: {e}")
    
    logger.info("ARFF conversion complete")

# Run the converter
try:
    logger.info(f"Starting conversion of {input_file} to {output_file}")
    process_in_chunks()
    logger.info("Conversion completed successfully")
except Exception as e:
    logger.error(f"Conversion failed: {e}")
