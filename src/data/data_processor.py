"""
Data processing module for the churn prediction project.
Handles loading, cleaning, and preprocessing of data.
"""

import os
import polars as pl
import numpy as np
from scipy.io import arff
import yaml
from typing import Tuple, Dict, List, Optional, Union
import logging
from pathlib import Path
import pandas as pd
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChurnDataProcessor:
    """
    Class to handle all data processing for the churn prediction project.
    Uses Polars for high-performance data operations.
    """
    
    def __init__(self, config_path: Union[str, Path] = 'configs/config.yml'):
        """
        Initialize the data processor with project configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_raw = None
        self.data_processed = None
        logger.info("Data processor initialized")
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_data(self) -> pl.DataFrame:
        """
        Load raw data from ARFF file.
        
        Returns:
            DataFrame with loaded data
        """
        raw_path = self.config['data']['raw_path']
        
        # Override with command line argument if provided
        for i, arg in enumerate(sys.argv):
            if arg == '--data_path' and i+1 < len(sys.argv):
                raw_path = sys.argv[i+1]
                logger.info(f"Using data path from command line: {raw_path}")
        
        logger.info(f"Loading raw data from {raw_path}")
        
        # Handle ARFF files
        if raw_path.endswith('.arff') or (not raw_path.endswith('.csv') and not raw_path.endswith('.parquet')):
            try:
                # Use safer approach for ARFF files
                logger.info("Loading ARFF file with safer method")
                
                # Use chunks to reduce memory issues
                with open(raw_path, 'r') as f:
                    header = []
                    for line in f:
                        if line.strip().lower() == '@data':
                            break
                        header.append(line)
                
                # Parse ARFF more carefully
                data, meta = arff.loadarff(raw_path)
                df = pd.DataFrame(data)
                
                # Convert bytes to strings (common in ARFF)
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].str.decode('utf-8')
                    
                # Convert to polars DataFrame
                df_pl = pl.from_pandas(df)
                
                logger.info(f"Data loaded successfully: {df_pl.shape}")
                self.data_raw = df_pl
                return df_pl
                
            except Exception as e:
                logger.error(f"Error loading ARFF file: {e}")
                raise
        
        # ... existing code for CSV and Parquet ...
    
    def examine_nulls(self, df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Check for null values and common null placeholders.
        
        Args:
            df: DataFrame to check (uses self.data_raw if None)
            
        Returns:
            DataFrame summarizing null information
        """
        if df is None:
            df = self.data_raw
            
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Check for actual null values
        null_counts = df.null_count()
        
        # Check for common null placeholders in numeric columns
        numeric_cols = [name for name, dtype in df.schema.items() 
                       if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        
        # Common numeric placeholders: -999, -1, 99999, etc.
        placeholder_suspects = {}
        
        for col in numeric_cols:
            # Get value counts for this column
            value_counts = df.group_by(col).agg(pl.count()).sort('count')
            
            # Check the least frequent values
            if value_counts.height > 0:
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Check if extreme values appear with suspiciously low frequency
                least_common = value_counts.head(3)
                for i in range(min(least_common.height, 3)):
                    val = least_common[col][i]
                    count = least_common['count'][i]
                    
                    # If it's an extreme value with few occurrences, flag it
                    if (val == min_val or val == max_val or 
                        val in [-999, -99, -9, -1, 9, 99, 999, 9999] or
                        abs(val) > 1e6):
                        
                        if col not in placeholder_suspects:
                            placeholder_suspects[col] = []
                        
                        placeholder_suspects[col].append((val, count))
        
        # Create summary DataFrame
        summary_rows = []
        
        # Add null counts
        for col, count in null_counts.items():
            summary_rows.append({
                'column': col,
                'issue_type': 'Null Values',
                'value': 'null',
                'count': count,
                'percentage': (count / df.height) * 100
            })
        
        # Add placeholder suspects
        for col, suspects in placeholder_suspects.items():
            for val, count in suspects:
                summary_rows.append({
                    'column': col,
                    'issue_type': 'Possible Null Placeholder',
                    'value': str(val),
                    'count': count,
                    'percentage': (count / df.height) * 100
                })
        
        # Convert to Polars DataFrame
        summary_df = pl.DataFrame(summary_rows)
        
        if summary_df.height > 0:
            # Sort by count descending
            summary_df = summary_df.sort('count', descending=True)
            
            # Log only the meaningful issues (count > 0)
            meaningful_issues = summary_df.filter(pl.col('count') > 0)
            if meaningful_issues.height > 0:
                logger.info(f"Found potential data quality issues:\n{meaningful_issues}")
            else:
                logger.info("No data quality issues found")
        else:
            logger.info("No data quality issues found")
            
        return summary_df
    
    def identify_exclude_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Identify columns that should be excluded from modeling like IDs,
        timestamps, and other non-predictive columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names to exclude
        """
        exclude_columns = []
        
        # 1. Get column names from config (if specified)
        config_excludes = self.config.get('preprocessing', {}).get('exclude_columns', [])
        if config_excludes:
            exclude_columns.extend(config_excludes)
            logger.info(f"Excluding columns specified in config: {config_excludes}")
        
        # 2. Find potential ID columns by name pattern
        id_patterns = ['id', 'customer', 'account', 'record', 'number', 'uuid', 'guid']
        pattern_matches = [col for col in df.columns if any(
            pattern.lower() in col.lower() for pattern in id_patterns
        )]
        
        if pattern_matches:
            logger.info(f"Found potential ID columns: {pattern_matches}")
            exclude_columns.extend(pattern_matches)
        
        # 3. Find columns with unique or near-unique values (possible IDs)
        for col in df.columns:
            # Skip columns already marked for exclusion
            if col in exclude_columns:
                continue
            
            # Check numerical columns for high cardinality
            if df[col].dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                # Get count of unique values
                unique_count = df[col].n_unique()
                
                # If unique values > 95% of total rows, likely an ID
                if unique_count > 0.95 * df.height:
                    logger.warning(f"Column '{col}' has {unique_count} unique values "
                                  f"({unique_count/df.height:.2%} of rows), likely an ID or unique identifier")
                    exclude_columns.append(col)
        
        # 4. Check for timestamp/date columns
        date_patterns = ['date', 'time', 'created', 'modified', 'timestamp']
        potential_timestamps = [col for col in df.columns if any(
            pattern.lower() in col.lower() for pattern in date_patterns
        )]
        
        if potential_timestamps:
            logger.info(f"Found potential timestamp columns: {potential_timestamps}")
            exclude_columns.extend(potential_timestamps)
        
        # De-duplicate the list
        exclude_columns = list(set(exclude_columns))
        
        logger.info(f"Total {len(exclude_columns)} columns marked for exclusion")
        return exclude_columns
    
    def preprocess_data(self, df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Preprocess the data: handle missing values, outliers, remove ID columns, etc.
        
        Args:
            df: DataFrame to process (uses self.data_raw if None)
            
        Returns:
            Processed DataFrame
        """
        if df is None:
            df = self.data_raw
            
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying the original
        processed_df = df.clone()
        
        # Identify and remove columns that shouldn't be used for modeling
        exclude_columns = self.identify_exclude_columns(processed_df)
        
        # Log the columns being removed
        if exclude_columns:
            logger.info(f"Removing the following columns from modeling: {exclude_columns}")
            processed_df = processed_df.drop(exclude_columns)
            logger.info(f"Data shape after removing excluded columns: {processed_df.shape}")
        
        # 1. Handle potential null placeholders
        null_info = self.examine_nulls(processed_df)
        placeholders = null_info.filter(pl.col('issue_type') == 'Possible Null Placeholder')
        
        if placeholders.height > 0:
            for i in range(placeholders.height):
                col = placeholders['column'][i]
                val = placeholders['value'][i]
                
                # Try to convert placeholder value to float
                try:
                    placeholder_val = float(val)
                    
                    # Replace with nulls
                    processed_df = processed_df.with_columns([
                        pl.when(pl.col(col) == placeholder_val)
                        .then(None)
                        .otherwise(pl.col(col))
                        .alias(col)
                    ])
                    
                    logger.info(f"Replaced placeholder {val} in column '{col}' with nulls")
                except ValueError:
                    # If it's not a numeric value, skip
                    continue
        
        # 2. Handle actual nulls with appropriate imputation
        # For numeric columns, use median imputation
        numeric_cols = [name for name, dtype in processed_df.schema.items() 
                      if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        
        # Fill nulls with median for each numeric column
        for col in numeric_cols:
            if processed_df[col].null_count() > 0:
                median_val = processed_df[col].median()
                
                processed_df = processed_df.with_columns([
                    pl.col(col).fill_null(median_val).alias(col)
                ])
                
                logger.info(f"Imputed nulls in column '{col}' with median value {median_val:.4f}")
        
        # 3. Handle outliers in numeric columns
        # Using IQR method to detect and cap outliers
        for col in numeric_cols:
            # Compute IQR
            q1 = processed_df[col].quantile(0.25)
            q3 = processed_df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            n_lower_outliers = processed_df.filter(pl.col(col) < lower_bound).height
            n_upper_outliers = processed_df.filter(pl.col(col) > upper_bound).height
            
            if n_lower_outliers + n_upper_outliers > 0:
                # Cap outliers
                processed_df = processed_df.with_columns([
                    pl.when(pl.col(col) < lower_bound)
                    .then(lower_bound)
                    .when(pl.col(col) > upper_bound)
                    .then(upper_bound)
                    .otherwise(pl.col(col))
                    .alias(col)
                ])
                
                logger.info(f"Capped {n_lower_outliers + n_upper_outliers} outliers in '{col}'")
        
        # 4. Final check for any remaining nulls
        remaining_nulls = processed_df.null_count()
        total_nulls = sum(remaining_nulls.values())
        
        if total_nulls > 0:
            logger.warning(f"There are still {total_nulls} null values in the dataset")
        else:
            logger.info("No null values remain in the dataset")
        
        # Store the processed data
        self.data_processed = processed_df
        
        # Save processed data
        processed_path = self.config['data']['processed_path']
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        processed_df.write_parquet(processed_path)
        logger.info(f"Saved processed data to {processed_path}")
        
        return processed_df
    
    def split_data(self, df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split the data into training and testing sets.
        
        Args:
            df: DataFrame to split (uses self.data_processed if None)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if df is None:
            df = self.data_processed
            
        if df is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        target_col = self.config['training']['target_column']
        test_size = self.config['models']['test_size']
        random_seed = self.config['models']['random_seed']
        
        logger.info(f"Splitting data with test_size={test_size}")
        
        # Ensure reproducibility
        np.random.seed(random_seed)
        
        # Get feature and target columns
        X = df.drop(target_col)
        y = df.select(target_col)
        
        # Create stratified split
        # To do this, we need to use scikit-learn since Polars doesn't have this built-in
        from sklearn.model_selection import train_test_split
        
        # Convert to pandas first
        X_pd = X.to_pandas()
        y_pd = y.to_pandas()
        
        X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
            X_pd, y_pd, test_size=test_size, random_state=random_seed, 
            stratify=y_pd
        )
        
        # Convert back to Polars
        X_train = pl.from_pandas(X_train_pd)
        X_test = pl.from_pandas(X_test_pd)
        y_train = pl.from_pandas(y_train_pd)
        y_test = pl.from_pandas(y_test_pd)
        
        logger.info(f"Split data into train ({X_train.height} rows) and test ({X_test.height} rows)")
        
        return X_train, X_test, y_train, y_test

def main():
    """Main execution function"""
    processor = ChurnDataProcessor()
    df = processor.load_data()
    processed_df = processor.preprocess_data(df)
    X_train, X_test, y_train, y_test = processor.split_data(processed_df)
    
    logger.info("Data processing completed successfully")

if __name__ == "__main__":
    main() 