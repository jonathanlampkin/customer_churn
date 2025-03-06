"""
Model monitoring system for churn prediction service
Tracks data drift, prediction quality, and system health
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import joblib
import logging
import datetime
import schedule
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'model_monitor_{datetime.datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance, data drift, and system health"""
    
    def __init__(self, config_path: str = 'configs/config.yml'):
        """Initialize the model monitor with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize monitoring data
        self.reference_data = None
        self.models = {}
        self.predictions = []
        self.alerts = []
        
        # Create monitoring directories
        os.makedirs('monitoring/drift', exist_ok=True)
        os.makedirs('monitoring/performance', exist_ok=True)
        os.makedirs('monitoring/alerts', exist_ok=True)
        
        logger.info("Model monitor initialized")

    def load_reference_data(self) -> pd.DataFrame:
        """Load reference data (training data) for drift comparison"""
        # Load from the interim path
        interim_path = self.config['data']['interim_path']
        base_path = os.path.splitext(interim_path)[0]
        
        logger.info(f"Loading reference data from {os.path.dirname(base_path)}")
        
        try:
            import polars as pl
            X_train = pl.read_parquet(f"{base_path}_X_train.parquet")
            
            # Convert to pandas
            self.reference_data = X_train.to_pandas()
            logger.info(f"Reference data loaded with shape: {self.reference_data.shape}")
            
            return self.reference_data
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            return None
    
    def load_production_data(self) -> pd.DataFrame:
        """Load recent production data for drift comparison"""
        # In a real system, this would fetch recent predictions from a database
        # For this example, we'll use the logs directory to find recent predictions
        
        logs_dir = "logs/predictions"
        if not os.path.exists(logs_dir):
            logger.warning(f"Production data directory {logs_dir} does not exist")
            return pd.DataFrame()
        
        # Get the most recent predictions (last 100 files)
        prediction_files = sorted(
            [f for f in os.listdir(logs_dir) if f.endswith('.json')],
            key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)),
            reverse=True
        )[:100]
        
        if not prediction_files:
            logger.warning("No prediction files found")
            return pd.DataFrame()
        
        # Load prediction data
        predictions = []
        for file in prediction_files:
            try:
                with open(os.path.join(logs_dir, file), 'r') as f:
                    pred_data = json.load(f)
                    predictions.append({
                        'prediction_id': pred_data.get('prediction_id'),
                        'timestamp': pred_data.get('timestamp'),
                        'features': pred_data.get('features', {}),
                        'prediction': pred_data.get('prediction'),
                        'probability': pred_data.get('probability')
                    })
            except Exception as e:
                logger.error(f"Error loading prediction file {file}: {e}")
                continue
        
        if not predictions:
            logger.warning("No valid prediction data found")
            return pd.DataFrame()
        
        # Create dataframe
        df_predictions = pd.DataFrame(predictions)
        
        # Expand features column
        if 'features' in df_predictions.columns:
            features_df = pd.json_normalize(df_predictions['features'])
            df_production = pd.concat([df_predictions.drop(columns=['features']), features_df], axis=1)
        else:
            df_production = df_predictions
        
        logger.info(f"Loaded production data with shape: {df_production.shape}")
        return df_production

    def detect_data_drift(self) -> Dict:
        """
        Detect data drift between reference and production data
        
        Returns:
            Dict with drift analysis results
        """
        # Load reference and production data if not already loaded
        if self.reference_data is None:
            self.reference_data = self.load_reference_data()
        
        # Load production data
        production_data = self.load_production_data()
        
        if self.reference_data is None or production_data.empty:
            logger.warning("Cannot perform drift detection without reference and production data")
            return {}
        
        # Find common feature columns
        ref_cols = set(self.reference_data.columns)
        prod_cols = set(production_data.columns)
        common_cols = ref_cols.intersection(prod_cols)
        
        if not common_cols:
            logger.warning("No common features found between reference and production data")
            return {}
        
        # Perform drift detection for each feature
        drift_results = {}
        drifted_features = []
        
        for column in common_cols:
            # Skip non-feature columns
            if column in ['prediction_id', 'timestamp', 'prediction', 'probability']:
                continue
            
            ref_values = self.reference_data[column]
            prod_values = production_data[column]
            
            # Skip columns with all unique values (e.g., IDs)
            if len(ref_values.unique()) == len(ref_values) or len(prod_values.unique()) == len(prod_values):
                drift_results[column] = {"status": "skipped", "reason": "unique_values"}
                continue
            
            # Different tests for numerical vs categorical
            if pd.api.types.is_numeric_dtype(ref_values) and pd.api.types.is_numeric_dtype(prod_values):
                # Kolmogorov-Smirnov test for numerical features
                try:
                    stat, p_value = ks_2samp(ref_values.dropna(), prod_values.dropna())
                    drift_detected = p_value < 0.05
                except Exception as e:
                    logger.warning(f"Error running KS test for {column}: {e}")
                    stat, p_value = np.nan, np.nan
                    drift_detected = False
            else:
                # Convert to category and then to string
                ref_cat = ref_values.astype(str)
                prod_cat = prod_values.astype(str)
                
                # Get value counts
                ref_counts = ref_cat.value_counts().reset_index()
                prod_counts = prod_cat.value_counts().reset_index()
                
                # Merge to get common categories
                merged = pd.merge(
                    ref_counts, 
                    prod_counts, 
                    on='index', 
                    how='outer', 
                    suffixes=('_ref', '_prod')
                ).fillna(0)
                
                # Chi-square test
                try:
                    # Create contingency table
                    obs = np.vstack([merged['count_ref'].values, merged['count_prod'].values])
                    chi2, p_value, _, _ = chi2_contingency(obs)
                    drift_detected = p_value < 0.05
                except Exception as e:
                    logger.warning(f"Error running Chi-square test for {column}: {e}")
                    chi2, p_value = np.nan, np.nan
                    drift_detected = False
            
            # Store results
            result = {
                'test': 'ks_test' if pd.api.types.is_numeric_dtype(ref_values) else 'chi2_test',
                'p_value': float(p_value) if not np.isnan(p_value) else None,
                'drift_detected': drift_detected,
                'ref_mean': float(ref_values.mean()) if pd.api.types.is_numeric_dtype(ref_values) else None,
                'prod_mean': float(prod_values.mean()) if pd.api.types.is_numeric_dtype(prod_values) else None,
                'ref_unique': int(len(ref_values.unique())),
                'prod_unique': int(len(prod_values.unique()))
            }
            
            drift_results[column] = result
            
            if drift_detected:
                drifted_features.append(column)
        
        # Generate drift summary
        drift_summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_features_analyzed': len(drift_results),
            'n_features_drifted': len(drifted_features),
            'drift_ratio': len(drifted_features) / len(drift_results) if drift_results else 0,
            'drifted_features': drifted_features,
            'feature_details': drift_results
        }
        
        # Generate alert if too many features are drifting
        if drift_summary['drift_ratio'] > 0.3:
            self.generate_alert(
                alert_type="DATA_DRIFT",
                severity="HIGH",
                message=f"Severe data drift detected in {drift_summary['n_features_drifted']} features",
                details=drift_summary
            )
        elif drift_summary['drift_ratio'] > 0.1:
            self.generate_alert(
                alert_type="DATA_DRIFT",
                severity="MEDIUM",
                message=f"Moderate data drift detected in {drift_summary['n_features_drifted']} features",
                details=drift_summary
            )
        
        # Save drift report
        report_path = f"monitoring/drift/drift_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(drift_summary, f, indent=2)
        
        logger.info(f"Data drift analysis completed. Results saved to {report_path}")
        
        # Visualize drift
        self.visualize_data_drift(drift_summary)
        
        return drift_summary 

    def visualize_data_drift(self, drift_summary: Dict) -> None:
        """Create visualizations for data drift"""
        # Create bar chart of drifted features
        feature_details = drift_summary['feature_details']
        
        # Extract p-values for all features
        features = []
        p_values = []
        is_drifted = []
        
        for feature, details in feature_details.items():
            if 'p_value' in details and details['p_value'] is not None:
                features.append(feature)
                p_values.append(details['p_value'])
                is_drifted.append(details.get('drift_detected', False))
        
        if not features:
            logger.warning("No valid features to visualize drift")
            return
        
        # Sort by p-value
        sorted_indices = np.argsort(p_values)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_p_values = [p_values[i] for i in sorted_indices]
        sorted_is_drifted = [is_drifted[i] for i in sorted_indices]
        
        # Plot p-values
        plt.figure(figsize=(14, 10))
        bars = plt.barh(
            sorted_features[:20],  # Show top 20 features
            sorted_p_values[:20],
            color=[
                'red' if drifted else 'green' 
                for drifted in sorted_is_drifted[:20]
            ]
        )
        
        plt.axvline(x=0.05, color='black', linestyle='--', label='Significance Level (p=0.05)')
        plt.xlabel('p-value')
        plt.ylabel('Feature')
        plt.title('Feature Drift Analysis (p-values)')
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        drift_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'monitoring/drift/p_values_{drift_time}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a drift heatmap for numeric features
        numeric_drifted = {}
        
        for feature, details in feature_details.items():
            if details.get('ref_mean') is not None and details.get('prod_mean') is not None:
                numeric_drifted[feature] = {
                    'ref_mean': details['ref_mean'],
                    'prod_mean': details['prod_mean'],
                    'percent_change': abs(details['prod_mean'] - details['ref_mean']) / (abs(details['ref_mean']) + 1e-10) * 100
                }
        
        if numeric_drifted:
            # Create dataframe for heatmap
            drift_df = pd.DataFrame.from_dict(numeric_drifted, orient='index')
            drift_df = drift_df.sort_values('percent_change', ascending=False)
            
            # Plot heatmap of percent changes
            plt.figure(figsize=(10, max(8, len(numeric_drifted) * 0.4)))
            ax = sns.heatmap(
                drift_df[['percent_change']].head(20), 
                annot=True, 
                fmt=".1f",
                cmap="YlOrRd"
            )
            plt.title('Feature Drift - Percent Change in Mean Values')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(f'monitoring/drift/mean_changes_{drift_time}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Data drift visualizations created")
    
    def monitor_model_health(self) -> Dict:
        """
        Monitor model health metrics
        
        Returns:
            Dict with health metrics
        """
        # Get list of model files
        model_dir = 'models'
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} does not exist")
            return {}
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        
        if not model_files:
            logger.warning("No model files found")
            return {}
        
        # Get latest model file
        latest_model_file = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model_file)
        
        try:
            # Load model metadata
            metadata_dir = os.path.join(model_dir, 'metadata')
            if os.path.exists(metadata_dir):
                metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.yml')]
                if metadata_files:
                    latest_metadata_file = sorted(metadata_files)[-1]
                    with open(os.path.join(metadata_dir, latest_metadata_file), 'r') as f:
                        model_metadata = yaml.safe_load(f)
                else:
                    model_metadata = {}
            else:
                model_metadata = {}
            
            # Check model file stats
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # size in MB
            model_age = (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(model_path))).days
            
            # Load model to check loading time
            start_time = time.time()
            model = joblib.load(model_path)
            loading_time = time.time() - start_time
            
            # Try a sample prediction if test data is available
            prediction_time = None
            if hasattr(self, 'reference_data') and self.reference_data is not None and not self.reference_data.empty:
                sample = self.reference_data.iloc[[0]]
                start_time = time.time()
                try:
                    _ = model.predict_proba(sample)
                    prediction_time = time.time() - start_time
                except Exception as e:
                    logger.warning(f"Error making test prediction: {e}")
            
            # Get performance metrics from metadata
            performance_metrics = model_metadata.get('metrics', {})
            
            # Compile health metrics
            health_metrics = {
                'model_name': latest_model_file,
                'model_size_mb': model_size,
                'model_age_days': model_age,
                'loading_time_seconds': loading_time,
                'prediction_time_seconds': prediction_time,
                'performance_metrics': performance_metrics
            }
            
            # Generate alert if model is too old
            if model_age > 30:  # Alert if model is older than 30 days
                self.generate_alert(
                    alert_type="MODEL_AGE",
                    severity="MEDIUM",
                    message=f"Model is {model_age} days old - consider retraining",
                    details={'model_name': latest_model_file, 'model_age_days': model_age}
                )
            
            # Save health report
            report_path = f"monitoring/performance/health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                # Convert non-serializable values
                health_metrics_json = {k: v if isinstance(v, (dict, list, str, int, float, bool, type(None))) else str(v) 
                                       for k, v in health_metrics.items()}
                json.dump(health_metrics_json, f, indent=2)
            
            logger.info(f"Model health report created: {report_path}")
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring model health: {e}")
            return {'error': str(e)}
    
    def generate_alert(self, alert_type: str, severity: str, message: str, details: Dict = None) -> Dict:
        """
        Generate and save an alert
        
        Args:
            alert_type: Type of alert (e.g., DATA_DRIFT, MODEL_HEALTH)
            severity: Alert severity (LOW, MEDIUM, HIGH)
            message: Alert message
            details: Additional alert details
            
        Returns:
            Dict with alert information
        """
        alert_id = f"{alert_type.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = {
            'id': alert_id,
            'type': alert_type,
            'severity': severity,
            'message': message,
            'timestamp': datetime.datetime.now().isoformat(),
            'details': details or {}
        }
        
        # Add to alerts list
        self.alerts.append(alert)
        
        # Save alert to file
        alerts_dir = 'monitoring/alerts'
        os.makedirs(alerts_dir, exist_ok=True)
        
        alert_path = os.path.join(alerts_dir, f"{alert['id']}.json")
        with open(alert_path, 'w') as f:
            json.dump(alert, f, indent=2)
        
        logger.warning(f"Alert generated: [{severity}] {message}")
        
        return alert 

    def run_monitoring_cycle(self) -> Dict:
        """
        Run a complete monitoring cycle
        
        Returns:
            Dict with monitoring results
        """
        logger.info("Starting monitoring cycle")
        
        # Clear previous alerts
        self.alerts = []
        
        # Monitor model health
        health_metrics = self.monitor_model_health()
        
        # Detect data drift
        drift_summary = self.detect_data_drift()
        
        # Return combined results
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'health_metrics': health_metrics,
            'drift_summary': drift_summary,
            'alerts': self.alerts
        }
        
        logger.info(f"Monitoring cycle completed with {len(self.alerts)} alerts")
        
        return results

def schedule_monitoring(interval_hours: int = 24):
    """
    Schedule periodic monitoring
    
    Args:
        interval_hours: Interval between monitoring runs in hours
    """
    monitor = ModelMonitor()
    
    def run_monitoring():
        monitor.run_monitoring_cycle()
    
    # Run once immediately
    run_monitoring()
    
    # Schedule regular runs
    schedule.every(interval_hours).hours.do(run_monitoring)
    
    logger.info(f"Monitoring scheduled every {interval_hours} hours")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model monitoring system')
    parser.add_argument('--run-once', action='store_true', help='Run monitoring once and exit')
    parser.add_argument('--interval', type=int, default=24, help='Monitoring interval in hours')
    
    args = parser.parse_args()
    
    if args.run_once:
        monitor = ModelMonitor()
        monitor.run_monitoring_cycle()
    else:
        schedule_monitoring(args.interval)

if __name__ == "__main__":
    main() 