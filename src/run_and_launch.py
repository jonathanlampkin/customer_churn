#!/usr/bin/env python
"""
Run the complete ML pipeline and launch the dashboard when ready.
"""
import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(data_path=None):
    """Run the full ML pipeline with the specified data path"""
    try:
        # Build command with data path if provided
        cmd = ["python", "src/run_pipeline.py"]
        if data_path:
            cmd.extend(["--data_path", data_path])
        
        logger.info(f"Starting ML pipeline: {' '.join(cmd)}")
        
        # Run the pipeline and show output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
        # Wait for process to complete
        returncode = process.wait()
        
        if returncode != 0:
            logger.error("ML pipeline failed. Check the logs for details.")
            return False
        
        logger.info("ML pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error running ML pipeline: {str(e)}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        logger.info("Starting Streamlit dashboard...")
        
        # Launch Streamlit dashboard
        subprocess.Popen(["streamlit", "run", "app.py"])
        
        logger.info("Dashboard started! Opening in your browser...")
        return True
        
    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")
        return False

def main():
    """Main entry point for running pipeline and dashboard"""
    # Get data path from command line if provided
    data_path = None
    if len(sys.argv) > 1 and sys.argv[1] == '--data_path' and len(sys.argv) > 2:
        data_path = sys.argv[2]
    
    # Default to the ARFF file path if not specified
    if not data_path:
        data_path = "/mnt/hdd/Documents/mobile_churn_66kx66_numeric_nonull"
        logger.info(f"No data path provided, using default: {data_path}")
    
    # Check if metrics files already exist
    metrics_file = "reports/metrics/final_model_metrics.json"
    if os.path.exists(metrics_file):
        response = input(f"Metrics file {metrics_file} already exists. Run pipeline again? [y/N] ")
        if response.lower() != 'y':
            logger.info("Skipping pipeline, using existing metrics.")
            launch_dashboard()
            return
    
    # Run the pipeline
    pipeline_success = run_pipeline(data_path)
    
    if pipeline_success:
        # Launch the dashboard
        launch_dashboard()
    else:
        logger.error("Pipeline failed. Dashboard not launched.")
        sys.exit(1)

if __name__ == "__main__":
    main() 