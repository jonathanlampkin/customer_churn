"""
Modified EDA script to use external HDD for storage with sudo
"""
import os
import sys
import logging
import subprocess

# Configure logging to external HDD (use subprocess for sudo)
subprocess.run(["sudo", "mkdir", "-p", "/mnt/hdd/churn_project/logs"], check=True)
subprocess.run(["sudo", "chmod", "777", "/mnt/hdd/churn_project/logs"], check=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/mnt/hdd/churn_project/logs/eda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Make sure reports directory exists on external drive with sudo
subprocess.run(["sudo", "mkdir", "-p", "/mnt/hdd/churn_project/reports/figures/eda"], check=True)
subprocess.run(["sudo", "chmod", "777", "/mnt/hdd/churn_project/reports/figures/eda"], check=True)

# Import the original EDA script
import EDA

# Custom mkdir function to use sudo
def sudo_mkdir(path, **kwargs):
    # Replace 'reports/' with the external path
    if path.startswith('reports/'):
        external_path = path.replace('reports/', '/mnt/hdd/churn_project/reports/')
        subprocess.run(["sudo", "mkdir", "-p", external_path], check=True)
        # Make the directory writable
        subprocess.run(["sudo", "chmod", "777", external_path], check=True)
    else:
        # Handle other directories
        subprocess.run(["sudo", "mkdir", "-p", path], check=True)
        subprocess.run(["sudo", "chmod", "777", path], check=True)

# Override the mkdir function in EDA
EDA.os.makedirs = sudo_mkdir

def main():
    """Run EDA with external HDD storage using sudo"""
    logger.info("Starting EDA with external HDD storage")
    
    # Run the main EDA function
    try:
        EDA.main()
        logger.info("EDA completed successfully")
    except Exception as e:
        logger.error(f"Error in EDA: {e}", exc_info=True)
    
if __name__ == "__main__":
    main()
