#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Customer Churn Prediction Dashboard ===${NC}"
echo

# Check if Python virtual environment exists, if not create it
if [ ! -d "customer_churn_venv" ]; then
    echo -e "${BLUE}Creating Python virtual environment...${NC}"
    python -m venv customer_churn_venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source customer_churn_venv/bin/activate

# Install required packages if needed
if ! pip show streamlit &>/dev/null; then
    echo -e "${BLUE}Installing required packages...${NC}"
    pip install -r requirements.txt
fi

# Create necessary directories if they don't exist
mkdir -p reports/figures
mkdir -p models

# Run the dashboard
echo -e "${GREEN}Starting dashboard...${NC}"
echo -e "${BLUE}Dashboard will be available at: http://localhost:8501${NC}"
streamlit run app.py

# Deactivate virtual environment when done
deactivate 