FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p reports/figures models

# Expose port for Streamlit
EXPOSE 8501

# Set entry point to run the dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"] 