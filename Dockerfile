FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY data_loader.py .
COPY endpoints.py .
COPY llm_search.py .

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
