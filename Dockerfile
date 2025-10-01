# Dockerfile for Quant Fund Trading System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app/src

# Create logs directory
RUN mkdir -p /app/logs/live

# Default command (can be overridden)
CMD ["python", "scripts/run_live.py", "--mode", "paper"]
