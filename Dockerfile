# Dockerfile for Quant Fund Trading System
FROM python:3.11-slim

# Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/

# NOTE: data/ directory is mounted as volume in docker-compose.yml
# This keeps the image size small and data persistent

# Set Python path
ENV PYTHONPATH=/app/src

# Create logs directory
RUN mkdir -p /app/logs/live

# Health check - runs every hour to verify system is operational
HEALTHCHECK --interval=1h --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Default command - run paper trading
CMD ["python", "scripts/run_live.py", "--mode", "paper"]
