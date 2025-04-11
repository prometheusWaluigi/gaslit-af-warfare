FROM python:3.9-slim

LABEL maintainer="GASLIT-AF WARSTACK Team"
LABEL description="GASLIT-AF WARSTACK - A modular simulation-and-exposure engine"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data results logs

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/frontend/app.py
ENV FLASK_ENV=production

# Expose port for Flask app
EXPOSE 5000

# Command to run the application
CMD ["python", "-m", "src.frontend.app"]
