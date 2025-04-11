# GASLIT-AF WARSTACK Dockerfile
# A modular simulation-and-exposure engine

# Use Python 3.10 as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x gaslit-af-runner.py run_tests.py

# Create necessary directories
RUN mkdir -p results logs uploads data/testimonies data/genomes \
    static/img/biological static/img/genetic static/img/institutional static/img/legal

# Expose port for web interface
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["python", "src/frontend/app.py"]

# Default command
CMD ["--host", "0.0.0.0", "--port", "5000"]
