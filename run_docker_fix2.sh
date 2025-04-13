#!/bin/bash

# Stop and remove any existing container
echo "[INFO] Stopping and removing any existing container..."
docker stop gaslit-af-container 2>/dev/null || true
docker rm gaslit-af-container 2>/dev/null || true

# Count VCF files in data/vcf directory
VCF_COUNT=$(find data/vcf -name "*.vcf" 2>/dev/null | wc -l)
echo "[INFO] Found $VCF_COUNT VCF files in data/vcf directory."

# Build the Docker image with the fixed Dockerfile
echo "[INFO] Building Docker image with fixed Dockerfile..."
docker build -t gaslit-af-warstack:fix2 -f Dockerfile.fix2 .

echo ""
echo "Running GASLIT-AF WARSTACK container with fixed Dockerfile"
echo "=============================================="
echo "[INFO] Container name: gaslit-af-container"
echo "[INFO] Command: run_tests.py"

# Run the container
docker run --name gaslit-af-container \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  gaslit-af-warstack:fix2 run_tests.py

# After tests, run the application
echo ""
echo "Starting the application..."
echo "=============================================="
docker stop gaslit-af-container 2>/dev/null || true
docker rm gaslit-af-container 2>/dev/null || true

docker run --name gaslit-af-container \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  -d \
  gaslit-af-warstack:fix2 src/frontend/app.py

echo "[INFO] Application running at http://localhost:5000"
