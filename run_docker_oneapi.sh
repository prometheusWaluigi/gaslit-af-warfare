#!/bin/bash
# Script to build and run the GASLIT-AF WARSTACK Docker container with Intel OneAPI support

set -e  # Exit on error

# Default command to run inside the container
DEFAULT_CMD="run_tests.py"

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}==============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}==============================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_intel() {
    echo -e "${CYAN}[INTEL]${NC} $1"
}

show_help() {
    echo "GASLIT-AF WARSTACK Docker Runner with Intel OneAPI"
    echo ""
    echo "Usage: $0 [options] [command]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -b, --build       Build the Docker image (default: use existing if available)"
    echo "  -r, --rebuild     Force rebuild the Docker image"
    echo "  -n, --name NAME   Specify container name (default: gaslit-af-oneapi)"
    echo "  -c, --cpu-only    Disable GPU acceleration and use CPU only"
    echo ""
    echo "Commands:"
    echo "  tests             Run tests (default)"
    echo "  analyze_vcf       Run VCF analysis"
    echo "  app               Run the Flask application"
    echo "  shell             Start a bash shell inside the container"
    echo "  benchmark         Run performance benchmarks"
    echo "  command           Any other Python command to run"
    echo ""
    echo "Examples:"
    echo "  $0 --build app                        # Build image and run the app"
    echo "  $0 analyze_vcf                        # Run VCF analysis"
    echo "  $0 shell                              # Start a shell in the container"
    echo "  $0 --cpu-only tests                   # Run tests without GPU acceleration"
    echo ""
}

# Default values
BUILD=false
REBUILD=false
CPU_ONLY=false
CONTAINER_NAME="gaslit-af-oneapi"
IMAGE_NAME="gaslit-af-warstack-oneapi"
COMMAND=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -r|--rebuild)
            REBUILD=true
            BUILD=true
            shift
            ;;
        -n|--name)
            CONTAINER_NAME=$2
            shift 2
            ;;
        -c|--cpu-only)
            CPU_ONLY=true
            shift
            ;;
        tests|analyze_vcf|app|shell|benchmark)
            COMMAND=$1
            shift
            ;;
        *)
            # Assume anything else is a command to run
            COMMAND=$1
            shift
            ;;
    esac
done

# Make sure data/vcf directory exists
mkdir -p data/vcf

# Check if VCF files are present
VCF_COUNT=$(ls -1 data/vcf/*.vcf.gz 2>/dev/null | wc -l)
if [ "$VCF_COUNT" -eq 0 ]; then
    print_info "No VCF files found in data/vcf directory."
    print_info "The analysis will run in simulation mode."
else
    print_info "Found $VCF_COUNT VCF files in data/vcf directory."
fi

# Check for Intel GPU
if ! $CPU_ONLY; then
    if command -v sycl-ls &> /dev/null; then
        print_intel "Checking for Intel GPU with sycl-ls..."
        GPU_INFO=$(sycl-ls 2>/dev/null | grep -i "Intel.*Graphics")
        if [ -n "$GPU_INFO" ]; then
            print_intel "Intel GPU detected: $GPU_INFO"
        else
            print_intel "No Intel GPU detected. Will use CPU fallback."
        fi
    else
        print_intel "sycl-ls not found. Cannot verify Intel GPU presence."
        print_intel "If you have an Intel GPU, make sure OneAPI is installed on the host."
    fi
fi

# Build the Docker image if requested or if it doesn't exist
if $BUILD || $REBUILD || [ -z "$(docker images -q $IMAGE_NAME 2>/dev/null)" ]; then
    print_header "Building Docker image with Intel OneAPI: $IMAGE_NAME"
    
    if $REBUILD; then
        print_info "Forcing rebuild of the image."
        docker build --no-cache -t $IMAGE_NAME -f Dockerfile.oneapi.fix2 .
    else
        docker build -t $IMAGE_NAME -f Dockerfile.oneapi.fix2 .
    fi
    
    if [ $? -ne 0 ]; then
        print_error "Docker build failed."
        exit 1
    fi
    
    print_info "Docker image built successfully."
fi

# Set up environment variables for container
DOCKER_ENV_VARS=()

# Configure for CPU-only mode if requested
if $CPU_ONLY; then
    print_intel "Running in CPU-only mode (GPU disabled)"
    DOCKER_ENV_VARS+=("-e" "ONEAPI_DEVICE_SELECTOR=opencl:cpu")
    DOCKER_ENV_VARS+=("-e" "USE_INTEL_EXTENSION=0")
else
    print_intel "Running with GPU acceleration enabled"
    DOCKER_ENV_VARS+=("-e" "ONEAPI_DEVICE_SELECTOR=level_zero:gpu")
    DOCKER_ENV_VARS+=("-e" "USE_INTEL_EXTENSION=1")
fi

# Map the command to the proper Docker command
case $COMMAND in
    "")
        # Default command
        DOCKER_CMD=$DEFAULT_CMD
        ;;
    "tests")
        DOCKER_CMD="run_tests.py"
        ;;
    "analyze_vcf")
        DOCKER_CMD="fixed_analyze_vcf.py"
        ;;
    "app")
        DOCKER_CMD="src/frontend/app.py --host 0.0.0.0 --port 5000"
        ;;
    "benchmark")
        # Create a simple benchmark script if it doesn't exist
        if [ ! -f "benchmark.py" ]; then
            cat > benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
Benchmark script for GASLIT-AF WARSTACK with Intel optimizations.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.biological_modeling.neuroimmune_simulator import NeuroimmuneDynamics

def run_benchmark(use_intel=True):
    """Run performance benchmark with optional Intel optimizations."""
    print(f"Running benchmark with Intel optimizations: {use_intel}")
    
    # Configure environment
    if use_intel:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = 'level_zero:gpu'
        os.environ['USE_INTEL_EXTENSION'] = '1'
    else:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = 'opencl:cpu'
        os.environ['USE_INTEL_EXTENSION'] = '0'
    
    # Configuration for benchmark
    config = {
        'grid_size': 512,  # Larger grid for benchmark
        'time_steps': 100,
        'dt': 0.1,
        'noise_strength': 0.1,
        'diffusion_constant': 0.5,
        'reaction_rate': 1.0,
        'coupling_strength': 0.8,
        'initial_condition': 'random',
        'boundary_condition': 'periodic',
        'use_hardware_acceleration': use_intel,
        'output_dir': 'results/benchmark',
        'random_seed': 42
    }
    
    # Create simulator
    simulator = NeuroimmuneDynamics(config)
    simulator.initialize_grid()
    
    # Run simulation and time it
    start_time = time.time()
    simulator.run_simulation()
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")
    
    return elapsed

if __name__ == "__main__":
    # Create output directory
    os.makedirs("results/benchmark", exist_ok=True)
    
    # Run with Intel optimizations
    print("\n=== Running with Intel optimizations ===")
    intel_time = run_benchmark(use_intel=True)
    
    # Run without Intel optimizations
    print("\n=== Running without Intel optimizations ===")
    standard_time = run_benchmark(use_intel=False)
    
    # Calculate speedup
    speedup = standard_time / intel_time if intel_time > 0 else 0
    
    print("\n=== Benchmark Results ===")
    print(f"Standard execution time: {standard_time:.2f} seconds")
    print(f"Intel optimized time:    {intel_time:.2f} seconds")
    print(f"Speedup factor:          {speedup:.2f}x")
    
    # Create a simple bar chart
    labels = ['Standard', 'Intel Optimized']
    times = [standard_time, intel_time]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['blue', 'cyan'])
    plt.ylabel('Execution Time (seconds)')
    plt.title('GASLIT-AF Performance Benchmark')
    
    # Add time labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # Add speedup annotation
    plt.annotate(f'Speedup: {speedup:.2f}x',
                xy=(1, intel_time), 
                xytext=(0.75, intel_time/2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig('results/benchmark/performance_comparison.png')
    print("Benchmark chart saved to results/benchmark/performance_comparison.png")
EOF
            chmod +x benchmark.py
            print_info "Created benchmark.py script"
        fi
        DOCKER_CMD="benchmark.py"
        ;;
    "shell")
        # Override entrypoint to get a shell
        print_header "Starting bash shell in container with Intel OneAPI"
        docker run --rm -it \
            --name $CONTAINER_NAME \
            "${DOCKER_ENV_VARS[@]}" \
            --device /dev/dri \
            -v "$(pwd)/data/vcf:/app/data/vcf" \
            -v "$(pwd)/results:/app/results" \
            -p 5000:5000 \
            --entrypoint /bin/bash \
            $IMAGE_NAME
        exit $?
        ;;
    *)
        # Use the command as is
        DOCKER_CMD=$COMMAND
        ;;
esac

print_header "Running GASLIT-AF WARSTACK container with Intel OneAPI"
print_info "Container name: $CONTAINER_NAME"
print_info "Command: $DOCKER_CMD"
if $CPU_ONLY; then
    print_intel "Hardware acceleration: DISABLED (CPU only)"
else
    print_intel "Hardware acceleration: ENABLED (Intel GPU)"
fi

# Run the Docker container
docker run --rm -it \
    --name $CONTAINER_NAME \
    "${DOCKER_ENV_VARS[@]}" \
    --device /dev/dri \
    -v "$(pwd)/data/vcf:/app/data/vcf" \
    -v "$(pwd)/results:/app/results" \
    -p 5000:5000 \
    $IMAGE_NAME $DOCKER_CMD

print_info "Container finished execution."
