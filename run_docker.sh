#!/bin/bash
# Script to build and run the GASLIT-AF WARSTACK Docker container

set -e  # Exit on error

# Default command to run inside the container
DEFAULT_CMD="run_tests.py"

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

show_help() {
    echo "GASLIT-AF WARSTACK Docker Runner"
    echo ""
    echo "Usage: $0 [options] [command]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -b, --build       Build the Docker image (default: use existing if available)"
    echo "  -r, --rebuild     Force rebuild the Docker image"
    echo "  -n, --name NAME   Specify container name (default: gaslit-af-container)"
    echo ""
    echo "Commands:"
    echo "  tests             Run tests (default)"
    echo "  analyze_vcf       Run VCF analysis"
    echo "  app               Run the Flask application"
    echo "  shell             Start a bash shell inside the container"
    echo "  command           Any other Python command to run"
    echo ""
    echo "Examples:"
    echo "  $0 --build app                        # Build image and run the app"
    echo "  $0 analyze_vcf                        # Run VCF analysis"
    echo "  $0 shell                              # Start a shell in the container"
    echo "  $0 \"src/genetic_risk/genetic_scanner.py\" # Run a specific script"
    echo ""
}

# Default values
BUILD=false
REBUILD=false
CONTAINER_NAME="gaslit-af-container"
IMAGE_NAME="gaslit-af-warstack"
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
        tests|analyze_vcf|app|shell)
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

# Build the Docker image if requested or if it doesn't exist
if $BUILD || $REBUILD || [ -z "$(docker images -q $IMAGE_NAME 2>/dev/null)" ]; then
    print_header "Building Docker image: $IMAGE_NAME"
    
    if $REBUILD; then
        print_info "Forcing rebuild of the image."
        docker build --no-cache -t $IMAGE_NAME -f Dockerfile.fix .
    else
        docker build -t $IMAGE_NAME -f Dockerfile.fix .
    fi
    
    if [ $? -ne 0 ]; then
        print_error "Docker build failed."
        exit 1
    fi
    
    print_info "Docker image built successfully."
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
    "shell")
        # Override entrypoint to get a shell
        print_header "Starting bash shell in container"
        docker run --rm -it \
            --name $CONTAINER_NAME \
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

print_header "Running GASLIT-AF WARSTACK container"
print_info "Container name: $CONTAINER_NAME"
print_info "Command: $DOCKER_CMD"

# Run the Docker container
docker run --rm -it \
    --name $CONTAINER_NAME \
    -v "$(pwd)/data/vcf:/app/data/vcf" \
    -v "$(pwd)/results:/app/results" \
    -p 5000:5000 \
    $IMAGE_NAME $DOCKER_CMD

print_info "Container finished execution."
