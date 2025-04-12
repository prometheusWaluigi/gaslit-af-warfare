# Dockerized GASLIT-AF WARSTACK

This document explains how to use the Docker containerization we've created to solve dependency issues and ensure consistent environment behavior across systems.

## Why Docker?

The project has complex dependencies including scikit-allel for genetic analysis, spaCy for NLP, and other libraries that can be challenging to install correctly in all environments. The specific issue we encountered was with the `scikit-allel` package, which installs properly but has import path inconsistencies between different Python versions and environments.

Docker solves this by:

1. Creating a consistent, isolated environment with all dependencies properly configured
2. Ensuring import paths work correctly regardless of host system setup
3. Handling the specific `scikit-allel` vs `allel` import issue with a patch

## Files Added

- **Dockerfile.fix**: A custom Dockerfile that properly installs dependencies and includes patches for the import issues
- **run_docker.sh**: A helper script to easily build and run the Docker container with various commands
- **README_DOCKER.md**: This documentation

## How to Use

### Prerequisites

- Docker installed on your system
- The VCF files placed in the `data/vcf/` directory (optional - simulation mode will work without them)

### Basic Usage

```bash
# First time: Build the Docker image and run tests
./run_docker.sh --build

# Run VCF analysis (will use real data if available, simulation otherwise)
./run_docker.sh analyze_vcf

# Run the web application
./run_docker.sh app

# Get a shell inside the container
./run_docker.sh shell

# Run any custom Python command
./run_docker.sh "path/to/your/script.py"
```

### Command Line Options

```
Options:
  -h, --help        Show help message
  -b, --build       Build the Docker image (default: use existing if available)
  -r, --rebuild     Force rebuild the Docker image
  -n, --name NAME   Specify container name (default: gaslit-af-container)

Commands:
  tests             Run tests (default)
  analyze_vcf       Run VCF analysis
  app               Run the Flask application
  shell             Start a bash shell inside the container
  command           Any other Python command to run
```

## How It Works

The solution employs several strategies to fix the dependency issues:

1. **Explicit Installation**: We install scikit-allel with a specific version in the Docker container
2. **Import Path Patching**: We modify sys.modules to make `scikit_allel` available when code tries to import it
3. **Custom Import Wrapper**: We add a special import fix module that runs before the main code

## Directory Mapping

The Docker container maps these directories between your host and the container:

- `./data/vcf:/app/data/vcf`: Allows the container to access your VCF files
- `./results:/app/results`: Saves analysis results from the container to your host system

## Troubleshooting

If you encounter issues:

1. Try rebuilding the image with `./run_docker.sh --rebuild`
2. Check that your VCF files are in the correct location
3. Run `./run_docker.sh shell` to get a shell inside the container for debugging
4. Check the output of `python test_imports.py` inside the container

## Development Workflow

For development work:

1. Make changes to your local files
2. Run the Docker container to test your changes
3. Files in mapped volumes (data, results) will persist between container runs
