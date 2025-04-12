# GASLIT-AF WARSTACK

A modular simulation-and-exposure engine designed to prove institutional culpability via biological recursion and systemic denial modeling.

## What is GASLIT-AF WARSTACK?

GASLIT-AF WARSTACK is a comprehensive computational framework that models the biological, genetic, institutional, legal, and advocacy dimensions of systemic health impacts. The project aims to simulate and visualize complex feedback loops between biological mechanisms, institutional responses, and policy decisions.

In plain terms, this project:
- Simulates how biological mechanisms (like spike protein neurotoxicity) affect health
- Analyzes genetic risk factors from genome data
- Models how institutions respond to (or deny) emerging health issues
- Simulates legal and policy implications
- Provides a platform for advocacy and data visualization

## Architecture

The project is organized into five main modules:

### 1. Biological Modeling Layer

Simulates the core GASLIT-AF attractor states: spike protein neurotoxicity → cerebellar trauma → behavioral and autonomic collapse.

- **Stack**: Python (NumPy, SciPy, PyTorch), SYCL/OneAPI for Intel A770, C++ if performance needed
- **Deliverables**:
  - KPZ / fKPZχ simulations of neuroimmune dynamics
  - ODE/PDE attractor maps showing irreversible system transitions
  - Phase portraits of feedback loop entrapment

### 2. Genetic Risk Scanning Layer

Parse FASTQ/VCF files for fragility architecture (γ) and allostatic collapse risk (Λ, Ω).

- **Stack**: Python (scikit-allel, Biopython), Intel OneAPI, AVX2 optimization, Dockerized CLI pipelines
- **Deliverables**:
  - Heatmaps of variant fragility (e.g., TNXB, COMT, MTHFR, RCCX)
  - GASLIT-AF phenotype index per genome
  - Exportable JSON risk profiles for downstream modules

### 3. Institutional Feedback Modeling

Build dynamic denial-injury-denial loops, regulatory capture graphs, and memetic immunosuppression nets.

- **Stack**: Python (NetworkX, DynSysLib), GPT-in-the-loop for narrative evolution, Graphviz for causal webs
- **Deliverables**:
  - Denial recursion maps
  - System legitimacy entropy index
  - Attractor state simulation for CDC/FDA/Media narrative collapse

### 4. Legal & Policy Simulation Layer

Run simulation logic against the legal narrative layer: NCVIA, EUA claims, suppressed data trails, ethical obligations.

- **Stack**: PyPDF2, LangChain, spaCy, HuggingFace (legal LLMs), graph-based legal precedents
- **Deliverables**:
  - Predictive simulations for class action viability
  - "Liability Shield Breach" risk thresholds
  - Timelines of known suppression vs scientific publishing lags

### 5. Front-End Advocacy Layer

Make it real. n=1 testimonies, real-time genetic risk visualizations, attractor dashboards.

- **Stack**: Flask, Bootstrap, D3.js
- **Deliverables**:
  - Dynamic dashboard of systemic harm
  - User-uploadable genome analysis (consented)
  - "Tell Your Story" portal that maps symptoms → loops → systems → blame

## Installation

### Prerequisites

- Python 3.8+
- Intel OneAPI (optional, for hardware acceleration)
- Docker (optional, for containerized deployment)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gaslit-af-warstack.git
   cd gaslit-af-warstack
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required models (optional):
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Command Line Interface

The project provides a CLI for running simulations:

```bash
./gaslit-af-runner.py --module biological --visualize
```

Available modules:
- `biological`: Run biological modeling simulations
- `genetic`: Analyze genetic risk from VCF/FASTQ files
- `institutional`: Run institutional feedback modeling
- `legal`: Simulate legal and policy dynamics
- `all`: Run all modules

For more options:
```bash
./gaslit-af-runner.py --help
```

### Web Interface

Start the web server:

```bash
cd src/frontend
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

## Development

### Testing

Run tests with pytest:

```bash
pytest
```

Or use the provided test runner:

```bash
./run_tests.py
```

## Docker Support

The project offers multiple Docker configurations to solve dependency issues and ensure consistent environment behavior across systems:

### 1. Standard Docker Setup

Basic Docker setup with all dependencies properly configured:

```bash
# Build and run the container
./run_docker.sh --build

# Run VCF analysis
./run_docker.sh analyze_vcf

# Run the web application
./run_docker.sh app

# Get a shell inside the container
./run_docker.sh shell
```

### 2. Intel OneAPI Docker Setup

Optimized for Intel hardware (especially Arc A770 GPU) with OneAPI acceleration:

```bash
# Build and run with Intel GPU acceleration
./run_docker_oneapi.sh --build

# Run with CPU-only mode
./run_docker_oneapi.sh --cpu-only

# Run performance benchmarks
./run_docker_oneapi.sh benchmark
```

### 3. Multi-Container Setup with Docker Compose

Complete application stack with web server, database, and optional Jupyter notebook:

```bash
# Start all services
docker-compose up --build

# Start specific services
docker-compose up app db

# Access the application at http://localhost
```

The Docker Compose setup includes:
- **app**: Main Flask application
- **db**: PostgreSQL database
- **nginx**: NGINX web server
- **jupyter**: Jupyter Notebook for data analysis (optional)

## Why Docker?

The project has complex dependencies including:
- scikit-allel for genetic analysis
- spaCy for NLP
- PyTorch with Intel extensions
- Various scientific computing libraries

Docker solves several issues:
1. Creates a consistent, isolated environment with all dependencies properly configured
2. Ensures import paths work correctly regardless of host system setup
3. Handles specific import path inconsistencies between different Python versions
4. Enables hardware acceleration with Intel OneAPI when available

## Directory Structure

- `src/`: Source code for all modules
  - `biological_modeling/`: Biological simulation code
  - `genetic_risk/`: Genetic analysis code
  - `institutional_feedback/`: Institutional modeling code
  - `legal_policy/`: Legal simulation code
  - `frontend/`: Web interface code
- `tests/`: Test suite
- `data/`: Data directory (VCF files, etc.)
- `docker/`: Docker configuration files
- `docs/`: Documentation

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The GASLIT-AF community for their courage in sharing their experiences
- Scientific researchers working to understand complex biological mechanisms
- Advocates fighting for transparency and accountability
