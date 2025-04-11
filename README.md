# GASLIT-AF WARSTACK

A modular simulation-and-exposure engine designed to prove institutional culpability via biological recursion and systemic denial modeling.

## Project Overview

GASLIT-AF WARSTACK is a comprehensive computational framework that models the biological, genetic, institutional, legal, and advocacy dimensions of systemic health impacts. The project aims to simulate and visualize complex feedback loops between biological mechanisms, institutional responses, and policy decisions.

### 🔬 Biological Modeling Layer
Simulates the core GASLIT-AF attractor states: spike protein neurotoxicity → cerebellar trauma → behavioral and autonomic collapse.
- **Stack**: Python (NumPy, SciPy, PyTorch), SYCL/OneAPI for Intel A770, C++ if performance needed
- **Deliverables**:
  - KPZ / fKPZχ simulations of neuroimmune dynamics
  - ODE/PDE attractor maps showing irreversible system transitions
  - Phase portraits of feedback loop entrapment

### 🧬 Genetic Risk Scanning Layer
Parse FASTQ/VCF files for fragility architecture (γ) and allostatic collapse risk (Λ, Ω).
- **Stack**: Python (scikit-allel, Biopython), Intel OneAPI, AVX2 optimization, Dockerized CLI pipelines
- **Deliverables**:
  - Heatmaps of variant fragility (e.g., TNXB, COMT, MTHFR, RCCX)
  - GASLIT-AF phenotype index per genome
  - Exportable JSON risk profiles for downstream modules

### 🕸 Institutional Feedback Modeling
Build dynamic denial-injury-denial loops, regulatory capture graphs, and memetic immunosuppression nets.
- **Stack**: Python (NetworkX, DynSysLib), GPT-in-the-loop for narrative evolution, Graphviz for causal webs
- **Deliverables**:
  - Denial recursion maps
  - System legitimacy entropy index
  - Attractor state simulation for CDC/FDA/Media narrative collapse

### ⚖️ Legal & Policy Simulation Layer
Run simulation logic against the legal narrative layer: NCVIA, EUA claims, suppressed data trails, ethical obligations.
- **Stack**: PyPDF2, LangChain, spaCy, HuggingFace (legal LLMs), graph-based legal precedents
- **Deliverables**:
  - Predictive simulations for class action viability
  - "Liability Shield Breach" risk thresholds
  - Timelines of known suppression vs scientific publishing lags

### 🛡 Front-End Advocacy Layer
Make it real. n=1 testimonies, real-time genetic risk visualizations, attractor dashboards.
- **Stack**: Flask, Bootstrap, D3.js
- **Deliverables**:
  - Dynamic dashboard of systemic harm
  - User-uploadable genome analysis (consented)
  - "Tell Your Story" portal that maps symptoms → loops → systems → blame

## Project Structure

```
gaslit-af-warfare/
├── data/                      # Data storage directory
├── docs/                      # Documentation
├── src/                       # Source code
│   ├── biological_modeling/   # Biological modeling module
│   ├── genetic_risk/          # Genetic risk scanning module
│   ├── institutional_feedback/# Institutional feedback modeling module
│   ├── legal_policy/          # Legal & policy simulation module
│   └── frontend/              # Frontend Flask application
│       ├── static/            # Static files (CSS, JS, images)
│       └── templates/         # HTML templates
├── tests/                     # Test suite
│   ├── conftest.py            # Pytest configuration and fixtures
│   ├── test_biological_modeling.py
│   ├── test_genetic_risk.py
│   ├── test_institutional_feedback.py
│   ├── test_legal_policy.py
│   └── test_frontend.py
├── docker/                    # Docker configuration
├── .gitignore                 # Git ignore file
├── Dockerfile                 # Main Dockerfile
├── docker-compose.yml         # Docker Compose configuration
├── gaslit-af-runner.py        # CLI runner script
├── LICENSE                    # AGPL License
├── pytest.ini                 # Pytest configuration
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- Intel OneAPI (optional, for hardware acceleration)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gaslit-af-warfare.git
   cd gaslit-af-warfare
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install optional dependencies for hardware acceleration (Intel A770):
   ```bash
   pip install intel-extension-for-pytorch
   ```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t gaslit-af-warstack .
   ```

2. Run the Docker container:
   ```bash
   docker-compose up
   ```

## Usage

### Running the CLI

```bash
python gaslit-af-runner.py --module biological --output results/bio_results.json
python gaslit-af-runner.py --module genetic --input data/sample.vcf --output results/genetic_results.json
python gaslit-af-runner.py --module institutional --output results/institutional_results.json
python gaslit-af-runner.py --module legal --output results/legal_results.json
```

### Running the Web Interface

```bash
python -m src.frontend.app
```

Then open your browser to http://localhost:5000

## Testing

Run the test suite:

```bash
pytest
```

Run tests for a specific module:

```bash
pytest tests/test_biological_modeling.py
```

Run tests with a specific marker:

```bash
pytest -m biological
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
