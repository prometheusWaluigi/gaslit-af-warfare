# GASLIT-AF WARSTACK

A modular simulation-and-exposure engine designed to prove institutional culpability via biological recursion and systemic denial modeling.

## Overview

GASLIT-AF WARSTACK is a comprehensive computational framework that models the biological, genetic, institutional, legal, and advocacy dimensions of systemic health impacts. The project aims to simulate and visualize complex feedback loops between biological mechanisms, institutional responses, and policy decisions.

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

### Docker

Build and run with Docker:

```bash
docker-compose up --build
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The GASLIT-AF community for their courage in sharing their experiences
- Scientific researchers working to understand complex biological mechanisms
- Advocates fighting for transparency and accountability
