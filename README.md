# GASLIT-AF WARSTACK

A modular simulation-and-exposure engine designed to prove institutional culpability via biological recursion and systemic denial modeling.

## Project Overview

GASLIT-AF WARSTACK is a comprehensive computational framework that models the biological, genetic, institutional, legal, and advocacy dimensions of systemic health impacts. The project aims to simulate and visualize complex feedback loops between biological mechanisms, institutional responses, and policy decisions.

## Architecture

The project is structured in five interconnected layers:

### 🔬 1. Biological Modeling Layer

Simulates the core GASLIT-AF attractor states: spike protein neurotoxicity → cerebellar trauma → behavioral and autonomic collapse.

**Technologies:**
- Python (NumPy, SciPy, PyTorch)
- SYCL/OneAPI for Intel A770 acceleration
- C++ for performance-critical components

**Key Components:**
- KPZ / fKPZχ simulations of neuroimmune dynamics
- ODE/PDE attractor maps showing irreversible system transitions
- Phase portraits of feedback loop entrapment

### 🧬 2. Genetic Risk Scanning Layer

Parse FASTQ/VCF files for fragility architecture (γ) and allostatic collapse risk (Λ, Ω).

**Technologies:**
- Python (scikit-allel, Biopython)
- Intel OneAPI
- AVX2 optimization
- Dockerized CLI pipelines

**Key Components:**
- Heatmaps of variant fragility (e.g., TNXB, COMT, MTHFR, RCCX)
- GASLIT-AF phenotype index per genome
- Exportable JSON risk profiles for downstream modules

### 🕸 3. Institutional Feedback Modeling

Build dynamic denial-injury-denial loops, regulatory capture graphs, and memetic immunosuppression nets.

**Technologies:**
- Python (NetworkX, DynSysLib)
- GPT-in-the-loop for narrative evolution
- Graphviz for causal webs

**Key Components:**
- Denial recursion maps
- System legitimacy entropy index
- Attractor state simulation for CDC/FDA/Media narrative collapse

### ⚖️ 4. Legal & Policy Simulation Layer

Run simulation logic against the legal narrative layer: NCVIA, EUA claims, suppressed data trails, ethical obligations.

**Technologies:**
- PyPDF2, LangChain, spaCy
- HuggingFace (legal LLMs)
- Graph-based legal precedents

**Key Components:**
- Predictive simulations for class action viability
- "Liability Shield Breach" risk thresholds
- Timelines of known suppression vs scientific publishing lags

### 🛡 5. Front-End Advocacy Layer

Make it real. n=1 testimonies, real-time genetic risk visualizations, attractor dashboards.

**Technologies:**
- React + Tailwind (Next.js)
- D3.js
- HuggingFace Spaces for story-driven AI overlays

**Key Components:**
- Dynamic dashboard of systemic harm
- User-uploadable genome analysis (consented)
- "Tell Your Story" portal that maps symptoms → loops → systems → blame

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Intel OneAPI toolkit
- Node.js 18+ (for front-end development)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gaslit-af-warfare.git
cd gaslit-af-warfare

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Build Docker containers
docker-compose build

# Run the simulation
python gaslit-af-runner.py
```

## Development Roadmap

1. Implement core biological modeling components
2. Develop genetic risk scanning modules
3. Build institutional feedback simulation
4. Integrate legal and policy analysis
5. Create front-end visualization and advocacy tools

## License

This project is licensed under the AGPL License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.
