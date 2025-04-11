# GASLIT-AF WARSTACK Documentation

This directory contains documentation for the GASLIT-AF WARSTACK project.

## Overview

GASLIT-AF WARSTACK is a modular simulation-and-exposure engine designed to prove institutional culpability via biological recursion and systemic denial modeling. The project consists of five main modules:

1. **Biological Modeling Layer**: Simulates neuroimmune dynamics and system transitions
2. **Genetic Risk Scanning Layer**: Analyzes genetic data for fragility architecture
3. **Institutional Feedback Modeling**: Models denial-injury-denial loops and regulatory capture
4. **Legal & Policy Simulation Layer**: Simulates legal dynamics and liability thresholds
5. **Front-End Advocacy Layer**: Provides user interfaces for visualization and testimony collection

## Documentation Structure

- `architecture/`: System architecture and design documents
- `api/`: API documentation (auto-generated)
- `modules/`: Detailed documentation for each module
- `user_guides/`: End-user documentation
- `development/`: Developer guides and contribution guidelines

## Module Documentation

### Biological Modeling

The Biological Modeling module simulates the core GASLIT-AF attractor states:
- Spike protein neurotoxicity
- Cerebellar trauma
- Behavioral and autonomic collapse

Key components:
- KPZ / fKPZχ simulations of neuroimmune dynamics
- ODE/PDE attractor maps showing irreversible system transitions
- Phase portraits of feedback loop entrapment

### Genetic Risk Scanning

The Genetic Risk Scanning module parses genomic data files for:
- Fragility architecture (γ)
- Allostatic collapse risk (Λ, Ω)

Key components:
- Variant fragility heatmaps
- GASLIT-AF phenotype indexing
- JSON risk profile generation

### Institutional Feedback Modeling

The Institutional Feedback Modeling module builds:
- Dynamic denial-injury-denial loops
- Regulatory capture graphs
- Memetic immunosuppression nets

Key components:
- Denial recursion maps
- System legitimacy entropy index
- Narrative collapse simulation

### Legal & Policy Simulation

The Legal & Policy Simulation module runs simulation logic against:
- NCVIA frameworks
- EUA claims
- Suppressed data trails
- Ethical obligations

Key components:
- Class action viability predictions
- Liability shield breach thresholds
- Evidence timeline analysis

### Front-End Advocacy

The Front-End Advocacy module provides:
- Dynamic dashboards of systemic harm
- User-uploadable genome analysis
- "Tell Your Story" portal

Key components:
- Visualization interfaces
- Data collection forms
- Result interpretation guides

## Building the Documentation

To build the documentation locally:

1. Install Sphinx and required extensions:
   ```bash
   pip install -r docs/requirements.txt
   ```

2. Build the HTML documentation:
   ```bash
   cd docs
   make html
   ```

3. View the documentation:
   ```bash
   open _build/html/index.html
   ```

## Contributing to Documentation

When contributing to the documentation:

1. Follow the established structure
2. Use clear, concise language
3. Include examples where appropriate
4. Add diagrams for complex concepts
5. Ensure all code examples are tested and working
6. Update the table of contents when adding new sections

## API Documentation

API documentation is automatically generated from docstrings in the code. To update the API documentation:

```bash
cd docs
sphinx-apidoc -o api ../src
make html
