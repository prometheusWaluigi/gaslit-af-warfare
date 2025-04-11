"""
Biological Modeling Module for GASLIT-AF WARSTACK

This module simulates the core GASLIT-AF attractor states:
- Spike protein neurotoxicity
- Cerebellar trauma
- Behavioral and autonomic collapse

Using KPZ / fKPZχ simulations of neuroimmune dynamics, ODE/PDE attractor maps,
and phase portraits of feedback loop entrapment.
"""

from .neuroimmune_simulator import NeuroimmuneDynamics, run_sample_simulation

__all__ = ['NeuroimmuneDynamics', 'run_sample_simulation']
