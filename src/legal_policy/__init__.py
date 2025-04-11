"""
Legal Policy Simulation Module for GASLIT-AF WARSTACK

This module simulates legal and policy dynamics related to GASLIT-AF syndrome,
including liability shield analysis, evidence timelines, and class action viability.
"""

from .legal_simulator import LegalPolicySimulator, run_sample_simulation

__all__ = ['LegalPolicySimulator', 'run_sample_simulation']
