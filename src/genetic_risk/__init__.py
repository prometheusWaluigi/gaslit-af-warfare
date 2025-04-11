"""
Genetic Risk Scanning Module for GASLIT-AF WARSTACK

This module parses FASTQ/VCF files for fragility architecture (γ) and allostatic
collapse risk (Λ, Ω). It generates heatmaps of variant fragility and exportable
JSON risk profiles.
"""

from .genetic_scanner import GeneticRiskScanner, run_sample_analysis

__all__ = ['GeneticRiskScanner', 'run_sample_analysis']
