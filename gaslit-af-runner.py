#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GASLIT-AF WARSTACK Runner

Command-line interface for running simulations and analyses across all layers
of the GASLIT-AF WARSTACK project.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules from each layer
try:
    from src.biological_modeling.neuroimmune_simulator import NeuroimmuneDynamics, run_sample_simulation
    BIOLOGICAL_MODELING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Biological modeling module not available: {e}")
    BIOLOGICAL_MODELING_AVAILABLE = False


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GASLIT-AF WARSTACK - Simulation and Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--config', type=str,
                        help='Path to configuration JSON file')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    # Create subparsers for different modules
    subparsers = parser.add_subparsers(dest='module', help='Module to run')
    
    # Biological modeling subparser
    if BIOLOGICAL_MODELING_AVAILABLE:
        bio_parser = subparsers.add_parser('bio', help='Biological modeling simulations')
        bio_parser.add_argument('--time-steps', type=int, default=1000,
                                help='Number of time steps for simulation')
        bio_parser.add_argument('--dt', type=float, default=0.01,
                                help='Time step size')
        bio_parser.add_argument('--spike-toxicity', type=float, default=0.75,
                                help='Spike protein toxicity factor (0-1)')
        bio_parser.add_argument('--cerebellar-vulnerability', type=float, default=0.65,
                                help='Cerebellar vulnerability factor (0-1)')
        bio_parser.add_argument('--autonomic-resilience', type=float, default=0.3,
                                help='Autonomic resilience factor (0-1)')
        bio_parser.add_argument('--spatial-resolution', type=int, default=100,
                                help='Spatial grid resolution')
        bio_parser.add_argument('--plot-only', action='store_true',
                                help='Only generate plots from existing results')
    
    # Genetic risk scanning subparser (placeholder)
    genetic_parser = subparsers.add_parser('genetic', help='Genetic risk scanning')
    genetic_parser.add_argument('--vcf-file', type=str,
                                help='Path to VCF file for genetic analysis')
    genetic_parser.add_argument('--risk-factors', type=str, nargs='+',
                                default=['TNXB', 'COMT', 'MTHFR', 'RCCX'],
                                help='Genetic risk factors to analyze')
    
    # Institutional feedback subparser (placeholder)
    inst_parser = subparsers.add_parser('institutional', help='Institutional feedback modeling')
    inst_parser.add_argument('--model-type', type=str, choices=['denial', 'capture', 'memetic'],
                             default='denial', help='Type of institutional model to simulate')
    
    # Legal & policy subparser (placeholder)
    legal_parser = subparsers.add_parser('legal', help='Legal and policy simulation')
    legal_parser.add_argument('--documents-dir', type=str,
                              help='Directory containing legal documents to analyze')
    
    # Frontend subparser (placeholder)
    frontend_parser = subparsers.add_parser('frontend', help='Start the frontend server')
    frontend_parser.add_argument('--port', type=int, default=3000,
                                 help='Port for the frontend server')
    
    return parser


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    return config


def run_biological_modeling(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Run biological modeling simulations.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    if not BIOLOGICAL_MODELING_AVAILABLE:
        logger.error("Biological modeling module not available")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare parameters
    params = {
        'spike_toxicity': args.spike_toxicity,
        'cerebellar_vulnerability': args.cerebellar_vulnerability,
        'autonomic_resilience': args.autonomic_resilience,
        'time_steps': args.time_steps,
        'spatial_resolution': args.spatial_resolution
    }
    
    # Override with config file if provided
    if 'biological_modeling' in config:
        params.update(config.get('biological_modeling', {}).get('params', {}))
    
    # Create simulator
    simulator = NeuroimmuneDynamics({'params': params})
    
    if not args.plot_only:
        # Run simulation
        logger.info("Running neuroimmune dynamics simulation...")
        results = simulator.run_simulation(time_steps=args.time_steps, dt=args.dt)
        
        # Save results
        results_path = os.path.join(args.output_dir, 'neuroimmune_results.json')
        simulator.save_results(results, results_path)
    else:
        # Load existing results
        results_path = os.path.join(args.output_dir, 'neuroimmune_results.json')
        if not os.path.exists(results_path):
            logger.error(f"Results file not found: {results_path}")
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded existing results from {results_path}")
    
    # Generate plots
    logger.info("Generating phase portrait...")
    portrait_path = os.path.join(args.output_dir, 'phase_portrait.png')
    simulator.plot_phase_portrait(save_path=portrait_path)
    
    logger.info(f"Biological modeling completed. Results saved to {args.output_dir}")


def run_genetic_risk_scanning(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Run genetic risk scanning (placeholder).
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Genetic risk scanning module not yet implemented")
    logger.info(f"Would analyze VCF file: {args.vcf_file}")
    logger.info(f"Would analyze risk factors: {args.risk_factors}")


def run_institutional_feedback(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Run institutional feedback modeling (placeholder).
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Institutional feedback modeling module not yet implemented")
    logger.info(f"Would run model type: {args.model_type}")


def run_legal_policy(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Run legal and policy simulation (placeholder).
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Legal and policy simulation module not yet implemented")
    logger.info(f"Would analyze documents in: {args.documents_dir}")


def run_frontend(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Start the frontend server (placeholder).
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    logger.info("Frontend module not yet implemented")
    logger.info(f"Would start server on port: {args.port}")


def main():
    """Main entry point for the CLI."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the selected module
    if args.module == 'bio':
        run_biological_modeling(args, config)
    elif args.module == 'genetic':
        run_genetic_risk_scanning(args, config)
    elif args.module == 'institutional':
        run_institutional_feedback(args, config)
    elif args.module == 'legal':
        run_legal_policy(args, config)
    elif args.module == 'frontend':
        run_frontend(args, config)
    else:
        # If no module specified, run a sample simulation
        if BIOLOGICAL_MODELING_AVAILABLE:
            logger.info("Running sample biological simulation...")
            run_sample_simulation()
        else:
            parser.print_help()
    
    logger.info("GASLIT-AF WARSTACK execution completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)
