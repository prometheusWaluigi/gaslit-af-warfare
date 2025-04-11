#!/usr/bin/env python3
"""
GASLIT-AF WARSTACK Runner

This script provides a command-line interface for running the GASLIT-AF WARSTACK
simulations and analyses.
"""

import os
import sys
import argparse
import logging
import json
import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import modules
try:
    from src.biological_modeling import run_sample_simulation as run_biological
except ImportError:
    def run_biological(*args, **kwargs):
        print("Biological modeling module not available")
        return None

try:
    from src.genetic_risk import run_sample_analysis as run_genetic
except ImportError:
    def run_genetic(*args, **kwargs):
        print("Genetic risk scanning module not available")
        return None

try:
    from src.institutional_feedback import run_sample_simulation as run_institutional
except ImportError:
    def run_institutional(*args, **kwargs):
        print("Institutional feedback modeling module not available")
        return None

try:
    from src.legal_policy import run_sample_simulation as run_legal
except ImportError:
    def run_legal(*args, **kwargs):
        print("Legal policy simulation module not available")
        return None


def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/gaslit-af-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GASLIT-AF WARSTACK Runner"
    )
    
    parser.add_argument(
        '--module', '-m',
        choices=['biological', 'genetic', 'institutional', 'legal', 'all'],
        default='all',
        help="Module to run (default: all)"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help="Directory for output files (default: results)"
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help="Generate visualizations"
    )
    
    parser.add_argument(
        '--save-results', '-s',
        action='store_true',
        help="Save results to file"
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help="Use GPU acceleration if available"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose output"
    )
    
    parser.add_argument(
        '--input-file', '-i',
        type=str,
        help="Input file for genetic analysis (VCF/FASTQ)"
    )
    
    return parser.parse_args()


def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    default_config = {
        'biological': {
            'grid_size': 100,
            'time_steps': 1000,
            'dt': 0.01,
            'noise_strength': 0.1,
            'diffusion_constant': 1.0,
            'reaction_rate': 1.0,
            'coupling_strength': 0.5,
            'initial_condition': 'random',
            'boundary_condition': 'periodic',
            'use_hardware_acceleration': False,
            'output_dir': 'results/biological_modeling',
            'random_seed': 42
        },
        'genetic': {
            'output_dir': 'results/genetic_risk',
            'risk_threshold': 0.7,
            'high_risk_threshold': 0.9,
            'use_hardware_acceleration': False,
            'random_seed': 42
        },
        'institutional': {
            'output_dir': 'results/institutional_feedback',
            'simulation_steps': 100,
            'network_size': 20,
            'initial_evidence': 0.1,
            'evidence_growth_rate': 0.01,
            'denial_effectiveness': 0.8,
            'capture_spread_rate': 0.05,
            'random_seed': 42
        },
        'legal': {
            'output_dir': 'results/legal_policy',
            'simulation_steps': 100,
            'initial_evidence_level': 0.1,
            'evidence_growth_rate': 0.01,
            'shield_decay_rate': 0.005,
            'random_seed': 42,
            'timeline_start': '2019-01-01',
            'timeline_end': '2025-01-01'
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Merge user config with defaults
            for module, module_config in user_config.items():
                if module in default_config:
                    default_config[module].update(module_config)
                else:
                    default_config[module] = module_config
                    
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
    
    return default_config


def run_modules(args, config):
    """Run the specified modules."""
    results = {}
    
    # Create output directories
    for module in config:
        os.makedirs(config[module]['output_dir'], exist_ok=True)
    
    # Update GPU settings
    if args.use_gpu:
        for module in config:
            config[module]['use_hardware_acceleration'] = True
    
    # Run biological modeling
    if args.module in ['biological', 'all']:
        logging.info("Running biological modeling simulation...")
        try:
            results['biological'] = run_biological(
                config=config['biological'],
                visualize=args.visualize,
                save_results=args.save_results
            )
            logging.info("Biological modeling simulation completed")
        except Exception as e:
            logging.error(f"Error in biological modeling: {e}")
    
    # Run genetic risk scanning
    if args.module in ['genetic', 'all']:
        logging.info("Running genetic risk scanning...")
        try:
            results['genetic'] = run_genetic(
                config=config['genetic'],
                input_file=args.input_file,
                visualize=args.visualize,
                save_results=args.save_results
            )
            logging.info("Genetic risk scanning completed")
        except Exception as e:
            logging.error(f"Error in genetic risk scanning: {e}")
    
    # Run institutional feedback modeling
    if args.module in ['institutional', 'all']:
        logging.info("Running institutional feedback modeling...")
        try:
            results['institutional'] = run_institutional(
                config=config['institutional'],
                visualize=args.visualize,
                save_results=args.save_results
            )
            logging.info("Institutional feedback modeling completed")
        except Exception as e:
            logging.error(f"Error in institutional feedback modeling: {e}")
    
    # Run legal policy simulation
    if args.module in ['legal', 'all']:
        logging.info("Running legal policy simulation...")
        try:
            results['legal'] = run_legal(
                config=config['legal'],
                visualize=args.visualize,
                save_results=args.save_results
            )
            logging.info("Legal policy simulation completed")
        except Exception as e:
            logging.error(f"Error in legal policy simulation: {e}")
    
    return results


def save_results(results, output_dir):
    """Save results to file."""
    if not results:
        logging.warning("No results to save")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_file = os.path.join(output_dir, f"gaslit-af-results-{timestamp}.json")
    
    # Convert non-serializable objects to strings
    serializable_results = {}
    for module, module_results in results.items():
        if module_results is None:
            continue
        
        if isinstance(module_results, dict):
            serializable_results[module] = {}
            for key, value in module_results.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    serializable_results[module][key] = value
                except (TypeError, OverflowError):
                    # If not, convert to string
                    serializable_results[module][key] = str(value)
        else:
            serializable_results[module] = str(module_results)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    
    # Run modules
    results = run_modules(args, config)
    
    # Save results
    if args.save_results:
        save_results(results, args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
