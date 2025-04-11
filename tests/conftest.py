"""
Pytest configuration file for GASLIT-AF WARSTACK tests.

This file contains fixtures and configuration for the test suite.
"""

import os
import sys
import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
try:
    from src.biological_modeling.neuroimmune_simulator import NeuroimmuneDynamics
except ImportError:
    pass

try:
    from src.genetic_risk.genetic_scanner import GeneticRiskScanner
except ImportError:
    pass

try:
    from src.institutional_feedback.institutional_model import InstitutionalFeedbackModel
except ImportError:
    pass

try:
    from src.legal_policy.legal_simulator import LegalPolicySimulator
except ImportError:
    pass

try:
    from src.frontend.app import app as flask_app
except ImportError:
    flask_app = None


# Fixtures for biological modeling tests
@pytest.fixture
def bio_config():
    """Return a test configuration for biological modeling."""
    return {
        'grid_size': 20,  # Smaller grid for faster tests
        'time_steps': 10,
        'dt': 0.1,
        'noise_strength': 0.1,
        'diffusion_constant': 0.5,
        'reaction_rate': 1.0,
        'coupling_strength': 0.8,
        'initial_condition': 'random',
        'boundary_condition': 'periodic',
        'use_hardware_acceleration': False,  # Disable for tests
        'output_dir': 'test_results/biological_modeling',
        'random_seed': 42
    }


@pytest.fixture
def bio_simulator(bio_config):
    """Return a biological simulator instance for testing."""
    try:
        # Create test output directory
        os.makedirs(bio_config['output_dir'], exist_ok=True)
        
        # Create simulator
        simulator = NeuroimmuneDynamics(bio_config)
        simulator.initialize_grid()
        
        yield simulator
        
        # Cleanup
        if os.path.exists(bio_config['output_dir']):
            shutil.rmtree(bio_config['output_dir'])
    
    except (ImportError, NameError):
        pytest.skip("Biological modeling module not available")


# Fixtures for genetic risk scanning tests
@pytest.fixture
def genetic_config():
    """Return a test configuration for genetic risk scanning."""
    return {
        'output_dir': 'test_results/genetic_risk',
        'risk_threshold': 0.6,
        'high_risk_threshold': 0.8,
        'use_hardware_acceleration': False,  # Disable for tests
        'random_seed': 42
    }


@pytest.fixture
def genetic_scanner(genetic_config):
    """Return a genetic scanner instance for testing."""
    try:
        # Create test output directory
        os.makedirs(genetic_config['output_dir'], exist_ok=True)
        
        # Create scanner
        scanner = GeneticRiskScanner(genetic_config)
        
        yield scanner
        
        # Cleanup
        if os.path.exists(genetic_config['output_dir']):
            shutil.rmtree(genetic_config['output_dir'])
    
    except (ImportError, NameError):
        pytest.skip("Genetic risk scanning module not available")


@pytest.fixture
def sample_vcf_file():
    """Create a sample VCF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.vcf', delete=False) as f:
        f.write(b"""##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
1\t11856378\trs1801133\tG\tA\t.\tPASS\t.\tGT\t0/1
1\t11854476\trs1801131\tT\tG\t.\tPASS\t.\tGT\t1/1
6\t32045572\trs121912172\tG\tA\t.\tPASS\t.\tGT\t0/1
22\t19951271\trs4680\tG\tA\t.\tPASS\t.\tGT\t1/1
""")
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


# Fixtures for institutional feedback modeling tests
@pytest.fixture
def institutional_config():
    """Return a test configuration for institutional feedback modeling."""
    return {
        'output_dir': 'test_results/institutional_feedback',
        'simulation_steps': 10,
        'network_size': 5,  # Smaller network for faster tests
        'initial_evidence': 0.1,
        'evidence_growth_rate': 0.02,
        'denial_effectiveness': 0.8,
        'capture_spread_rate': 0.05,
        'random_seed': 42
    }


@pytest.fixture
def institutional_model(institutional_config):
    """Return an institutional feedback model instance for testing."""
    try:
        # Create test output directory
        os.makedirs(institutional_config['output_dir'], exist_ok=True)
        
        # Create model
        model = InstitutionalFeedbackModel(institutional_config)
        
        yield model
        
        # Cleanup
        if os.path.exists(institutional_config['output_dir']):
            shutil.rmtree(institutional_config['output_dir'])
    
    except (ImportError, NameError):
        pytest.skip("Institutional feedback modeling module not available")


# Fixtures for legal policy simulation tests
@pytest.fixture
def legal_config():
    """Return a test configuration for legal policy simulation."""
    return {
        'output_dir': 'test_results/legal_policy',
        'simulation_steps': 10,
        'initial_evidence_level': 0.1,
        'evidence_growth_rate': 0.02,
        'shield_decay_rate': 0.01,
        'random_seed': 42,
        'timeline_start': '2019-01-01',
        'timeline_end': '2025-01-01'
    }


@pytest.fixture
def legal_simulator(legal_config):
    """Return a legal policy simulator instance for testing."""
    try:
        # Create test output directory
        os.makedirs(legal_config['output_dir'], exist_ok=True)
        
        # Create simulator
        simulator = LegalPolicySimulator(legal_config)
        
        yield simulator
        
        # Cleanup
        if os.path.exists(legal_config['output_dir']):
            shutil.rmtree(legal_config['output_dir'])
    
    except (ImportError, NameError):
        pytest.skip("Legal policy simulation module not available")


@pytest.fixture
def sample_legal_document():
    """Create a sample legal document for testing."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"""
National Childhood Vaccine Injury Act of 1986

This act established the National Vaccine Injury Compensation Program (VICP) as a no-fault alternative to the traditional tort system for resolving vaccine injury claims.

Key provisions:
- Liability protection for vaccine manufacturers
- Compensation for vaccine-related injuries or deaths
- Reporting requirements for adverse events

The VICP was established after a series of lawsuits against vaccine manufacturers and healthcare providers threatened to cause vaccine shortages and reduce vaccination rates.
""")
        filepath = f.name
    
    yield filepath
    
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


# Fixtures for frontend tests
@pytest.fixture
def flask_client():
    """Return a Flask test client."""
    if flask_app is None:
        pytest.skip("Frontend module not available")
    
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Helper functions
def save_test_json(data, filename, directory):
    """Save test data as JSON file."""
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f)
    return filepath


def create_test_testimony(test_data_dir):
    """Create a test testimony file."""
    testimony = {
        'name': 'Test User',
        'email': 'test@example.com',
        'age_range': '30-40',
        'story': 'This is a test testimony.',
        'institutional_response': 'No response received.',
        'symptoms': ['fatigue', 'headache', 'brain fog'],
        'other_symptoms': 'Occasional dizziness',
        'onset_date': '2023-01-15',
        'contact_consent': True,
        'submission_date': '2023-02-01T12:34:56'
    }
    
    return save_test_json(testimony, 'testimony_test.json', test_data_dir)
