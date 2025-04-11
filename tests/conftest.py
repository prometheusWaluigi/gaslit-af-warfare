"""
Global pytest configuration and fixtures for GASLIT-AF WARSTACK tests.
"""

import os
import sys
import pytest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Global fixtures that can be used across all test modules

@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        'params': {
            'spike_toxicity': 0.5,
            'cerebellar_vulnerability': 0.4,
            'autonomic_resilience': 0.6,
            'time_steps': 100,
            'spatial_resolution': 50,
            'diffusion_coefficient': 0.1,
            'noise_amplitude': 0.05,
            'lambda_nonlinearity': 1.0,
            'nu_viscosity': 0.1,
        }
    }

@pytest.fixture
def output_dir(tmpdir):
    """Provide a temporary directory for test outputs."""
    output = tmpdir.mkdir('test_output')
    return str(output)
