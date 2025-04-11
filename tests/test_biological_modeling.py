"""
Tests for the biological modeling module.
"""

import os
import numpy as np
import pytest
from src.biological_modeling.neuroimmune_simulator import NeuroimmuneDynamics

class TestNeuroimmuneDynamics:
    """Test suite for the NeuroimmuneDynamics class."""
    
    def test_initialization(self, sample_config):
        """Test that the NeuroimmuneDynamics class initializes correctly."""
        # Initialize with sample config
        simulator = NeuroimmuneDynamics(sample_config)
        
        # Check that parameters were set correctly
        assert simulator.params['spike_toxicity'] == sample_config['params']['spike_toxicity']
        assert simulator.params['cerebellar_vulnerability'] == sample_config['params']['cerebellar_vulnerability']
        assert simulator.params['autonomic_resilience'] == sample_config['params']['autonomic_resilience']
        
        # Check that state variables were initialized
        assert simulator.time == 0
        assert simulator.iteration == 0
        assert simulator.h.shape == (sample_config['params']['spatial_resolution'], 
                                    sample_config['params']['spatial_resolution'])
        assert simulator.immune_activation.shape == (sample_config['params']['spatial_resolution'], 
                                                    sample_config['params']['spatial_resolution'])
        assert simulator.neurotoxicity.shape == (sample_config['params']['spatial_resolution'], 
                                                sample_config['params']['spatial_resolution'])
    
    def test_reset_state(self, sample_config):
        """Test that the reset_state method properly resets the simulation state."""
        simulator = NeuroimmuneDynamics(sample_config)
        
        # Run a few steps to change the state
        simulator.time = 10
        simulator.iteration = 5
        
        # Reset state
        simulator.reset_state()
        
        # Check that state was reset
        assert simulator.time == 0
        assert simulator.iteration == 0
        assert len(simulator.history['time']) == 0
        assert len(simulator.history['mean_h']) == 0
        assert len(simulator.history['mean_immune']) == 0
        assert len(simulator.history['mean_neurotox']) == 0
        assert len(simulator.history['attractor_state']) == 0
    
    def test_kpz_step(self, sample_config):
        """Test that the KPZ step function updates the height field."""
        simulator = NeuroimmuneDynamics(sample_config)
        
        # Store initial height field
        initial_h = simulator.h.copy()
        
        # Perform a KPZ step
        dt = 0.01
        updated_h = simulator.kpz_step(dt)
        
        # Check that the height field was updated
        assert not np.array_equal(updated_h, initial_h)
        assert updated_h.shape == initial_h.shape
    
    def test_coupled_dynamics_step(self, sample_config):
        """Test that the coupled dynamics step updates all fields."""
        simulator = NeuroimmuneDynamics(sample_config)
        
        # Store initial values
        initial_h = simulator.h.copy()
        initial_immune = simulator.immune_activation.copy()
        initial_neurotox = simulator.neurotoxicity.copy()
        initial_time = simulator.time
        initial_iteration = simulator.iteration
        
        # Perform a coupled dynamics step
        dt = 0.01
        simulator.coupled_dynamics_step(dt)
        
        # Check that all fields were updated
        assert not np.array_equal(simulator.h, initial_h)
        assert not np.array_equal(simulator.immune_activation, initial_immune)
        assert not np.array_equal(simulator.neurotoxicity, initial_neurotox)
        assert simulator.time > initial_time
        assert simulator.iteration > initial_iteration
        
        # Check that history was updated
        assert len(simulator.history['time']) == 1
        assert len(simulator.history['mean_h']) == 1
        assert len(simulator.history['mean_immune']) == 1
        assert len(simulator.history['mean_neurotox']) == 1
        assert len(simulator.history['attractor_state']) == 1
    
    def test_run_simulation(self, sample_config):
        """Test that the run_simulation method completes without errors."""
        simulator = NeuroimmuneDynamics(sample_config)
        
        # Run a short simulation
        time_steps = 10
        results = simulator.run_simulation(time_steps=time_steps)
        
        # Check that results contain expected keys
        assert 'params' in results
        assert 'final_state' in results
        assert 'history' in results
        
        # Check that history has the correct length
        assert len(results['history']['time']) == time_steps
        assert len(results['history']['mean_h']) == time_steps
        assert len(results['history']['mean_immune']) == time_steps
        assert len(results['history']['mean_neurotox']) == time_steps
        assert len(results['history']['attractor_state']) == time_steps
        
        # Check that final state contains expected keys
        assert 'time' in results['final_state']
        assert 'mean_height' in results['final_state']
        assert 'mean_immune_activation' in results['final_state']
        assert 'mean_neurotoxicity' in results['final_state']
        assert 'final_attractor_state' in results['final_state']
    
    def test_save_results(self, sample_config, output_dir):
        """Test that results can be saved to a file."""
        simulator = NeuroimmuneDynamics(sample_config)
        
        # Run a short simulation
        results = simulator.run_simulation(time_steps=5)
        
        # Save results
        output_file = os.path.join(output_dir, 'test_results.json')
        simulator.save_results(results, output_file)
        
        # Check that the file was created
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    @pytest.mark.parametrize("spike_toxicity,cerebellar_vulnerability,autonomic_resilience", [
        (0.9, 0.8, 0.1),  # High toxicity, high vulnerability, low resilience -> likely collapse
        (0.1, 0.2, 0.9),  # Low toxicity, low vulnerability, high resilience -> likely resilient
    ])
    def test_attractor_states(self, spike_toxicity, cerebellar_vulnerability, autonomic_resilience):
        """Test that different parameter combinations lead to different attractor states."""
        # Create config with specific parameters
        config = {
            'params': {
                'spike_toxicity': spike_toxicity,
                'cerebellar_vulnerability': cerebellar_vulnerability,
                'autonomic_resilience': autonomic_resilience,
                'time_steps': 50,
                'spatial_resolution': 20,  # Lower resolution for faster tests
            }
        }
        
        simulator = NeuroimmuneDynamics(config)
        
        # Run simulation
        results = simulator.run_simulation()
        
        # For high toxicity, high vulnerability, low resilience, expect higher neurotoxicity
        if spike_toxicity > 0.8 and cerebellar_vulnerability > 0.7 and autonomic_resilience < 0.2:
            assert results['final_state']['mean_neurotoxicity'] > 0.5
        
        # For low toxicity, low vulnerability, high resilience, expect lower neurotoxicity
        if spike_toxicity < 0.2 and cerebellar_vulnerability < 0.3 and autonomic_resilience > 0.8:
            assert results['final_state']['mean_neurotoxicity'] < 0.3
