"""
Tests for the institutional feedback modeling module.
"""

import os
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import the module, but mock the dependencies that might not be installed
with patch.dict('sys.modules', {
    'networkx': MagicMock(),
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'matplotlib.cm': MagicMock()
}):
    from src.institutional_feedback.institutional_model import InstitutionalFeedbackModel

class TestInstitutionalFeedbackModel:
    """Test suite for the InstitutionalFeedbackModel class."""
    
    @pytest.fixture
    def sample_institutions(self):
        """Create sample institutions for testing."""
        return [
            {"name": "CDC", "type": "government", "influence": 0.9, "denial_bias": 0.7},
            {"name": "FDA", "type": "government", "influence": 0.85, "denial_bias": 0.65},
            {"name": "NIH", "type": "government", "influence": 0.8, "denial_bias": 0.6},
            {"name": "Media Corp", "type": "media", "influence": 0.75, "denial_bias": 0.8},
            {"name": "Social Platform", "type": "media", "influence": 0.7, "denial_bias": 0.75},
            {"name": "Pharma A", "type": "industry", "influence": 0.65, "denial_bias": 0.9},
            {"name": "Pharma B", "type": "industry", "influence": 0.6, "denial_bias": 0.85}
        ]
    
    def test_initialization(self, sample_institutions):
        """Test that the InstitutionalFeedbackModel class initializes correctly."""
        # Initialize with default config
        model = InstitutionalFeedbackModel()
        
        # Check that default parameters were set correctly
        assert model.params['time_steps'] == 100
        assert model.params['entropy_threshold'] == 1.5
        assert model.params['narrative_collapse_threshold'] == 2.0
        assert len(model.institutions) > 0
        
        # Initialize with custom config
        custom_config = {
            'params': {
                'time_steps': 50,
                'entropy_threshold': 1.2,
                'narrative_collapse_threshold': 1.8,
                'institutions': sample_institutions
            }
        }
        model = InstitutionalFeedbackModel(custom_config)
        
        # Check that custom parameters were set correctly
        assert model.params['time_steps'] == custom_config['params']['time_steps']
        assert model.params['entropy_threshold'] == custom_config['params']['entropy_threshold']
        assert model.params['narrative_collapse_threshold'] == custom_config['params']['narrative_collapse_threshold']
        assert model.institutions == sample_institutions
    
    def test_reset_state(self):
        """Test that the reset_state method properly resets the simulation state."""
        model = InstitutionalFeedbackModel()
        
        # Run a few steps to change the state
        model.time = 10
        model.iteration = 5
        model.entropy = 1.0
        model.narrative_stability = 0.5
        
        # Reset state
        model.reset_state()
        
        # Check that state was reset
        assert model.time == 0
        assert model.iteration == 0
        assert model.entropy == 0.0
        assert model.narrative_stability == 1.0
        assert len(model.history['time']) == 0
        assert len(model.history['entropy']) == 0
        assert len(model.history['narrative_stability']) == 0
        assert len(model.history['system_state']) == 0
    
    @patch('src.institutional_feedback.institutional_model.nx.DiGraph')
    def test_build_institutional_network(self, mock_digraph, sample_institutions):
        """Test that the build_institutional_network method correctly builds a network."""
        # Set up the mock
        mock_graph = MagicMock()
        mock_digraph.return_value = mock_graph
        
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Build network
        network = model.build_institutional_network()
        
        # Check that the graph was created
        mock_digraph.assert_called_once()
        
        # Check that nodes were added for each institution
        assert mock_graph.add_node.call_count == len(sample_institutions)
        
        # Check that edges were added between institutions
        assert mock_graph.add_edge.call_count > 0
    
    def test_calculate_denial_loop_strength(self, sample_institutions):
        """Test that the calculate_denial_loop_strength method correctly calculates loop strength."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Calculate denial loop strength
        loop_strength = model.calculate_denial_loop_strength()
        
        # Check that the result is a float between 0 and 1
        assert isinstance(loop_strength, float)
        assert 0 <= loop_strength <= 1
    
    def test_calculate_regulatory_capture(self, sample_institutions):
        """Test that the calculate_regulatory_capture method correctly calculates capture."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Calculate regulatory capture
        capture = model.calculate_regulatory_capture()
        
        # Check that the result is a float between 0 and 1
        assert isinstance(capture, float)
        assert 0 <= capture <= 1
    
    def test_calculate_memetic_immunosuppression(self, sample_institutions):
        """Test that the calculate_memetic_immunosuppression method correctly calculates immunosuppression."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Calculate memetic immunosuppression
        immunosuppression = model.calculate_memetic_immunosuppression()
        
        # Check that the result is a float between 0 and 1
        assert isinstance(immunosuppression, float)
        assert 0 <= immunosuppression <= 1
    
    def test_update_entropy(self, sample_institutions):
        """Test that the update_entropy method correctly updates entropy."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Store initial entropy
        initial_entropy = model.entropy
        
        # Update entropy
        model.update_entropy()
        
        # Check that entropy was updated
        assert model.entropy != initial_entropy
    
    def test_update_narrative_stability(self, sample_institutions):
        """Test that the update_narrative_stability method correctly updates stability."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Store initial stability
        initial_stability = model.narrative_stability
        
        # Update stability
        model.update_narrative_stability()
        
        # Check that stability was updated
        assert model.narrative_stability != initial_stability
    
    def test_determine_system_state(self):
        """Test that the determine_system_state method correctly determines the system state."""
        model = InstitutionalFeedbackModel()
        
        # Test low entropy case
        model.entropy = 0.5
        model.narrative_stability = 0.9
        state = model.determine_system_state()
        assert state == "Stable"
        
        # Test medium entropy case
        model.entropy = 1.6
        model.narrative_stability = 0.5
        state = model.determine_system_state()
        assert state == "Unstable"
        
        # Test high entropy case
        model.entropy = 2.5
        model.narrative_stability = 0.1
        state = model.determine_system_state()
        assert state == "Collapsed"
    
    def test_step(self, sample_institutions):
        """Test that the step method correctly advances the simulation by one step."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Store initial values
        initial_time = model.time
        initial_iteration = model.iteration
        initial_entropy = model.entropy
        initial_stability = model.narrative_stability
        
        # Perform a step
        model.step()
        
        # Check that values were updated
        assert model.time > initial_time
        assert model.iteration > initial_iteration
        assert model.entropy != initial_entropy
        assert model.narrative_stability != initial_stability
        
        # Check that history was updated
        assert len(model.history['time']) == 1
        assert len(model.history['entropy']) == 1
        assert len(model.history['narrative_stability']) == 1
        assert len(model.history['system_state']) == 1
    
    def test_run_simulation(self, sample_institutions):
        """Test that the run_simulation method completes without errors."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {
                'institutions': sample_institutions,
                'time_steps': 10  # Use a small number for testing
            }
        })
        
        # Run simulation
        results = model.run_simulation()
        
        # Check that results contain expected keys
        assert 'params' in results
        assert 'final_state' in results
        assert 'history' in results
        assert 'institutions' in results
        
        # Check that history has the correct length
        assert len(results['history']['time']) == 10
        assert len(results['history']['entropy']) == 10
        assert len(results['history']['narrative_stability']) == 10
        assert len(results['history']['system_state']) == 10
        
        # Check that final state contains expected keys
        assert 'time' in results['final_state']
        assert 'entropy' in results['final_state']
        assert 'narrative_stability' in results['final_state']
        assert 'system_state' in results['final_state']
    
    def test_save_results(self, sample_institutions, output_dir):
        """Test that results can be saved to a file."""
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {
                'institutions': sample_institutions,
                'time_steps': 5  # Use a small number for testing
            }
        })
        
        # Run a short simulation
        results = model.run_simulation()
        
        # Save results
        output_file = os.path.join(output_dir, 'institutional_results.json')
        model.save_results(results, output_file)
        
        # Check that the file was created
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
        
        # Check that the file contains valid JSON
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        # Check that the loaded results match the original results
        assert loaded_results['params'] == results['params']
        assert loaded_results['final_state'] == results['final_state']
        assert loaded_results['institutions'] == results['institutions']
    
    @patch('src.institutional_feedback.institutional_model.InstitutionalFeedbackModel.build_institutional_network')
    @patch('src.institutional_feedback.institutional_model.InstitutionalFeedbackModel.visualize_network')
    def test_generate_network_visualization(self, mock_visualize, mock_build, sample_institutions, output_dir):
        """Test that the generate_network_visualization method correctly generates a visualization."""
        # Set up the mocks
        mock_network = MagicMock()
        mock_build.return_value = mock_network
        
        # Initialize model with sample institutions
        model = InstitutionalFeedbackModel({
            'params': {'institutions': sample_institutions}
        })
        
        # Generate visualization
        output_file = os.path.join(output_dir, 'network.png')
        model.generate_network_visualization(output_file)
        
        # Check that the mocks were called
        mock_build.assert_called_once()
        mock_visualize.assert_called_once_with(mock_network, output_file)
    
    @pytest.mark.parametrize("denial_bias,influence", [
        (0.9, 0.9),  # High denial bias, high influence -> likely collapse
        (0.1, 0.1),  # Low denial bias, low influence -> likely stable
    ])
    def test_system_states(self, denial_bias, influence):
        """Test that different parameter combinations lead to different system states."""
        # Create institutions with specific parameters
        institutions = [
            {"name": "Inst1", "type": "government", "influence": influence, "denial_bias": denial_bias},
            {"name": "Inst2", "type": "media", "influence": influence, "denial_bias": denial_bias},
            {"name": "Inst3", "type": "industry", "influence": influence, "denial_bias": denial_bias}
        ]
        
        # Initialize model with these institutions
        model = InstitutionalFeedbackModel({
            'params': {
                'institutions': institutions,
                'time_steps': 50
            }
        })
        
        # Run simulation
        results = model.run_simulation()
        
        # For high denial bias and high influence, expect higher entropy
        if denial_bias > 0.8 and influence > 0.8:
            assert results['final_state']['entropy'] > 1.0
        
        # For low denial bias and low influence, expect lower entropy
        if denial_bias < 0.2 and influence < 0.2:
            assert results['final_state']['entropy'] < 1.0
