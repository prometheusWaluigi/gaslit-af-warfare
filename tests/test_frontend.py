"""
Tests for the frontend Flask application.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Import the module, but mock the dependencies that might not be installed
with patch.dict('sys.modules', {
    'flask': MagicMock(),
    'flask.render_template': MagicMock(),
    'flask.request': MagicMock(),
    'flask.jsonify': MagicMock(),
    'flask.redirect': MagicMock(),
    'flask.url_for': MagicMock(),
    'flask.flash': MagicMock(),
    'flask.send_from_directory': MagicMock()
}):
    from src.frontend.app import app, create_app

class TestFrontendApp:
    """Test suite for the frontend Flask application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_simulation_results(self):
        """Create mock simulation results for testing."""
        return {
            'biological': {
                'final_state': {
                    'time': 100,
                    'mean_height': 0.5,
                    'mean_immune_activation': 0.6,
                    'mean_neurotoxicity': 0.7,
                    'final_attractor_state': 'Vulnerable'
                },
                'history': {
                    'time': list(range(100)),
                    'mean_h': [0.1 * i for i in range(100)],
                    'mean_immune': [0.2 * i for i in range(100)],
                    'mean_neurotox': [0.3 * i for i in range(100)],
                    'attractor_state': ['Resilient'] * 50 + ['Vulnerable'] * 50
                }
            },
            'genetic': {
                'risk_results': {
                    'risk_scores': {
                        'TNXB': 0.5,
                        'COMT': 0.3,
                        'MTHFR': 0.7,
                        'RCCX': 0.2
                    },
                    'fragility_gamma': 0.425,
                    'allostatic_lambda': 0.51,
                    'allostatic_omega': 0.34,
                    'risk_category': "Moderate Risk - Fragility"
                },
                'heatmap_data': {
                    'genes': ['TNXB', 'COMT', 'MTHFR', 'RCCX'],
                    'scores': [0.5, 0.3, 0.7, 0.2],
                    'thresholds': {'risk': 0.7, 'collapse': 0.85}
                }
            },
            'institutional': {
                'final_state': {
                    'time': 100,
                    'entropy': 1.5,
                    'narrative_stability': 0.5,
                    'system_state': 'Unstable'
                },
                'history': {
                    'time': list(range(100)),
                    'entropy': [0.01 * i for i in range(100)],
                    'narrative_stability': [1.0 - 0.005 * i for i in range(100)],
                    'system_state': ['Stable'] * 50 + ['Unstable'] * 50
                },
                'institutions': [
                    {"name": "CDC", "type": "government", "influence": 0.9, "denial_bias": 0.7},
                    {"name": "FDA", "type": "government", "influence": 0.85, "denial_bias": 0.65},
                    {"name": "Media Corp", "type": "media", "influence": 0.75, "denial_bias": 0.8}
                ]
            },
            'legal': {
                'final_state': {
                    'time': 100,
                    'evidence_level': 0.7,
                    'shield_strength': 0.3,
                    'shield_breach_probability': 0.8
                },
                'history': {
                    'time': list(range(100)),
                    'evidence_level': [0.007 * i for i in range(100)],
                    'shield_strength': [1.0 - 0.007 * i for i in range(100)],
                    'shield_breach_probability': [0.008 * i for i in range(100)]
                },
                'events': [
                    {
                        "date": "2020-01-15",
                        "type": "regulation",
                        "title": "Emergency Use Authorization",
                        "description": "FDA issues EUA for medical products",
                        "impact": 0.8,
                        "liability_shield": 0.9
                    },
                    {
                        "date": "2022-02-15",
                        "type": "scientific_publication",
                        "title": "Safety Concerns Study",
                        "description": "Study reveals potential safety issues",
                        "impact": 0.6,
                        "liability_shield": -0.3
                    }
                ]
            }
        }
    
    @pytest.fixture
    def mock_testimonies(self):
        """Create mock testimonies for testing."""
        return [
            {
                'name': 'John Doe',
                'date': '2023-05-15',
                'story': 'I experienced severe neurological symptoms after exposure...',
                'symptoms': ['fatigue', 'brain fog', 'tremors']
            },
            {
                'name': 'Jane Smith',
                'date': '2023-06-22',
                'story': 'My autonomic nervous system has been severely affected...',
                'symptoms': ['pots', 'tachycardia', 'fatigue']
            }
        ]
    
    @patch('src.frontend.app.render_template')
    def test_index_route(self, mock_render_template, client):
        """Test that the index route renders the correct template."""
        # Make a request to the index route
        response = client.get('/')
        
        # Check that the response status code is 200 (OK)
        assert response.status_code == 200
        
        # Check that render_template was called with the correct template
        mock_render_template.assert_called_once_with('index.html')
    
    @patch('src.frontend.app.render_template')
    def test_dashboard_route(self, mock_render_template, client, mock_simulation_results):
        """Test that the dashboard route renders the correct template with the correct data."""
        # Set up the mock to return the simulation results
        with patch('src.frontend.app.get_simulation_results', return_value=mock_simulation_results):
            # Make a request to the dashboard route
            response = client.get('/dashboard')
            
            # Check that the response status code is 200 (OK)
            assert response.status_code == 200
            
            # Check that render_template was called with the correct template and data
            mock_render_template.assert_called_once_with(
                'dashboard.html',
                results=mock_simulation_results,
                testimonies_count=0,
                genomes_count=0
            )
    
    @patch('src.frontend.app.render_template')
    def test_biological_dashboard_route(self, mock_render_template, client, mock_simulation_results):
        """Test that the biological dashboard route renders the correct template with the correct data."""
        # Set up the mock to return the simulation results
        with patch('src.frontend.app.get_simulation_results', return_value=mock_simulation_results):
            # Make a request to the biological dashboard route
            response = client.get('/biological')
            
            # Check that the response status code is 200 (OK)
            assert response.status_code == 200
            
            # Check that render_template was called with the correct template and data
            mock_render_template.assert_called_once_with(
                'biological.html',
                results=mock_simulation_results['biological']
            )
    
    @patch('src.frontend.app.render_template')
    def test_genetic_dashboard_route(self, mock_render_template, client, mock_simulation_results):
        """Test that the genetic dashboard route renders the correct template with the correct data."""
        # Set up the mock to return the simulation results
        with patch('src.frontend.app.get_simulation_results', return_value=mock_simulation_results):
            # Make a request to the genetic dashboard route
            response = client.get('/genetic')
            
            # Check that the response status code is 200 (OK)
            assert response.status_code == 200
            
            # Check that render_template was called with the correct template and data
            mock_render_template.assert_called_once_with(
                'genetic.html',
                results=mock_simulation_results['genetic']
            )
