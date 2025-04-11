"""
Tests for the genetic risk scanning module.
"""

import os
import json
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import the module, but mock the dependencies that might not be installed
with patch.dict('sys.modules', {
    'allel': MagicMock(),
    'Bio': MagicMock(),
    'Bio.SeqIO': MagicMock()
}):
    from src.genetic_risk.genetic_scanner import GeneticRiskScanner

class TestGeneticRiskScanner:
    """Test suite for the GeneticRiskScanner class."""
    
    @pytest.fixture
    def mock_vcf_data(self):
        """Create mock VCF data for testing."""
        # Create a mock for the variants and samples
        variants = {
            'CHROM': np.array(['1', '6', '22', '1']),
            'POS': np.array([100, 300, 400, 500]),
            'ID': np.array(['rs123', 'TNXB_var1', 'COMT_var1', 'MTHFR_var1']),
            'REF': np.array(['A', 'G', 'T', 'A']),
            'ALT': np.array(['G', 'A', 'C', 'G'])
        }
        samples = np.array(['SAMPLE1'])
        
        return {
            'callset': {'variants': variants, 'samples': samples},
            'variants': variants,
            'samples': samples
        }
    
    @pytest.fixture
    def mock_fastq_data(self):
        """Create mock FASTQ data for testing."""
        # Create a mock for the FASTQ records
        records = [MagicMock() for _ in range(10)]
        
        return {
            'records': records,
            'count': len(records)
        }
    
    def test_initialization(self):
        """Test that the GeneticRiskScanner class initializes correctly."""
        # Initialize with default config
        scanner = GeneticRiskScanner()
        
        # Check that default parameters were set correctly
        assert 'TNXB' in scanner.params['risk_genes']
        assert 'COMT' in scanner.params['risk_genes']
        assert 'MTHFR' in scanner.params['risk_genes']
        assert 'RCCX' in scanner.params['risk_genes']
        assert scanner.params['risk_threshold'] == 0.7
        assert scanner.params['collapse_threshold'] == 0.85
        
        # Initialize with custom config
        custom_config = {
            'params': {
                'risk_genes': ['GENE1', 'GENE2'],
                'risk_threshold': 0.6,
                'collapse_threshold': 0.8
            }
        }
        scanner = GeneticRiskScanner(custom_config)
        
        # Check that custom parameters were set correctly
        assert scanner.params['risk_genes'] == custom_config['params']['risk_genes']
        assert scanner.params['risk_threshold'] == custom_config['params']['risk_threshold']
        assert scanner.params['collapse_threshold'] == custom_config['params']['collapse_threshold']
    
    @patch('src.genetic_risk.genetic_scanner.HAS_ALLEL', True)
    @patch('src.genetic_risk.genetic_scanner.allel.read_vcf')
    def test_load_vcf(self, mock_read_vcf, mock_vcf_data):
        """Test that the load_vcf method correctly loads VCF data."""
        # Set up the mock
        mock_read_vcf.return_value = mock_vcf_data['callset']
        
        # Create a temporary VCF file
        with tempfile.NamedTemporaryFile(suffix='.vcf') as temp_vcf:
            # Initialize scanner
            scanner = GeneticRiskScanner()
            
            # Load VCF file
            vcf_data = scanner.load_vcf(temp_vcf.name)
            
            # Check that the mock was called with the correct file path
            mock_read_vcf.assert_called_once_with(temp_vcf.name)
            
            # Check that the returned data has the expected structure
            assert 'callset' in vcf_data
            assert 'variants' in vcf_data
            assert 'samples' in vcf_data
    
    @patch('src.genetic_risk.genetic_scanner.HAS_BIOPYTHON', True)
    @patch('src.genetic_risk.genetic_scanner.SeqIO.parse')
    def test_load_fastq(self, mock_parse, mock_fastq_data):
        """Test that the load_fastq method correctly loads FASTQ data."""
        # Set up the mock
        mock_parse.return_value = mock_fastq_data['records']
        
        # Create a temporary FASTQ file
        with tempfile.NamedTemporaryFile(suffix='.fastq') as temp_fastq:
            # Initialize scanner
            scanner = GeneticRiskScanner()
            
            # Load FASTQ file
            fastq_data = scanner.load_fastq(temp_fastq.name)
            
            # Check that the mock was called with the correct file path and format
            mock_parse.assert_called_once_with(temp_fastq.name, "fastq")
            
            # Check that the returned data has the expected structure
            assert 'records' in fastq_data
            assert 'count' in fastq_data
            assert fastq_data['count'] == len(mock_fastq_data['records'])
    
    @patch('src.genetic_risk.genetic_scanner.HAS_ALLEL', True)
    def test_analyze_risk_genes(self, mock_vcf_data):
        """Test that the analyze_risk_genes method correctly analyzes risk genes."""
        # Initialize scanner
        scanner = GeneticRiskScanner()
        
        # Analyze risk genes
        risk_results = scanner.analyze_risk_genes(mock_vcf_data)
        
        # Check that the results have the expected structure
        assert 'risk_scores' in risk_results
        assert 'fragility_gamma' in risk_results
        assert 'allostatic_lambda' in risk_results
        assert 'allostatic_omega' in risk_results
        assert 'risk_category' in risk_results
        
        # Check that risk scores were calculated for all risk genes
        for gene in scanner.params['risk_genes']:
            assert gene in risk_results['risk_scores']
        
        # Check that the risk metrics are in the expected range
        assert 0 <= risk_results['fragility_gamma'] <= 1
        assert 0 <= risk_results['allostatic_lambda'] <= 1.2  # Can be slightly higher than 1 due to calculation
        assert 0 <= risk_results['allostatic_omega'] <= 1
        
        # Check that the risk category is one of the expected values
        assert risk_results['risk_category'] in [
            "Low Risk", 
            "Moderate Risk - Fragility", 
            "High Risk - Allostatic Collapse"
        ]
    
    def test_generate_heatmap_data(self):
        """Test that the generate_heatmap_data method correctly generates heatmap data."""
        # Create sample risk results
        risk_results = {
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
        }
        
        # Initialize scanner
        scanner = GeneticRiskScanner()
        
        # Generate heatmap data
        heatmap_data = scanner.generate_heatmap_data(risk_results)
        
        # Check that the heatmap data has the expected structure
        assert 'genes' in heatmap_data
        assert 'scores' in heatmap_data
        assert 'thresholds' in heatmap_data
        
        # Check that the genes and scores match the risk results
        assert set(heatmap_data['genes']) == set(risk_results['risk_scores'].keys())
        assert len(heatmap_data['scores']) == len(risk_results['risk_scores'])
        
        # Check that the thresholds match the scanner parameters
        assert heatmap_data['thresholds']['risk'] == scanner.params['risk_threshold']
        assert heatmap_data['thresholds']['collapse'] == scanner.params['collapse_threshold']
    
    def test_export_risk_profile(self, output_dir):
        """Test that the export_risk_profile method correctly exports a risk profile."""
        # Create sample risk results
        risk_results = {
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
        }
        
        # Initialize scanner
        scanner = GeneticRiskScanner()
        
        # Export risk profile
        output_file = os.path.join(output_dir, 'risk_profile.json')
        scanner.export_risk_profile(risk_results, output_file)
        
        # Check that the file was created
        assert os.path.exists(output_file)
        
        # Check that the file contains the expected data
        with open(output_file, 'r') as f:
            exported_data = json.load(f)
        
        assert 'risk_scores' in exported_data
        assert 'metrics' in exported_data
        assert 'risk_category' in exported_data
        assert 'parameters' in exported_data
        
        assert exported_data['risk_scores'] == risk_results['risk_scores']
        assert exported_data['metrics']['fragility_gamma'] == risk_results['fragility_gamma']
        assert exported_data['metrics']['allostatic_lambda'] == risk_results['allostatic_lambda']
        assert exported_data['metrics']['allostatic_omega'] == risk_results['allostatic_omega']
        assert exported_data['risk_category'] == risk_results['risk_category']
        assert exported_data['parameters'] == scanner.params
    
    @patch('src.genetic_risk.genetic_scanner.GeneticRiskScanner.load_vcf')
    @patch('src.genetic_risk.genetic_scanner.GeneticRiskScanner.analyze_risk_genes')
    @patch('src.genetic_risk.genetic_scanner.GeneticRiskScanner.generate_heatmap_data')
    @patch('src.genetic_risk.genetic_scanner.GeneticRiskScanner.export_risk_profile')
    def test_run_analysis(self, mock_export, mock_generate, mock_analyze, mock_load, mock_vcf_data, output_dir):
        """Test that the run_analysis method correctly runs a complete analysis."""
        # Set up the mocks
        mock_load.return_value = mock_vcf_data
        mock_analyze.return_value = {
            'risk_scores': {'TNXB': 0.5, 'COMT': 0.3, 'MTHFR': 0.7, 'RCCX': 0.2},
            'fragility_gamma': 0.425,
            'allostatic_lambda': 0.51,
            'allostatic_omega': 0.34,
            'risk_category': "Moderate Risk - Fragility"
        }
        mock_generate.return_value = {
            'genes': ['TNXB', 'COMT', 'MTHFR', 'RCCX'],
            'scores': [0.5, 0.3, 0.7, 0.2],
            'thresholds': {'risk': 0.7, 'collapse': 0.85}
        }
        
        # Create a temporary VCF file
        with tempfile.NamedTemporaryFile(suffix='.vcf') as temp_vcf:
            # Initialize scanner
            scanner = GeneticRiskScanner()
            
            # Run analysis
            results = scanner.run_analysis(temp_vcf.name, output_dir)
            
            # Check that all the mocks were called
            mock_load.assert_called_once_with(temp_vcf.name)
            mock_analyze.assert_called_once_with(mock_vcf_data)
            mock_generate.assert_called_once()
            mock_export.assert_called_once()
            
            # Check that the results have the expected structure
            assert 'risk_results' in results
            assert 'heatmap_data' in results
            assert 'output_dir' in results
            assert results['output_dir'] == output_dir
