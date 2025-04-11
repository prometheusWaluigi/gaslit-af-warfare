"""
Tests for the legal policy simulation module.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Import the module, but mock the dependencies that might not be installed
with patch.dict('sys.modules', {
    'spacy': MagicMock(),
    'PyPDF2': MagicMock(),
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock()
}):
    from src.legal_policy.legal_simulator import LegalPolicySimulator

class TestLegalPolicySimulator:
    """Test suite for the LegalPolicySimulator class."""
    
    @pytest.fixture
    def sample_legal_events(self):
        """Create sample legal events for testing."""
        return [
            {
                "date": "2020-01-15",
                "type": "regulation",
                "title": "Emergency Use Authorization",
                "description": "FDA issues EUA for medical products",
                "impact": 0.8,
                "liability_shield": 0.9
            },
            {
                "date": "2020-03-10",
                "type": "legislation",
                "title": "PREP Act Declaration",
                "description": "HHS invokes PREP Act for liability protection",
                "impact": 0.9,
                "liability_shield": 0.95
            },
            {
                "date": "2021-05-20",
                "type": "court_case",
                "title": "Doe v. Manufacturer",
                "description": "Court upholds liability shield",
                "impact": 0.7,
                "liability_shield": 0.85
            },
            {
                "date": "2022-02-15",
                "type": "scientific_publication",
                "title": "Safety Concerns Study",
                "description": "Study reveals potential safety issues",
                "impact": 0.6,
                "liability_shield": -0.3
            },
            {
                "date": "2022-08-10",
                "type": "court_case",
                "title": "Smith v. Agency",
                "description": "Court questions scope of liability protection",
                "impact": 0.5,
                "liability_shield": -0.4
            }
        ]
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample document paths for testing."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
            temp_file.write(f"Sample document {i+1} content.".encode('utf-8'))
            temp_file.close()
            temp_files.append(temp_file.name)
        
        yield temp_files
        
        # Clean up temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_initialization(self, sample_legal_events):
        """Test that the LegalPolicySimulator class initializes correctly."""
        # Initialize with default config
        simulator = LegalPolicySimulator()
        
        # Check that default parameters were set correctly
        assert simulator.params['time_steps'] == 100
        assert simulator.params['shield_breach_threshold'] == 0.6
        assert simulator.params['evidence_threshold'] == 0.7
        assert len(simulator.events) > 0
        
        # Initialize with custom config
        custom_config = {
            'params': {
                'time_steps': 50,
                'shield_breach_threshold': 0.5,
                'evidence_threshold': 0.8,
                'events': sample_legal_events
            }
        }
        simulator = LegalPolicySimulator(custom_config)
        
        # Check that custom parameters were set correctly
        assert simulator.params['time_steps'] == custom_config['params']['time_steps']
        assert simulator.params['shield_breach_threshold'] == custom_config['params']['shield_breach_threshold']
        assert simulator.params['evidence_threshold'] == custom_config['params']['evidence_threshold']
        assert simulator.events == sample_legal_events
    
    def test_reset_state(self):
        """Test that the reset_state method properly resets the simulation state."""
        simulator = LegalPolicySimulator()
        
        # Run a few steps to change the state
        simulator.time = 10
        simulator.iteration = 5
        simulator.evidence_level = 0.5
        simulator.shield_strength = 0.7
        
        # Reset state
        simulator.reset_state()
        
        # Check that state was reset
        assert simulator.time == 0
        assert simulator.iteration == 0
        assert simulator.evidence_level == 0.0
        assert simulator.shield_strength == 1.0
        assert len(simulator.history['time']) == 0
        assert len(simulator.history['evidence_level']) == 0
        assert len(simulator.history['shield_strength']) == 0
        assert len(simulator.history['shield_breach_probability']) == 0
    
    @patch('src.legal_policy.legal_simulator.HAS_SPACY', True)
    @patch('src.legal_policy.legal_simulator.spacy.load')
    def test_analyze_document(self, mock_load, sample_documents):
        """Test that the analyze_document method correctly analyzes a document."""
        # Set up the mock
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_load.return_value = mock_nlp
        mock_nlp.return_value = mock_doc
        
        # Set up mock entities
        mock_doc.ents = [
            MagicMock(label_="ORG", text="FDA"),
            MagicMock(label_="DATE", text="2020-01-15"),
            MagicMock(label_="PERSON", text="John Doe")
        ]
        
        # Initialize simulator
        simulator = LegalPolicySimulator()
        
        # Analyze document
        doc_analysis = simulator.analyze_document(sample_documents[0])
        
        # Check that the mock was called with the correct file path
        mock_nlp.assert_called_once()
        
        # Check that the returned data has the expected structure
        assert 'entities' in doc_analysis
        assert 'key_phrases' in doc_analysis
        assert 'sentiment' in doc_analysis
        
        # Check that entities were extracted
        assert len(doc_analysis['entities']) == 3
        assert any(e['text'] == "FDA" for e in doc_analysis['entities'])
        assert any(e['text'] == "2020-01-15" for e in doc_analysis['entities'])
        assert any(e['text'] == "John Doe" for e in doc_analysis['entities'])
    
    @patch('src.legal_policy.legal_simulator.HAS_PYPDF', True)
    @patch('src.legal_policy.legal_simulator.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        """Test that the extract_text_from_pdf method correctly extracts text from a PDF."""
        # Set up the mock
        mock_reader = MagicMock()
        mock_pdf_reader.return_value = mock_reader
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_page1.extract_text.return_value = "Page 1 content."
        mock_page2.extract_text.return_value = "Page 2 content."
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            # Initialize simulator
            simulator = LegalPolicySimulator()
            
            # Extract text from PDF
            text = simulator.extract_text_from_pdf(temp_pdf.name)
            
            # Check that the mock was called with the correct file path
            mock_pdf_reader.assert_called_once_with(temp_pdf.name)
            
            # Check that text was extracted from all pages
            assert "Page 1 content." in text
            assert "Page 2 content." in text
    
    def test_calculate_evidence_impact(self, sample_legal_events):
        """Test that the calculate_evidence_impact method correctly calculates impact."""
        # Initialize simulator with sample events
        simulator = LegalPolicySimulator({
            'params': {'events': sample_legal_events}
        })
        
        # Calculate evidence impact
        impact = simulator.calculate_evidence_impact()
        
        # Check that the result is a float between -1 and 1
        assert isinstance(impact, float)
        assert -1 <= impact <= 1
    
    def test_calculate_shield_impact(self, sample_legal_events):
        """Test that the calculate_shield_impact method correctly calculates impact."""
        # Initialize simulator with sample events
        simulator = LegalPolicySimulator({
            'params': {'events': sample_legal_events}
        })
        
        # Calculate shield impact
        impact = simulator.calculate_shield_impact()
        
        # Check that the result is a float between -1 and 1
        assert isinstance(impact, float)
        assert -1 <= impact <= 1
    
    def test_update_evidence_level(self, sample_legal_events):
        """Test that the update_evidence_level method correctly updates evidence level."""
        # Initialize simulator with sample events
        simulator = LegalPolicySimulator({
            'params': {'events': sample_legal_events}
        })
        
        # Store initial evidence level
        initial_level = simulator.evidence_level
        
        # Update evidence level
        simulator.update_evidence_level()
        
        # Check that evidence level was updated
        assert simulator.evidence_level != initial_level
        assert 0 <= simulator.evidence_level <= 1
    
    def test_update_shield_strength(self, sample_legal_events):
        """Test that the update_shield_strength method correctly updates shield strength."""
        # Initialize simulator with sample events
        simulator = LegalPolicySimulator({
            'params': {'events': sample_legal_events}
        })
        
        # Store initial shield strength
        initial_strength = simulator.shield_strength
        
        # Update shield strength
        simulator.update_shield_strength()
        
        # Check that shield strength was updated
        assert simulator.shield_strength != initial_strength
        assert 0 <= simulator.shield_strength <= 1
    
    def test_calculate_shield_breach_probability(self):
        """Test that the calculate_shield_breach_probability method correctly calculates probability."""
        simulator = LegalPolicySimulator()
        
        # Test various combinations of evidence level and shield strength
        test_cases = [
            (0.9, 0.1),  # High evidence, low shield -> high probability
            (0.1, 0.9),  # Low evidence, high shield -> low probability
            (0.5, 0.5)   # Medium evidence, medium shield -> medium probability
        ]
        
        for evidence, shield in test_cases:
            simulator.evidence_level = evidence
            simulator.shield_strength = shield
            probability = simulator.calculate_shield_breach_probability()
            
            # Check that the result is a float between 0 and 1
            assert isinstance(probability, float)
            assert 0 <= probability <= 1
            
            # Check that the probability makes sense for the inputs
            if evidence > 0.8 and shield < 0.2:
                assert probability > 0.7  # High probability
            elif evidence < 0.2 and shield > 0.8:
                assert probability < 0.3  # Low probability
    
    def test_step(self, sample_legal_events):
        """Test that the step method correctly advances the simulation by one step."""
        # Initialize simulator with sample events
        simulator = LegalPolicySimulator({
            'params': {'events': sample_legal_events}
        })
        
        # Store initial values
        initial_time = simulator.time
        initial_iteration = simulator.iteration
        initial_evidence = simulator.evidence_level
        initial_shield = simulator.shield_strength
        
        # Perform a step
        simulator.step()
        
        # Check that values were updated
        assert simulator.time > initial_time
        assert simulator.iteration > initial_iteration
        assert simulator.evidence_level != initial_evidence
        assert simulator.shield_strength != initial_shield
        
        # Check that history was updated
        assert len(simulator.history['time']) == 1
        assert len(simulator.history['evidence_level']) == 1
        assert len(simulator.history['shield_strength']) == 1
        assert len(simulator.history['shield_breach_probability']) == 1
    
    def test_run_simulation(self, sample_legal_events):
        """Test that the run_simulation method completes without errors."""
        # Initialize simulator with sample events
        simulator = LegalPolicySimulator({
            'params': {
                'events': sample_legal_events,
                'time_steps': 10  # Use a small number for testing
            }
        })
        
        # Run simulation
        results = simulator.run_simulation()
        
        # Check that results contain expected keys
        assert 'params' in results
        assert 'final_state' in results
        assert 'history' in results
        assert 'events' in results
        
        # Check that history has the correct length
        assert len(results['history']['time']) == 10
        assert len(results['history']['evidence_level']) == 10
        assert len(results['history']['shield_strength']) == 10
        assert len(results['history']['shield_breach_probability']) == 10
        
        # Check that final state contains expected keys
        assert 'time' in results['final_state']
        assert 'evidence_level' in results['final_state']
        assert 'shield_strength' in results['final_state']
        assert 'shield_breach_probability' in results['final_state']
    
    def test_save_results(self, sample_legal_events, output_dir):
        """Test that results can be saved to a file."""
        # Initialize simulator with sample events
        simulator = LegalPolicySimulator({
            'params': {
                'events': sample_legal_events,
                'time_steps': 5  # Use a small number for testing
            }
        })
        
        # Run a short simulation
        results = simulator.run_simulation()
        
        # Save results
        output_file = os.path.join(output_dir, 'legal_results.json')
        simulator.save_results(results, output_file)
        
        # Check that the file was created
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
        
        # Check that the file contains valid JSON
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        # Check that the loaded results match the original results
        assert loaded_results['params'] == results['params']
        assert loaded_results['final_state'] == results['final_state']
        assert loaded_results['events'] == results['events']
    
    @patch('src.legal_policy.legal_simulator.LegalPolicySimulator.analyze_document')
    def test_analyze_legal_corpus(self, mock_analyze, sample_documents):
        """Test that the analyze_legal_corpus method correctly analyzes a corpus of documents."""
        # Set up the mock
        mock_analyze.side_effect = [
            {'entities': [{'text': 'FDA', 'label': 'ORG'}], 'key_phrases': ['regulation'], 'sentiment': 0.2},
            {'entities': [{'text': 'HHS', 'label': 'ORG'}], 'key_phrases': ['liability'], 'sentiment': -0.1},
            {'entities': [{'text': 'Court', 'label': 'ORG'}], 'key_phrases': ['case'], 'sentiment': 0.0}
        ]
        
        # Initialize simulator
        simulator = LegalPolicySimulator()
        
        # Analyze corpus
        corpus_analysis = simulator.analyze_legal_corpus(sample_documents)
        
        # Check that the mock was called for each document
        assert mock_analyze.call_count == len(sample_documents)
        
        # Check that the returned data has the expected structure
        assert 'documents' in corpus_analysis
        assert 'entity_counts' in corpus_analysis
        assert 'key_phrase_counts' in corpus_analysis
        assert 'average_sentiment' in corpus_analysis
        
        # Check that document analyses were included
        assert len(corpus_analysis['documents']) == len(sample_documents)
        
        # Check that entity counts were aggregated
        assert 'FDA' in corpus_analysis['entity_counts']
        assert 'HHS' in corpus_analysis['entity_counts']
        assert 'Court' in corpus_analysis['entity_counts']
        
        # Check that key phrase counts were aggregated
        assert 'regulation' in corpus_analysis['key_phrase_counts']
        assert 'liability' in corpus_analysis['key_phrase_counts']
        assert 'case' in corpus_analysis['key_phrase_counts']
        
        # Check that average sentiment was calculated
        assert -1 <= corpus_analysis['average_sentiment'] <= 1
    
    @pytest.mark.parametrize("evidence_level,shield_strength", [
        (0.9, 0.1),  # High evidence, low shield -> high breach probability
        (0.1, 0.9),  # Low evidence, high shield -> low breach probability
    ])
    def test_breach_probabilities(self, evidence_level, shield_strength):
        """Test that different parameter combinations lead to different breach probabilities."""
        # Initialize simulator
        simulator = LegalPolicySimulator()
        
        # Set evidence level and shield strength
        simulator.evidence_level = evidence_level
        simulator.shield_strength = shield_strength
        
        # Calculate breach probability
        probability = simulator.calculate_shield_breach_probability()
        
        # For high evidence and low shield, expect high probability
        if evidence_level > 0.8 and shield_strength < 0.2:
            assert probability > 0.7
        
        # For low evidence and high shield, expect low probability
        if evidence_level < 0.2 and shield_strength > 0.8:
            assert probability < 0.3
