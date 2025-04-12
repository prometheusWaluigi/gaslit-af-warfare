#!/usr/bin/env python3
"""
Legal Policy Simulator for GASLIT-AF WARSTACK

This module simulates legal and policy dynamics related to GASLIT-AF syndrome,
including liability shield analysis, evidence timelines, and class action viability.
"""

import os
import sys
import json
import time
import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# Optional imports for NLP and document processing
HAS_SPACY = False
HAS_PYPDF = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    pass

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF = True
except ImportError:
    pass


class LegalPolicySimulator:
    """
    Simulates legal and policy dynamics related to GASLIT-AF syndrome.
    """
    
    # Key legal frameworks and their properties
    LEGAL_FRAMEWORKS = {
        'NCVIA': {
            'description': 'National Childhood Vaccine Injury Act',
            'shield_strength': 0.9,
            'evidence_threshold': 0.8,
            'year_enacted': 1986
        },
        'PREP_Act': {
            'description': 'Public Readiness and Emergency Preparedness Act',
            'shield_strength': 0.95,
            'evidence_threshold': 0.9,
            'year_enacted': 2005
        },
        'EUA': {
            'description': 'Emergency Use Authorization',
            'shield_strength': 0.85,
            'evidence_threshold': 0.75,
            'year_enacted': 2004
        },
        'Common_Law': {
            'description': 'Common Law Liability',
            'shield_strength': 0.3,
            'evidence_threshold': 0.6,
            'year_enacted': 0
        }
    }
    
    def __init__(self, config=None):
        """Initialize the legal policy simulator."""
        # Default parameters
        self.params = {
            'time_steps': 100,
            'initial_evidence_level': 0.1,
            'evidence_growth_rate': 0.02,
            'shield_decay_rate': 0.01,
            'shield_breach_threshold': 0.6,
            'simulation_steps': 100,
            'evidence_threshold': 0.7,
            'events': [
                {
                    "date": "2020-01-15",
                    "type": "regulation",
                    "title": "Emergency Use Authorization",
                    "description": "FDA issues EUA for medical products",
                    "impact": 0.8,
                    "liability_shield": 0.9
                }
            ]
        }

        # Default configuration
        self.config = {
            'output_dir': 'results/legal_policy',
            'random_seed': 42,
            'timeline_start': '2019-01-01',
            'timeline_end': '2025-01-01',
            'simulation_steps': 100,
            'evidence_growth_rate': 0.02
        }
        
        # Update with user configuration if provided
        if config is not None:
            if 'params' in config:
                self.params.update(config['params'])
            for key, value in config.items():
                if key != 'params':
                    self.config[key] = value
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['output_dir'], 'legal_simulator.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LegalPolicySimulator')
        
        self.reset_state()
    
    def reset_state(self):
        """Reset the simulation state to initial values."""
        self.time = 0
        self.iteration = 0
        self.evidence_level = 0.0  # Current evidence level
        self.events = self.params['events']  # Legal events affecting the simulation
        self.shield_strength = 1.0  # Current shield strength
        
        # Initialize NLP if available
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy NLP model")
            except Exception as e:
                self.logger.warning(f"Could not load spaCy model: {e}")
        
        # Initialize history tracking
        self.history = {
            'time': [],
            'evidence_level': [],
            'shield_strength': [],
            'shield_breach_probability': []
        }

    def analyze_legal_corpus(self, documents):
        """Analyze a corpus of legal documents."""
        if not documents:
            self.logger.warning("No documents provided for analysis")
            return None

        results = []
        for doc_path in documents:
            doc_analysis = self.analyze_document(doc_path)
            if doc_analysis:
                results.append({
                    'path': doc_path,
                    'analysis': doc_analysis
                })

        # Aggregate results
        entity_counts = {}
        key_phrase_counts = {}
        total_sentiment = 0.0
        
        for result in results:
            analysis = result['analysis']
            if analysis and 'entities' in analysis:
                for entity in analysis['entities']:
                    entity_counts[entity['text']] = entity_counts.get(entity['text'], 0) + 1
            if analysis and 'key_phrases' in analysis:
                for phrase in analysis['key_phrases']:
                    key_phrase_counts[phrase] = key_phrase_counts.get(phrase, 0) + 1
            if analysis and 'sentiment' in analysis:
                total_sentiment += analysis['sentiment']
        
        average_sentiment = total_sentiment / len(results) if results else 0.0
        
        return {
            'document_count': len(results),
            'documents': results,
            'entity_counts': entity_counts,
            'key_phrase_counts': key_phrase_counts,
            'average_sentiment': average_sentiment
        }

    def analyze_document(self, document_path):
        """Analyze a legal document for relevant information."""
        if not os.path.exists(document_path):
            self.logger.error(f"Document not found: {document_path}")
            return None
        
        # Extract text based on file type
        text = ""
        file_ext = os.path.splitext(document_path)[1].lower()
        
        if file_ext == '.pdf' and HAS_PYPDF:
            text = self.extract_text_from_pdf(document_path)
        elif file_ext in ['.txt', '.md', '.html', '.htm']:
            with open(document_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            self.logger.error(f"Unsupported file format: {file_ext}")
            return None
        
        # Analyze text with NLP if available
        if self.nlp is not None:
            doc = self.nlp(text)
            
            # Extract key information
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
            orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            
            # Look for legal terms
            legal_terms = []
            key_phrases = []
            for token in doc:
                if token.text.lower() in ['liability', 'immunity', 'compensation', 'damages', 'lawsuit']:
                    legal_terms.append(token.text)
                if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                    key_phrases.append(f"{token.text} {token.head.text}")
            
            return {
                'entities': [{'text': text, 'label': label} for text, label in entities],
                'dates': dates,
                'organizations': orgs,
                'legal_terms': legal_terms,
                'key_phrases': key_phrases,
                'sentiment': 0.0,  # Default neutral sentiment
                'text_length': len(text)
            }
        
        # Basic analysis without NLP
        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentiment': 0.0  # Default neutral sentiment
        }
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        if not HAS_PYPDF:
            self.logger.error("PyPDF2 is required for PDF processing but not installed.")
            return ""
        
        text = ""
        try:
            pdf_reader = PdfReader(pdf_path)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
        
        return text
    
    def calculate_evidence_impact(self):
        """Calculate the impact of new evidence on the simulation."""
        # Calculate impact based on current events
        total_impact = 0.0
        for event in self.events:
            if 'impact' in event:
                total_impact += event['impact']
        
        # Normalize impact to [-1, 1] range
        if total_impact != 0:
            total_impact = np.clip(total_impact / len(self.events), -1, 1)
        
        return total_impact
    
    def calculate_shield_impact(self):
        """Calculate the impact of evidence on liability shields."""
        # Calculate impact based on current events
        total_impact = 0.0
        for event in self.events:
            if 'liability_shield' in event:
                total_impact += event['liability_shield']
        
        # Normalize impact to [-1, 1] range
        if total_impact != 0:
            total_impact = np.clip(total_impact / len(self.events), -1, 1)
        
        return total_impact
    
    def update_evidence_level(self):
        """Update the evidence level based on simulation state."""
        # Calculate evidence impact from events
        impact = self.calculate_evidence_impact()
        
        # Apply impact to evidence level
        self.evidence_level = np.clip(self.evidence_level + impact, 0, 1)
        
        return self.evidence_level
    
    def update_shield_strength(self):
        """Update the liability shield strength based on evidence levels."""
        # Calculate shield decay
        shield_decay = self.calculate_shield_impact()
        
        # Calculate evidence change since last update
        if self.history['evidence_level']:
            prev_evidence = self.history['evidence_level'][-1]
            evidence_change = self.evidence_level - prev_evidence
            shield_decay += 0.1 * evidence_change  # Additional decay based on evidence change
        
        # Apply decay to shield strength
        self.shield_strength = np.clip(self.shield_strength - shield_decay, 0, 1)
        
        return self.shield_strength
    
    def calculate_shield_breach_probability(self):
        """Calculate the probability of liability shield breach."""
        # Compare evidence levels to shield thresholds
        thresholds = np.array([
            self.LEGAL_FRAMEWORKS['NCVIA']['evidence_threshold'],
            self.LEGAL_FRAMEWORKS['PREP_Act']['evidence_threshold'],
            self.LEGAL_FRAMEWORKS['EUA']['evidence_threshold'],
            self.LEGAL_FRAMEWORKS['Common_Law']['evidence_threshold']
        ])
        
        # Calculate breach probability for each shield
        # Higher evidence and lower shield strength increase breach probability
        evidence_factor = np.mean(self.evidence_level)
        shield_factor = np.mean(self.shield_strength)
        threshold_factor = np.mean(thresholds)
        
        # Overall breach probability
        breach_prob = evidence_factor * (1 - shield_factor) / threshold_factor
        breach_prob = min(1.0, max(0.0, breach_prob))
        
        return breach_prob
    
    def step(self):
        """Advance the simulation by one time step."""
        # Update evidence levels
        self.update_evidence_level()
        
        # Update shield strengths
        self.update_shield_strength()
        
        # Calculate breach probability
        breach_prob = self.calculate_shield_breach_probability()
        
        # Update time and iteration
        self.time += 1
        self.iteration += 1
        
        # Record history
        self.history['time'].append(self.time)
        self.history['evidence_level'].append(self.evidence_level)
        self.history['shield_strength'].append(self.shield_strength)
        self.history['shield_breach_probability'].append(breach_prob)
        
        return {
            'time': self.time,
            'evidence_level': self.evidence_level,
            'shield_strength': self.shield_strength,
            'shield_breach_probability': breach_prob
        }
    
    def run_simulation(self, steps=None):
        """Run the simulation for the specified number of steps."""
        if steps is None:
            steps = self.params['time_steps']
        
        self.logger.info(f"Running legal policy simulation for {steps} steps")
        
        # Run the simulation
        start_time = time.time()
        for _ in range(steps):
            self.step()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        
        # Return results
        return self.get_results()
    
    def get_results(self):
        """Get the simulation results."""
        return {
            'params': self.params,
            'final_state': {
                'time': self.time,
                'evidence_level': self.evidence_level,
                'shield_strength': self.shield_strength,
                'shield_breach_probability': self.calculate_shield_breach_probability()
            },
            'history': {
                'time': self.history['time'],
                'evidence_level': self.history['evidence_level'],
                'shield_strength': self.history['shield_strength'],
                'shield_breach_probability': self.history['shield_breach_probability']
            },
            'frameworks': self.LEGAL_FRAMEWORKS,
            'events': self.events
        }
    
    def visualize_simulation_results(self, save_path=None):
        """Visualize the simulation results."""
        if not self.history['time']:
            self.logger.warning("No simulation history available to visualize.")
            return None
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot evidence level
        axs[0].plot(self.history['time'], self.history['evidence_level'], 'b-', label='Evidence Level')
        axs[0].set_ylabel('Evidence Level')
        axs[0].set_title('Evidence Level Over Time')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot shield strength
        axs[1].plot(self.history['time'], self.history['shield_strength'], 'g-', label='Shield Strength')
        axs[1].set_ylabel('Shield Strength')
        axs[1].set_title('Shield Strength Over Time')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot breach probability
        axs[2].plot(self.history['time'], self.history['shield_breach_probability'], 'r-')
        axs[2].set_ylabel('Probability')
        axs[2].set_xlabel('Time')
        axs[2].set_title('Liability Shield Breach Probability')
        axs[2].grid(True)
        
        # Add overall title
        fig.suptitle('Legal Policy Simulation Results', fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.97))  # Adjust for suptitle
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Simulation results visualization saved to {save_path}")
        
        return fig
    
    def save_results(self, results=None, filename=None):
        """Save the simulation results to a file."""
        if results is None:
            results = self.get_results()
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"legal_simulation_{timestamp}.json"
        
        filepath = os.path.join(self.config['output_dir'], filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
        
        return filepath


def run_sample_simulation():
    """Run a sample legal policy simulation."""
    # Create a simulator with default configuration
    simulator = LegalPolicySimulator()
    
    # Run the simulation
    simulator.run_simulation(steps=50)
    
    # Visualize the results
    output_dir = simulator.config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    simulator.visualize_simulation_results(save_path=os.path.join(output_dir, 'legal_simulation_results.png'))
    
    # Save the results
    simulator.save_results('sample_results.json')
    
    print("Sample legal policy simulation completed successfully.")
    print(f"Results saved to {output_dir}")
    
    return simulator


if __name__ == "__main__":
    # Run a sample simulation if the script is executed directly
    simulator = run_sample_simulation()
