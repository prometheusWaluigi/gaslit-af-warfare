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
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


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
        # Default configuration
        self.config = {
            'output_dir': 'results/legal_policy',
            'simulation_steps': 100,
            'initial_evidence_level': 0.1,
            'evidence_growth_rate': 0.02,
            'shield_decay_rate': 0.01,
            'random_seed': 42,
            'timeline_start': '2019-01-01',
            'timeline_end': '2025-01-01'
        }
        
        # Update with user configuration if provided
        if config is not None:
            self.config.update(config)
        
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
        
        # Initialize simulation state
        self.time = 0
        self.iteration = 0
        self.evidence_level = np.zeros(4)  # Evidence levels for different claim types
        self.shield_strength = np.array([
            self.LEGAL_FRAMEWORKS['NCVIA']['shield_strength'],
            self.LEGAL_FRAMEWORKS['PREP_Act']['shield_strength'],
            self.LEGAL_FRAMEWORKS['EUA']['shield_strength'],
            self.LEGAL_FRAMEWORKS['Common_Law']['shield_strength']
        ])
        
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
            'time': [0],
            'evidence_level': [self.evidence_level.copy()],
            'shield_strength': [self.shield_strength.copy()],
            'breach_probability': [0.0]
        }
    
    def analyze_document(self, document_path):
        """Analyze a legal document for relevant information."""
        if not os.path.exists(document_path):
            self.logger.error(f"Document not found: {document_path}")
            return None
        
        # Extract text based on file type
        text = ""
        file_ext = os.path.splitext(document_path)[1].lower()
        
        if file_ext == '.pdf' and HAS_PYPDF2:
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
            for token in doc:
                if token.text.lower() in ['liability', 'immunity', 'compensation', 'damages', 'lawsuit']:
                    legal_terms.append(token.text)
            
            return {
                'entities': entities,
                'dates': dates,
                'organizations': orgs,
                'legal_terms': legal_terms,
                'text_length': len(text)
            }
        
        # Basic analysis without NLP
        return {
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        if not HAS_PYPDF2:
            self.logger.error("PyPDF2 is required for PDF processing but not installed.")
            return ""
        
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
        
        return text
    
    def calculate_evidence_impact(self, new_evidence):
        """Calculate the impact of new evidence on the simulation."""
        # Convert evidence value to an impact on different claim types
        # Claim types: [causation, failure_to_warn, negligence, fraud]
        impact = np.zeros(4)
        
        # Distribute evidence impact across claim types
        impact[0] = new_evidence * 0.4  # Causation
        impact[1] = new_evidence * 0.3  # Failure to warn
        impact[2] = new_evidence * 0.2  # Negligence
        impact[3] = new_evidence * 0.1  # Fraud
        
        return impact
    
    def calculate_shield_impact(self, evidence_change):
        """Calculate the impact of evidence on liability shields."""
        # Different shields are affected differently by evidence types
        # Shields: [NCVIA, PREP_Act, EUA, Common_Law]
        
        # Create impact matrix: rows=shields, cols=evidence types
        impact_matrix = np.array([
            [0.02, 0.03, 0.01, 0.05],  # NCVIA impact from each evidence type
            [0.01, 0.02, 0.01, 0.04],  # PREP_Act impact
            [0.03, 0.04, 0.02, 0.06],  # EUA impact
            [0.05, 0.06, 0.04, 0.08]   # Common_Law impact
        ])
        
        # Calculate shield decay based on evidence change
        shield_decay = np.dot(impact_matrix, evidence_change)
        
        return shield_decay
    
    def update_evidence_level(self, new_evidence=None):
        """Update the evidence level based on simulation state."""
        if new_evidence is None:
            # Default evidence growth
            base_growth = self.config['evidence_growth_rate']
            
            # Random variation in evidence growth
            variation = np.random.normal(0, 0.005, 4)
            
            # Calculate evidence change
            evidence_change = base_growth + variation
            
            # Apply change to evidence levels
            self.evidence_level = np.clip(self.evidence_level + evidence_change, 0, 1)
        else:
            # Apply specific new evidence
            impact = self.calculate_evidence_impact(new_evidence)
            self.evidence_level = np.clip(self.evidence_level + impact, 0, 1)
        
        return self.evidence_level
    
    def update_shield_strength(self):
        """Update the liability shield strength based on evidence levels."""
        # Calculate evidence change since last update
        prev_evidence = self.history['evidence_level'][-1]
        evidence_change = self.evidence_level - prev_evidence
        
        # Calculate shield decay based on evidence change
        shield_decay = self.calculate_shield_impact(evidence_change)
        
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
        self.history['evidence_level'].append(self.evidence_level.copy())
        self.history['shield_strength'].append(self.shield_strength.copy())
        self.history['breach_probability'].append(breach_prob)
        
        return {
            'time': self.time,
            'evidence_level': self.evidence_level.copy(),
            'shield_strength': self.shield_strength.copy(),
            'breach_probability': breach_prob
        }
    
    def run_simulation(self, steps=None):
        """Run the simulation for the specified number of steps."""
        if steps is None:
            steps = self.config['simulation_steps']
        
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
            'config': self.config,
            'final_state': {
                'time': self.time,
                'evidence_level': self.evidence_level.tolist(),
                'shield_strength': self.shield_strength.tolist(),
                'breach_probability': self.calculate_shield_breach_probability()
            },
            'history': {
                'time': self.history['time'],
                'evidence_level': [e.tolist() for e in self.history['evidence_level']],
                'shield_strength': [s.tolist() for s in self.history['shield_strength']],
                'breach_probability': self.history['breach_probability']
            },
            'frameworks': self.LEGAL_FRAMEWORKS
        }
    
    def visualize_simulation_results(self, save_path=None):
        """Visualize the simulation results."""
        if not self.history['time']:
            self.logger.warning("No simulation history available to visualize.")
            return None
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot evidence levels
        for i, claim_type in enumerate(['Causation', 'Failure to Warn', 'Negligence', 'Fraud']):
            evidence_values = [e[i] for e in self.history['evidence_level']]
            axs[0].plot(self.history['time'], evidence_values, label=claim_type)
        
        axs[0].set_ylabel('Evidence Level')
        axs[0].set_title('Evidence Levels by Claim Type')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot shield strengths
        for i, framework in enumerate(['NCVIA', 'PREP Act', 'EUA', 'Common Law']):
            shield_values = [s[i] for s in self.history['shield_strength']]
            axs[1].plot(self.history['time'], shield_values, label=framework)
        
        axs[1].set_ylabel('Shield Strength')
        axs[1].set_title('Liability Shield Strength')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot breach probability
        axs[2].plot(self.history['time'], self.history['breach_probability'], 'r-')
        axs[2].set_ylabel('Probability')
        axs[2].set_xlabel('Time')
        axs[2].set_title('Liability Shield Breach Probability')
        axs[2].grid(True)
        
        # Add overall title
        fig.suptitle('Legal Policy Simulation Results', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Simulation results visualization saved to {save_path}")
        
        return fig
    
    def save_results(self, filename=None):
        """Save the simulation results to a file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"legal_simulation_{timestamp}.json"
        
        filepath = os.path.join(self.config['output_dir'], filename)
        
        # Get results
        results = self.get_results()
        
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
