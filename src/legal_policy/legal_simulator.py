#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal & Policy Simulator for GASLIT-AF WARSTACK

This module implements legal and policy simulation to analyze NCVIA, EUA claims,
suppressed data trails, and ethical obligations.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

# Optional imports for NLP and document processing
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logging.warning("spaCy not available. Text analysis will be limited.")

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    logging.warning("PyPDF2 not available. PDF parsing will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalPolicySimulator:
    """
    Simulates legal and policy implications of GASLIT-AF scenarios.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the legal and policy simulator.
        
        Args:
            config: Configuration dictionary with simulator parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.params = {
            'liability_threshold': 0.75,  # Threshold for liability breach (0-1)
            'evidence_threshold': 0.65,  # Threshold for sufficient evidence (0-1)
            'class_action_threshold': 0.7,  # Threshold for class action viability (0-1)
            'simulation_years': 5,  # Number of years to simulate
            'simulation_start_date': '2020-01-01',  # Start date for simulation
            'legal_precedents': [
                'Wyeth v. Levine',
                'Bruesewitz v. Wyeth',
                'Doe v. Rumsfeld',
                'Buck v. Bell'
            ]
        }
        
        # Update with user-provided parameters
        self.params.update(self.config.get('params', {}))
        
        # Initialize NLP if available
        self._initialize_nlp()
        
        # Initialize state variables
        self.reset_state()
    
    def _initialize_nlp(self):
        """Initialize NLP components if available."""
        self.nlp = None
        if HAS_SPACY:
            try:
                # Load a small English model for efficiency
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Initialized spaCy NLP model")
            except Exception as e:
                logger.error(f"Error loading spaCy model: {e}")
    
    def reset_state(self):
        """Reset the simulation state."""
        # Parse start date
        start_date = datetime.strptime(self.params['simulation_start_date'], '%Y-%m-%d')
        
        # Initialize timeline
        self.timeline = pd.date_range(
            start=start_date,
            periods=self.params['simulation_years'] * 12,  # Monthly intervals
            freq='M'
        )
        
        # Initialize state variables
        n_steps = len(self.timeline)
        self.evidence_level = np.zeros(n_steps)
        self.liability_risk = np.zeros(n_steps)
        self.class_action_viability = np.zeros(n_steps)
        self.shield_breach_probability = np.zeros(n_steps)
        
        # Initialize with small random values
        self.evidence_level[0] = 0.1 + np.random.normal(0, 0.02)
        self.liability_risk[0] = 0.05 + np.random.normal(0, 0.01)
        self.class_action_viability[0] = 0.02 + np.random.normal(0, 0.01)
        self.shield_breach_probability[0] = 0.01 + np.random.normal(0, 0.005)
        
        # Ensure values are in valid range
        self.evidence_level[0] = np.clip(self.evidence_level[0], 0, 1)
        self.liability_risk[0] = np.clip(self.liability_risk[0], 0, 1)
        self.class_action_viability[0] = np.clip(self.class_action_viability[0], 0, 1)
        self.shield_breach_probability[0] = np.clip(self.shield_breach_probability[0], 0, 1)
        
        # Initialize events timeline
        self.events = []
        
        # Add some initial events
        self.events.append({
            'date': start_date,
            'type': 'regulatory',
            'description': 'Initial EUA issuance',
            'impact': {
                'evidence_level': 0,
                'liability_risk': -0.1,  # EUA reduces initial liability
                'class_action_viability': -0.05,
                'shield_breach_probability': -0.1
            }
        })
        
        # History for tracking simulation states
        self.history = {
            'timeline': self.timeline,
            'evidence_level': [],
            'liability_risk': [],
            'class_action_viability': [],
            'shield_breach_probability': [],
            'events': []
        }
    
    def load_legal_documents(self, documents_dir: str) -> Dict[str, Any]:
        """
        Load and parse legal documents.
        
        Args:
            documents_dir: Directory containing legal documents
            
        Returns:
            Dictionary with parsed document data
        """
        if not os.path.exists(documents_dir):
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
        
        logger.info(f"Loading legal documents from {documents_dir}")
        
        documents = {}
        
        # Process PDF files
        if HAS_PYPDF2:
            pdf_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
            for pdf_file in pdf_files:
                pdf_path = os.path.join(documents_dir, pdf_file)
                try:
                    reader = PdfReader(pdf_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    documents[pdf_file] = {
                        'text': text,
                        'pages': len(reader.pages),
                        'type': 'pdf'
                    }
                    
                    logger.info(f"Loaded PDF: {pdf_file} ({len(reader.pages)} pages)")
                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_file}: {e}")
        
        # Process text files
        txt_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.txt')]
        for txt_file in txt_files:
            txt_path = os.path.join(documents_dir, txt_file)
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                documents[txt_file] = {
                    'text': text,
                    'type': 'txt'
                }
                
                logger.info(f"Loaded text file: {txt_file}")
            except Exception as e:
                logger.error(f"Error loading text file {txt_file}: {e}")
        
        # Process JSON files (e.g., for structured legal data)
        json_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.json')]
        for json_file in json_files:
            json_path = os.path.join(documents_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                documents[json_file] = {
                    'data': data,
                    'type': 'json'
                }
                
                logger.info(f"Loaded JSON file: {json_file}")
            except Exception as e:
                logger.error(f"Error loading JSON file {json_file}: {e}")
        
        return documents
    
    def analyze_documents(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze legal documents for relevant information.
        
        Args:
            documents: Dictionary with parsed document data
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing legal documents...")
        
        analysis_results = {
            'key_terms': {},
            'liability_indicators': [],
            'shield_breach_indicators': [],
            'evidence_indicators': [],
            'class_action_indicators': []
        }
        
        # Define key terms to look for
        key_terms = {
            'liability': ['liability', 'negligence', 'tort', 'damages', 'compensation'],
            'shield': ['immunity', 'shield', 'protection', 'exemption', 'PREP Act', 'NCVIA'],
            'evidence': ['evidence', 'data', 'study', 'trial', 'research', 'finding'],
            'class_action': ['class action', 'mass tort', 'MDL', 'multi-district', 'consolidated']
        }
        
        # Count term occurrences
        for term_category, terms in key_terms.items():
            analysis_results['key_terms'][term_category] = {}
            
            for term in terms:
                analysis_results['key_terms'][term_category][term] = 0
        
        # Process each document
        for doc_name, doc_data in documents.items():
            if doc_data['type'] in ['pdf', 'txt']:
                text = doc_data['text'].lower()
                
                # Count term occurrences
                for term_category, terms in key_terms.items():
                    for term in terms:
                        count = text.count(term.lower())
                        analysis_results['key_terms'][term_category][term] += count
                
                # Perform NLP analysis if available
                if HAS_SPACY and self.nlp:
                    # Process text in chunks to avoid memory issues
                    chunk_size = 100000  # Process ~100k characters at a time
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i+chunk_size]
                        doc = self.nlp(chunk)
                        
                        # Extract entities
                        for ent in doc.ents:
                            if ent.label_ in ['ORG', 'LAW', 'DATE']:
                                # Add relevant entities to indicators
                                if any(term in ent.text.lower() for term in key_terms['liability']):
                                    analysis_results['liability_indicators'].append(ent.text)
                                elif any(term in ent.text.lower() for term in key_terms['shield']):
                                    analysis_results['shield_breach_indicators'].append(ent.text)
                                elif any(term in ent.text.lower() for term in key_terms['evidence']):
                                    analysis_results['evidence_indicators'].append(ent.text)
                                elif any(term in ent.text.lower() for term in key_terms['class_action']):
                                    analysis_results['class_action_indicators'].append(ent.text)
        
        # Remove duplicates from indicators
        for key in ['liability_indicators', 'shield_breach_indicators', 
                   'evidence_indicators', 'class_action_indicators']:
            analysis_results[key] = list(set(analysis_results[key]))
        
        logger.info("Document analysis completed")
        
        return analysis_results
    
    def simulate_timeline_step(self, step: int):
        """
        Simulate a single step in the legal timeline.
        
        Args:
            step: Current step index
        """
        if step == 0:
            return  # Skip first step as it's already initialized
        
        # Get previous values
        prev_evidence = self.evidence_level[step-1]
        prev_liability = self.liability_risk[step-1]
        prev_class_action = self.class_action_viability[step-1]
        prev_shield_breach = self.shield_breach_probability[step-1]
        
        # Base growth rates (evidence accumulates over time)
        evidence_growth = 0.01 + 0.02 * prev_evidence  # Evidence builds on itself
        liability_growth = 0.005 + 0.01 * prev_evidence  # Liability grows with evidence
        class_action_growth = 0.002 + 0.015 * prev_liability  # Class actions follow liability
        shield_breach_growth = 0.001 + 0.02 * prev_evidence * prev_liability  # Shield breach depends on both
        
        # Add some randomness
        evidence_growth += np.random.normal(0, 0.005)
        liability_growth += np.random.normal(0, 0.003)
        class_action_growth += np.random.normal(0, 0.002)
        shield_breach_growth += np.random.normal(0, 0.001)
        
        # Calculate new values
        self.evidence_level[step] = prev_evidence + evidence_growth
        self.liability_risk[step] = prev_liability + liability_growth
        self.class_action_viability[step] = prev_class_action + class_action_growth
        self.shield_breach_probability[step] = prev_shield_breach + shield_breach_growth
        
        # Check for threshold events
        current_date = self.timeline[step]
        
        # Evidence threshold event
        if (self.evidence_level[step] >= self.params['evidence_threshold'] and 
            prev_evidence < self.params['evidence_threshold']):
            self.events.append({
                'date': current_date,
                'type': 'evidence',
                'description': 'Evidence threshold reached',
                'impact': {
                    'evidence_level': 0.05,
                    'liability_risk': 0.1,
                    'class_action_viability': 0.08,
                    'shield_breach_probability': 0.05
                }
            })
            
            # Apply immediate impact
            self.liability_risk[step] += 0.1
            self.class_action_viability[step] += 0.08
            self.shield_breach_probability[step] += 0.05
        
        # Class action threshold event
        if (self.class_action_viability[step] >= self.params['class_action_threshold'] and 
            prev_class_action < self.params['class_action_threshold']):
            self.events.append({
                'date': current_date,
                'type': 'legal',
                'description': 'Class action viability threshold reached',
                'impact': {
                    'evidence_level': 0.02,
                    'liability_risk': 0.15,
                    'class_action_viability': 0.2,
                    'shield_breach_probability': 0.1
                }
            })
            
            # Apply immediate impact
            self.evidence_level[step] += 0.02
            self.liability_risk[step] += 0.15
            self.class_action_viability[step] += 0.2
            self.shield_breach_probability[step] += 0.1
        
        # Liability threshold event
        if (self.liability_risk[step] >= self.params['liability_threshold'] and 
            prev_liability < self.params['liability_threshold']):
            self.events.append({
                'date': current_date,
                'type': 'legal',
                'description': 'Liability threshold reached',
                'impact': {
                    'evidence_level': 0,
                    'liability_risk': 0.2,
                    'class_action_viability': 0.15,
                    'shield_breach_probability': 0.2
                }
            })
            
            # Apply immediate impact
            self.liability_risk[step] += 0.2
            self.class_action_viability[step] += 0.15
            self.shield_breach_probability[step] += 0.2
        
        # Ensure values stay in valid range
        self.evidence_level[step] = np.clip(self.evidence_level[step], 0, 1)
        self.liability_risk[step] = np.clip(self.liability_risk[step], 0, 1)
        self.class_action_viability[step] = np.clip(self.class_action_viability[step], 0, 1)
        self.shield_breach_probability[step] = np.clip(self.shield_breach_probability[step], 0, 1)
        
        # Add random events occasionally
        if np.random.random() < 0.05:  # 5% chance of random event
            event_types = ['regulatory', 'scientific', 'legal', 'media']
            event_type = np.random.choice(event_types)
            
            if event_type == 'regulatory':
                self.events.append({
                    'date': current_date,
                    'type': 'regulatory',
                    'description': 'Regulatory guidance update',
                    'impact': {
                        'evidence_level': 0.01,
                        'liability_risk': np.random.uniform(-0.05, 0.05),
                        'class_action_viability': np.random.uniform(-0.03, 0.03),
                        'shield_breach_probability': np.random.uniform(-0.02, 0.02)
                    }
                })
            elif event_type == 'scientific':
                self.events.append({
                    'date': current_date,
                    'type': 'scientific',
                    'description': 'New scientific study published',
                    'impact': {
                        'evidence_level': np.random.uniform(0.02, 0.1),
                        'liability_risk': np.random.uniform(0.01, 0.08),
                        'class_action_viability': np.random.uniform(0.01, 0.05),
                        'shield_breach_probability': np.random.uniform(0.01, 0.03)
                    }
                })
            elif event_type == 'legal':
                self.events.append({
                    'date': current_date,
                    'type': 'legal',
                    'description': 'Related legal precedent established',
                    'impact': {
                        'evidence_level': 0,
                        'liability_risk': np.random.uniform(0.05, 0.15),
                        'class_action_viability': np.random.uniform(0.05, 0.1),
                        'shield_breach_probability': np.random.uniform(0.03, 0.08)
                    }
                })
            elif event_type == 'media':
                self.events.append({
                    'date': current_date,
                    'type': 'media',
                    'description': 'Major media coverage of related issues',
                    'impact': {
                        'evidence_level': np.random.uniform(0.01, 0.03),
                        'liability_risk': np.random.uniform(0.02, 0.07),
                        'class_action_viability': np.random.uniform(0.03, 0.08),
                        'shield_breach_probability': np.random.uniform(0.01, 0.04)
                    }
                })
            
            # Apply event impact
            latest_event = self.events[-1]
            impact = latest_event['impact']
            
            self.evidence_level[step] += impact['evidence_level']
            self.liability_risk[step] += impact['liability_risk']
            self.class_action_viability[step] += impact['class_action_viability']
            self.shield_breach_probability[step] += impact['shield_breach_probability']
            
            # Ensure values stay in valid range after event
            self.evidence_level[step] = np.clip(self.evidence_level[step], 0, 1)
            self.liability_risk[step] = np.clip(self.liability_risk[step], 0, 1)
            self.class_action_viability[step] = np.clip(self.class_action_viability[step], 0, 1)
            self.shield_breach_probability[step] = np.clip(self.shield_breach_probability[step], 0, 1)
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the full legal timeline simulation.
        
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting legal timeline simulation for {self.params['simulation_years']} years")
        
        # Reset state before simulation
        self.reset_state()
        
        # Run simulation steps
        n_steps = len(self.timeline)
        for step in range(1, n_steps):
            self.simulate_timeline_step(step)
            
            # Log progress every 25% of steps
            if step % (n_steps // 4) == 0:
                logger.info(f"Simulation progress: {step / n_steps * 100:.1f}%")
        
        logger.info("Simulation completed")
        
        # Record final history
        self.history['evidence_level'] = self.evidence_level.tolist()
        self.history['liability_risk'] = self.liability_risk.tolist()
        self.history['class_action_viability'] = self.class_action_viability.tolist()
        self.history['shield_breach_probability'] = self.shield_breach_probability.tolist()
        self.history['events'] = self.events
        
        # Prepare results
        results = {
            'params': self.params.copy(),
            'timeline': [d.strftime('%Y-%m-%d') for d in self.timeline],
            'evidence_level': self.evidence_level.tolist(),
            'liability_risk': self.liability_risk.tolist(),
            'class_action_viability': self.class_action_viability.tolist(),
            'shield_breach_probability': self.shield_breach_probability.tolist(),
            'events': self.events,
            'final_state': {
                'evidence_level': float(self.evidence_level[-1]),
                'liability_risk': float(self.liability_risk[-1]),
                'class_action_viability': float(self.class_action_viability[-1]),
                'shield_breach_probability': float(self.shield_breach_probability[-1])
            }
        }
        
        return results
    
    def plot_timeline(self, save_path: Optional[str] = None):
        """
        Generate a timeline plot of legal risk factors.
        
        Args:
            save_path: Path to save the plot image
        """
        if len(self.timeline) < 2:
            logger.error("Not enough data for plotting. Run simulation first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Convert timeline to years for better x-axis
        years = [(d - self.timeline[0]).days / 365 for d in self.timeline]
        
        # Plot each metric
        plt.plot(years, self.evidence_level, 'b-', linewidth=2, label='Evidence Level')
        plt.plot(years, self.liability_risk, 'r-', linewidth=2, label='Liability Risk')
        plt.plot(years, self.class_action_viability, 'g-', linewidth=2, label='Class Action Viability')
        plt.plot(years, self.shield_breach_probability, 'm-', linewidth=2, label='Shield Breach Probability')
        
        # Add threshold lines
        plt.axhline(y=self.params['evidence_threshold'], color='b', linestyle='--', alpha=0.5, 
                   label='Evidence Threshold')
        plt.axhline(y=self.params['liability_threshold'], color='r', linestyle='--', alpha=0.5,
                   label='Liability Threshold')
        plt.axhline(y=self.params['class_action_threshold'], color='g', linestyle='--', alpha=0.5,
                   label='Class Action Threshold')
        
        # Add event markers
        for event in self.events:
            event_date = event['date']
            event_idx = np.argmin(np.abs(np.array(self.timeline) - event_date))
            event_year = years[event_idx]
            
            if event['type'] == 'regulatory':
                marker = 's'  # square
                color = 'cyan'
            elif event['type'] == 'scientific':
                marker = '^'  # triangle
                color = 'blue'
            elif event['type'] == 'legal':
                marker = 'o'  # circle
                color = 'red'
            elif event['type'] == 'evidence':
                marker = 'D'  # diamond
                color = 'green'
            elif event['type'] == 'media':
                marker = 'X'  # X
                color = 'orange'
            else:
                marker = '+'  # plus
                color = 'black'
            
            plt.plot(event_year, 0.05, marker, markersize=10, color=color, alpha=0.7)
            
            # Add annotation for major events
            if event['type'] in ['legal', 'evidence']:
                plt.annotate(
                    event['description'],
                    (event_year, 0.05),
                    xytext=(0, 20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                    rotation=45,
                    fontsize=8
                )
        
        plt.xlabel('Years from Simulation Start')
        plt.ylabel('Probability')
        plt.title('Legal & Policy Risk Timeline')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Add a legend for event markers
        from matplotlib.lines import Line2D
        event_legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markersize=10, label='Regulatory'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Scientific'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Legal'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=10, label='Evidence'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', markersize=10, label='Media')
        ]
        plt.legend(handles=event_legend_elements, loc='lower right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Timeline plot saved to {save_path}")
        
        plt.close()
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save simulation results to a JSON file.
        
        Args:
            results: Simulation results dictionary
            filepath: Path to save the results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


def run_sample_simulation():
    """Run a sample simulation with default parameters."""
    # Initialize simulator
    simulator = LegalPolicySimulator()
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Create output directory
    os.makedirs('output/legal', exist_ok=True)
    
    # Generate and save visualizations
    simulator.plot_timeline(save_path='output/legal/risk_timeline.png')
    
    # Save results
    simulator.save_results(results, 'output/legal/simulation_results.json')
    
    logger.info(f"Final liability risk: {results['final_state']['liability_risk']:.2f}")
    logger.info(f"Final shield breach probability: {results['final_state']['shield_breach_probability']:.2f}")
    
    return results


if __name__ == "__main__":
    # Run a sample simulation when executed directly
    run_sample_simulation()
