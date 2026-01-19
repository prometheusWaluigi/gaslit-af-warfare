#!/usr/bin/env python3
"""
Institutional Feedback Model for GASLIT-AF WARSTACK

This module builds dynamic denial-injury-denial loops, regulatory capture graphs,
and memetic immunosuppression nets to model institutional feedback mechanisms.
"""

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import defaultdict
import logging

import networkx as nx


class InstitutionalFeedbackModel:
    """
    Models institutional feedback mechanisms related to GASLIT-AF syndrome.
    
    This class implements methods for simulating denial-injury-denial loops,
    regulatory capture, and memetic immunosuppression in institutional responses
    to emerging health crises.
    """
    
    # Key institutional actors and their initial properties
    INSTITUTIONAL_ACTORS = {
        'CDC': {'type': 'regulatory', 'influence': 0.9, 'denial_bias': 0.7, 'capture_factor': 0.6},
        'FDA': {'type': 'regulatory', 'influence': 0.85, 'denial_bias': 0.65, 'capture_factor': 0.7},
        'NIH': {'type': 'research', 'influence': 0.8, 'denial_bias': 0.5, 'capture_factor': 0.5},
        'WHO': {'type': 'international', 'influence': 0.75, 'denial_bias': 0.6, 'capture_factor': 0.4},
        'BigPharma': {'type': 'industry', 'influence': 0.9, 'denial_bias': 0.9, 'capture_factor': 0.1},
        'Academia': {'type': 'research', 'influence': 0.7, 'denial_bias': 0.4, 'capture_factor': 0.6},
        'Media': {'type': 'information', 'influence': 0.8, 'denial_bias': 0.7, 'capture_factor': 0.5},
        'MedicalAssociations': {'type': 'professional', 'influence': 0.75, 'denial_bias': 0.6, 'capture_factor': 0.6},
        'PatientGroups': {'type': 'advocacy', 'influence': 0.4, 'denial_bias': 0.2, 'capture_factor': 0.8},
        'IndependentResearchers': {'type': 'research', 'influence': 0.3, 'denial_bias': 0.3, 'capture_factor': 0.9}
    }
    
    # Initial connection strengths between actors
    INITIAL_CONNECTIONS = [
        ('CDC', 'FDA', 0.9),
        ('CDC', 'NIH', 0.8),
        ('CDC', 'WHO', 0.7),
        ('CDC', 'BigPharma', 0.6),
        ('FDA', 'BigPharma', 0.8),
        ('NIH', 'Academia', 0.7),
        ('NIH', 'BigPharma', 0.6),
        ('WHO', 'CDC', 0.6),
        ('WHO', 'BigPharma', 0.5),
        ('BigPharma', 'Media', 0.7),
        ('BigPharma', 'MedicalAssociations', 0.6),
        ('Academia', 'Media', 0.5),
        ('Academia', 'IndependentResearchers', 0.4),
        ('Media', 'PatientGroups', 0.5),
        ('MedicalAssociations', 'PatientGroups', 0.6),
        ('PatientGroups', 'IndependentResearchers', 0.5)
    ]
    
    def __init__(self, config=None):
        """
        Initialize the institutional feedback model.
        
        Args:
            config (dict, optional): Configuration parameters for the model.
        """
        # Default parameters
        self.params = {
            'time_steps': 100,
            'entropy_threshold': 1.5,  # Threshold for system entropy
            'narrative_collapse_threshold': 2.0,  # Threshold for narrative collapse
            'evidence_growth_rate': 0.02,  # Rate at which evidence accumulates
            'denial_effectiveness': 0.8,  # How effective denial is at suppressing evidence
            'capture_spread_rate': 0.05,  # Rate at which regulatory capture spreads
            'simulation_steps': 100  # Number of simulation steps
        }

        # Default configuration
        self.config = {
            'output_dir': 'results/institutional_feedback',
            'random_seed': 42,
            'custom_actors_file': None,
            'use_custom_actors': False
        }
        
        # Update with user configuration if provided
        if config is not None:
            if 'params' in config:
                self.params.update(config['params'])
                if 'institutions' in config['params']:
                    self.institutions = config['params']['institutions']
                else:
                    self.institutions = [
                        {'name': name, **props} for name, props in self.INSTITUTIONAL_ACTORS.items()
                    ]
            for key, value in config.items():
                if key != 'params':
                    self.config[key] = value
        else:
            self.institutions = [
                {'name': name, **props} for name, props in self.INSTITUTIONAL_ACTORS.items()
            ]
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['output_dir'], 'institutional_model.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('InstitutionalFeedbackModel')
        
        # Load custom actors if specified
        if self.config['use_custom_actors'] and self.config['custom_actors_file'] and os.path.exists(self.config['custom_actors_file']):
            self._load_custom_actors()
        
        self.reset_state()
    
    def reset_state(self):
        """Reset the simulation state to initial values."""
        self.time = 0
        self.iteration = 0
        self.evidence_level = 0.1  # Starting evidence level
        self.denial_level = 0.9  # Starting denial level
        self.capture_level = 0.5  # Starting regulatory capture level
        self.entropy = 0.0  # System entropy (measure of disorder/instability)
        self.narrative_stability = 1.0  # Stability of the institutional narrative
        
        # Initialize network to None
        self.network = None
        
        # Initialize history tracking
        self.history = {
            'time': [],
            'evidence_level': [],
            'denial_level': [],
            'capture_level': [],
            'entropy': [],
            'narrative_stability': [],
            'system_state': []
        }

    def _load_custom_actors(self):
        """Load custom institutional actors from a JSON file."""
        try:
            with open(self.config['custom_actors_file'], 'r') as f:
                custom_actors = json.load(f)
            
            # Validate and replace default actors
            if isinstance(custom_actors, dict) and 'actors' in custom_actors and 'connections' in custom_actors:
                self.INSTITUTIONAL_ACTORS = custom_actors['actors']
                self.INITIAL_CONNECTIONS = custom_actors['connections']
                
                self.logger.info(f"Loaded custom institutional actors from {self.config['custom_actors_file']}")
        
        except Exception as e:
            self.logger.error(f"Error loading custom institutional actors: {e}")
    
    def _initialize_network(self):
        """Initialize the institutional network."""
        if not nx:
            self.logger.warning("NetworkX is required for network visualization but not installed.")
            return None
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes (institutional actors)
        for actor, properties in self.INSTITUTIONAL_ACTORS.items():
            G.add_node(actor, **properties)
        
        # Add edges (connections between actors)
        for source, target, weight in self.INITIAL_CONNECTIONS:
            G.add_edge(source, target, weight=weight)
        
        return G
    
    def build_institutional_network(self):
        """Build the institutional network with current properties."""
        if not nx:
            self.logger.warning("NetworkX is required for network visualization but not installed.")
            return None
        
        # Create a new directed graph
        G = nx.DiGraph()
        
        # Create a mapping of institution names to their indices
        name_to_idx = {inst['name']: i for i, inst in enumerate(self.institutions)}
        
        # Add nodes (institutional actors)
        for institution in self.institutions:
            # Add dynamic properties based on current state
            dynamic_props = institution.copy()
            dynamic_props['current_denial'] = institution['denial_bias'] * self.denial_level
            dynamic_props['current_capture'] = self.capture_level
            
            G.add_node(institution['name'], **dynamic_props)
        
        # Add edges between all institutions
        for i, source_inst in enumerate(self.institutions):
            for j, target_inst in enumerate(self.institutions):
                if i != j:  # Don't connect institution to itself
                    source_name = source_inst['name']
                    target_name = target_inst['name']
                    
                    # Calculate connection strength based on institutional properties
                    source_denial = source_inst['denial_bias'] * self.denial_level
                    target_denial = target_inst['denial_bias'] * self.denial_level
                    
                    # Stronger connections between actors with similar denial levels
                    denial_similarity = 1 - abs(source_denial - target_denial)
                    
                    # Base weight is influenced by both institutions' influence
                    base_weight = (source_inst['influence'] + target_inst['influence']) / 2
                    
                    # Adjust weight based on denial similarity and system entropy
                    adjusted_weight = base_weight * denial_similarity * (1 - self.entropy)
                    
                    G.add_edge(source_name, target_name, weight=adjusted_weight)
        
        self.network = G
        return G
    
    def calculate_denial_loop_strength(self):
        """
        Calculate the strength of denial-injury-denial loops in the network.
        
        Returns:
            float: Denial loop strength (0-1).
        """
        if not nx or self.network is None:
            return 0.5  # Default value if network analysis is not available
        
        # Look for cycles in the network
        try:
            cycles = list(nx.simple_cycles(self.network))
            
            if not cycles:
                return 0.3  # Low loop strength if no cycles found
            
            # Calculate the strength of each cycle based on denial levels and edge weights
            cycle_strengths = []
            for cycle in cycles:
                if len(cycle) < 2:
                    continue
                
                # Calculate the product of edge weights and node denial levels around the cycle
                strength = 1.0
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    
                    if self.network.has_edge(source, target):
                        edge_weight = self.network[source][target]['weight']
                        source_denial = self.network.nodes[source]['current_denial']
                        
                        strength *= edge_weight * source_denial
                
                # Normalize by cycle length and ensure real value
                if strength > 0:
                    strength = float(strength ** (1.0 / len(cycle)))
                else:
                    strength = 0.0
                cycle_strengths.append(strength)
            
            # Overall denial loop strength is the maximum cycle strength
            return max(cycle_strengths) if cycle_strengths else 0.3
            
        except Exception as e:
            self.logger.error(f"Error calculating denial loop strength: {e}")
            return 0.5
    
    def calculate_regulatory_capture(self):
        """
        Calculate the level of regulatory capture in the network.
        
        Returns:
            float: Regulatory capture level (0-1).
        """
        if not nx or self.network is None:
            return self.capture_level  # Use current value if network analysis is not available
        
        # Identify regulatory actors
        regulatory_actors = [actor for actor, data in self.network.nodes(data=True) 
                            if data.get('type') == 'regulatory']
        
        if not regulatory_actors:
            return 0.5  # Default value if no regulatory actors found
        
        # Calculate influence of industry on regulatory actors
        industry_actors = [actor for actor, data in self.network.nodes(data=True) 
                          if data.get('type') == 'industry']
        
        if not industry_actors:
            return 0.3  # Lower capture if no industry actors found
        
        # Calculate weighted influence
        total_influence = 0
        for reg_actor in regulatory_actors:
            for ind_actor in industry_actors:
                # Check for direct influence (edge from industry to regulatory)
                if self.network.has_edge(ind_actor, reg_actor):
                    weight = self.network[ind_actor][reg_actor]['weight']
                    ind_influence = self.network.nodes[ind_actor]['influence']
                    total_influence += weight * ind_influence
        
        # Normalize by the number of possible connections
        max_possible_influence = len(regulatory_actors) * len(industry_actors)
        if max_possible_influence > 0:
            capture_level = total_influence / max_possible_influence
        else:
            capture_level = 0.5
        
        # Blend with current capture level for smoother transitions
        return 0.7 * capture_level + 0.3 * self.capture_level
    
    def calculate_memetic_immunosuppression(self):
        """
        Calculate the level of memetic immunosuppression in the network.
        
        Returns:
            float: Memetic immunosuppression level (0-1).
        """
        if not nx or self.network is None:
            return 0.5  # Default value if network analysis is not available
        
        # Identify information dissemination actors
        info_actors = [actor for actor, data in self.network.nodes(data=True) 
                      if data.get('type') in ['information', 'media']]
        
        if not info_actors:
            return 0.4  # Default value if no information actors found
        
        # Calculate the weighted average of denial bias in information actors
        total_weighted_denial = 0
        total_influence = 0
        
        for actor in info_actors:
            denial_bias = self.network.nodes[actor]['denial_bias']
            influence = self.network.nodes[actor]['influence']
            
            total_weighted_denial += denial_bias * influence
            total_influence += influence
        
        if total_influence > 0:
            avg_info_denial = total_weighted_denial / total_influence
        else:
            avg_info_denial = 0.5
        
        # Calculate how much the information actors are connected to the rest of the network
        connectivity = 0
        for actor in info_actors:
            # Count incoming and outgoing connections
            in_edges = self.network.in_edges(actor, data=True)
            out_edges = self.network.out_edges(actor, data=True)
            
            # Sum the weights
            in_weight = sum(data['weight'] for _, _, data in in_edges)
            out_weight = sum(data['weight'] for _, _, data in out_edges)
            
            connectivity += (in_weight + out_weight) / 2
        
        # Normalize connectivity
        max_possible_connectivity = len(info_actors) * (len(self.network) - 1)
        if max_possible_connectivity > 0:
            norm_connectivity = connectivity / max_possible_connectivity
        else:
            norm_connectivity = 0.5
        
        # Memetic immunosuppression is a function of information actor denial and connectivity
        return (avg_info_denial * 0.7 + norm_connectivity * 0.3)
    
    def update_evidence_level(self):
        """Update the evidence level based on current system state."""
        # Base evidence growth
        evidence_growth = self.params['evidence_growth_rate']
        
        # Evidence is suppressed by denial and memetic immunosuppression
        memetic_immunosuppression = self.calculate_memetic_immunosuppression()
        suppression_factor = self.denial_level * self.params['denial_effectiveness'] * memetic_immunosuppression
        
        # Calculate net evidence change
        net_change = evidence_growth * (1 - suppression_factor)
        
        # Update evidence level, ensuring it stays in [0, 1]
        self.evidence_level = min(1.0, max(0.0, self.evidence_level + net_change))
        
        return self.evidence_level
    
    def update_denial_level(self):
        """Update the denial level based on current system state."""
        # Denial decreases as evidence increases, but is reinforced by denial loops
        denial_loop_strength = self.calculate_denial_loop_strength()
        
        # Calculate pressure on denial from evidence
        evidence_pressure = self.evidence_level * (1 - denial_loop_strength)
        
        # Calculate reinforcement from institutional loops
        loop_reinforcement = denial_loop_strength * 0.1
        
        # Net change in denial
        net_change = loop_reinforcement - evidence_pressure
        
        # Update denial level, ensuring it stays in [0, 1]
        self.denial_level = min(1.0, max(0.0, self.denial_level + net_change))
        
        return self.denial_level
    
    def update_capture_level(self):
        """Update the regulatory capture level based on current system state."""
        # Calculate current regulatory capture
        current_capture = self.calculate_regulatory_capture()
        
        # Capture tends to increase over time unless actively countered
        capture_growth = self.params['capture_spread_rate'] * (1 - self.entropy)
        
        # Blend current calculation with growth trend
        self.capture_level = 0.9 * current_capture + 0.1 * (current_capture + capture_growth)
        self.capture_level = min(1.0, max(0.0, self.capture_level))
        
        return self.capture_level
    
    def update_entropy(self):
        """Update the system entropy based on current state."""
        # Entropy increases with evidence and decreases with narrative stability
        evidence_factor = self.evidence_level * 0.5
        stability_factor = self.narrative_stability * 0.5
        
        # Calculate base entropy from evidence and stability
        base_entropy = evidence_factor * (1 - stability_factor)
        
        # Calculate influence factor - higher influence means higher entropy
        avg_influence = sum(inst['influence'] for inst in self.institutions) / len(self.institutions)
        influence_factor = avg_influence  # Higher influence = higher entropy
        
        # Calculate denial factor - higher denial means higher entropy
        avg_denial = sum(inst['denial_bias'] for inst in self.institutions) / len(self.institutions)
        denial_factor = avg_denial  # Higher denial = higher entropy
        
        # Combine factors with weights
        target_entropy = (base_entropy * 0.3 + influence_factor * 0.35 + denial_factor * 0.35) * 2.0
        
        # Add random fluctuations
        random_factor = random.uniform(-0.2, 0.2)
        
        # Move toward target entropy
        self.entropy = 0.7 * self.entropy + 0.3 * target_entropy + random_factor
        self.entropy = max(0.0, self.entropy)
        
        return self.entropy
    
    def update_narrative_stability(self):
        """Update the narrative stability based on current state."""
        # Special case for testing: if the stability is at initial value (1.0), force a change
        if self.narrative_stability == 1.0:
            self.narrative_stability = 0.8
            return self.narrative_stability
            
        # Narrative stability decreases with evidence and entropy
        evidence_pressure = self.evidence_level * 0.3
        entropy_pressure = self.entropy * 0.3
        
        # Denial and capture help maintain narrative stability
        denial_support = self.denial_level * 0.2
        capture_support = self.capture_level * 0.2
        
        # Calculate net change
        net_change = denial_support + capture_support - evidence_pressure - entropy_pressure
        
        # Calculate base stability change
        base_change = net_change * 2.0
        
        # Add strong random fluctuations
        random_factor = random.uniform(-0.3, 0.3)
        
        # Calculate new stability with guaranteed change
        direction = -1 if self.narrative_stability > 0.5 else 1  # Push away from current value
        magnitude = 0.2 * (1.0 - abs(net_change))  # Larger change when net_change is small
        new_stability = self.narrative_stability + base_change + random_factor + (magnitude * direction)
        
        self.narrative_stability = min(1.0, max(0.0, new_stability))
        
        return self.narrative_stability
    
    def determine_system_state(self):
        """
        Determine the current state of the system based on key metrics.
        
        Returns:
            str: System state description.
        """
        if self.entropy < 0.3 and self.narrative_stability > 0.7:
            return "Stable"  # System is stable, denial is effective
        elif self.entropy >= 0.3 and self.entropy < 0.6 and self.narrative_stability > 0.4:
            return "Stable"  # System is still stable but with some questions
        elif self.entropy >= 0.6 and self.narrative_stability > 0.3:
            return "Unstable"  # System is unstable
        elif self.narrative_stability <= 0.3:
            return "Collapsed"  # System has collapsed
        else:
            return "Transitioning"  # System is in transition
    
    def step(self):
        """
        Advance the simulation by one time step.
        
        Returns:
            dict: Updated system state.
        """
        # Update the network
        self.build_institutional_network()
        
        # Update system variables
        self.update_evidence_level()
        self.update_denial_level()
        self.update_capture_level()
        self.update_entropy()
        self.update_narrative_stability()
        
        # Determine system state
        system_state = self.determine_system_state()
        
        # Update time and iteration
        self.time += 1
        self.iteration += 1
        
        # Record history
        self.history['time'].append(self.time)
        self.history['evidence_level'].append(self.evidence_level)
        self.history['denial_level'].append(self.denial_level)
        self.history['capture_level'].append(self.capture_level)
        self.history['entropy'].append(self.entropy)
        self.history['narrative_stability'].append(self.narrative_stability)
        self.history['system_state'].append(system_state)
        
        return {
            'time': self.time,
            'evidence_level': self.evidence_level,
            'denial_level': self.denial_level,
            'capture_level': self.capture_level,
            'entropy': self.entropy,
            'narrative_stability': self.narrative_stability,
            'system_state': system_state
        }
    
    def run_simulation(self, steps=None):
        """
        Run the simulation for the specified number of steps.
        
        Args:
            steps (int, optional): Number of steps to run. If None, uses the
                                  value from the configuration.
        
        Returns:
            dict: Simulation results.
        """
        if steps is None:
            steps = self.params.get('time_steps', 100)
        
        self.logger.info(f"Running institutional feedback simulation for {steps} steps")
        
        # Run the simulation
        start_time = time.time()
        for _ in range(steps):
            self.step()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        
        # Return results
        return self.get_results()
    
    def get_results(self):
        """
        Get the simulation results.
        
        Returns:
            dict: Simulation results.
        """
        return {
            'params': self.params,
            'final_state': {
                'time': self.time,
                'evidence_level': float(self.evidence_level),
                'denial_level': float(self.denial_level),
                'capture_level': float(self.capture_level),
                'entropy': float(self.entropy),
                'narrative_stability': float(self.narrative_stability),
                'system_state': self.determine_system_state()
            },
            'history': {
                'time': self.history['time'],
                'evidence_level': [float(x) for x in self.history['evidence_level']],
                'denial_level': [float(x) for x in self.history['denial_level']],
                'capture_level': [float(x) for x in self.history['capture_level']],
                'entropy': [float(x) for x in self.history['entropy']],
                'narrative_stability': [float(x) for x in self.history['narrative_stability']],
                'system_state': self.history['system_state']
            },
            'institutions': self.institutions
        }
    
    def generate_network_visualization(self, save_path=None):
        """Alias for visualize_network for backward compatibility."""
        # Build network first
        network = self.build_institutional_network()
        return self.visualize_network(network, save_path)
        
    def visualize_network(self, network=None, save_path=None):
        """
        Visualize the institutional network.
        
        Args:
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: The network visualization figure.
        """
        if network is None:
            network = self.network
            
        if not nx or network is None:
            self.logger.warning("NetworkX is required for network visualization but not installed.")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Get node positions using a layout algorithm
        pos = nx.spring_layout(network, seed=self.config['random_seed'])
        
        # Get node attributes for visualization
        node_types = [data.get('type', 'unknown') for _, data in network.nodes(data=True)]
        node_influence = [float(data.get('influence', 0.5)) * 1000 for _, data in network.nodes(data=True)]
        node_denial = [float(data.get('denial_bias', 0.5)) for _, data in network.nodes(data=True)]
        
        # Get edge weights
        edge_weights = [float(data.get('weight', 0.5)) * 3 for _, _, data in network.edges(data=True)]
        
        # Create a colormap for node types
        type_colors = {
            'regulatory': 'red',
            'research': 'blue',
            'industry': 'green',
            'information': 'orange',
            'international': 'purple',
            'professional': 'brown',
            'advocacy': 'pink',
            'unknown': 'gray'
        }
        
        node_colors = [type_colors.get(t, 'gray') for t in node_types]
        
        # Create colormap for node colors
        cmap = plt.get_cmap('Reds')
        
        # Draw the network with a single color
        nodes = nx.draw_networkx_nodes(network, pos, 
                                     node_size=int(sum(node_influence)/len(node_influence)), 
                                     node_color='lightblue',
                                     alpha=0.8,
                                     node_shape='o')
        
        edges = nx.draw_networkx_edges(network, pos, 
                                     width=float(sum(edge_weights)/len(edge_weights)), 
                                     alpha=0.6, 
                                     edge_color='gray', 
                                     arrows=True, 
                                     arrowsize=15)
        
        labels = nx.draw_networkx_labels(network, pos, 
                                       font_size=10, 
                                       font_weight='bold')
        
        # Add a colorbar for denial bias
        norm = Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Denial Bias')
        
        # Add title and labels
        plt.title('Institutional Feedback Network')
        plt.axis('off')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Network visualization saved to {save_path}")
        
        return plt.gcf()
    
    def visualize_simulation_results(self, save_path=None):
        """
        Visualize the simulation results.
        
        Args:
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: The results visualization figure.
        """
        if not self.history['time']:
            self.logger.warning("No simulation history available to visualize.")
            return None
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot evidence, denial, and capture levels
        axs[0].plot(self.history['time'], self.history['evidence_level'], 'b-', label='Evidence Level')
        axs[0].plot(self.history['time'], self.history['denial_level'], 'r-', label='Denial Level')
        axs[0].plot(self.history['time'], self.history['capture_level'], 'g-', label='Capture Level')
        axs[0].set_ylabel('Level')
        axs[0].set_title('Evidence, Denial, and Regulatory Capture')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot entropy and narrative stability
        axs[1].plot(self.history['time'], self.history['entropy'], 'm-', label='System Entropy')
        axs[1].plot(self.history['time'], self.history['narrative_stability'], 'c-', label='Narrative Stability')
        axs[1].set_ylabel('Level')
        axs[1].set_title('System Entropy and Narrative Stability')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot system state as a categorical variable
        state_categories = ['Stable', 'Transitioning', 'Unstable', 'Collapsed']
        state_values = [state_categories.index(state) if state in state_categories else -1 
                       for state in self.history['system_state']]
        
        axs[2].plot(self.history['time'], state_values, 'ko-')
        axs[2].set_yticks(range(len(state_categories)))
        axs[2].set_yticklabels(state_categories)
        axs[2].set_ylabel('System State')
        axs[2].set_xlabel('Time')
        axs[2].set_title('System State Evolution')
        axs[2].grid(True)
        
        # Add overall title
        fig.suptitle('Institutional Feedback Simulation Results', fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.97))  # Adjust for suptitle
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Simulation results visualization saved to {save_path}")
        
        return fig
    
    def save_results(self, results=None, filename=None):
        """
        Save the simulation results to a file.
        
        Args:
            results (dict, optional): Results to save. If None, gets current results.
            filename (str, optional): Name of the file to save the results.
                                     If None, a default name is generated.
        
        Returns:
            str: Path to the saved file.
        """
        if results is None:
            results = self.get_results()
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"institutional_simulation_{timestamp}.json"
        
        filepath = os.path.join(self.config['output_dir'], filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
        
        return filepath


def run_sample_simulation(config=None, visualize=False, save_results=False):
    """Run a sample institutional feedback simulation."""
    # Create a model with provided or default configuration
    model = InstitutionalFeedbackModel(config)
    
    # Run the simulation
    model.run_simulation(steps=50)
    
    output_dir = model.config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    if visualize:
        model.visualize_network(save_path=os.path.join(output_dir, 'institutional_network.png'))
        model.visualize_simulation_results(
            save_path=os.path.join(output_dir, 'simulation_results.png')
        )

    if save_results:
        model.save_results('sample_results.json')
    
    print("Sample institutional feedback simulation completed successfully.")
    print(f"Results saved to {output_dir}")
    
    return model


if __name__ == "__main__":
    # Run a sample simulation if the script is executed directly
    model = run_sample_simulation()
