#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Institutional Feedback Model for GASLIT-AF WARSTACK

This module implements institutional feedback modeling to build dynamic denial-injury-denial loops,
regulatory capture graphs, and memetic immunosuppression nets.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Optional imports for network analysis
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("NetworkX not available. Graph analysis will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InstitutionalFeedbackModel:
    """
    Models institutional feedback loops and systemic denial mechanisms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the institutional feedback model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.params = {
            'institutions': ['CDC', 'FDA', 'Media', 'Academia', 'Industry'],
            'initial_legitimacy': 0.8,  # Initial legitimacy score (0-1)
            'denial_strength': 0.7,  # Strength of denial mechanisms (0-1)
            'capture_factor': 0.6,  # Regulatory capture factor (0-1)
            'memetic_spread_rate': 0.3,  # Rate of memetic spread (0-1)
            'time_steps': 100,  # Number of time steps for simulation
        }
        
        # Update with user-provided parameters
        self.params.update(self.config.get('params', {}))
        
        # Initialize state variables
        self.reset_state()
        
        # Check for required dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not HAS_NETWORKX:
            logger.warning("NetworkX is required for graph-based analysis")
    
    def reset_state(self):
        """Reset the simulation state."""
        self.time = 0
        self.iteration = 0
        
        # Initialize institutions
        self.institutions = self.params['institutions']
        n_inst = len(self.institutions)
        
        # Initialize state variables
        self.legitimacy = np.ones(n_inst) * self.params['initial_legitimacy']
        self.denial = np.zeros(n_inst)
        self.injury = np.zeros(n_inst)
        
        # Initialize network structure if NetworkX is available
        if HAS_NETWORKX:
            self.network = self._initialize_network()
        
        # History for tracking system states
        self.history = {
            'time': [],
            'legitimacy': [],
            'denial': [],
            'injury': [],
            'entropy': []
        }
    
    def _initialize_network(self) -> nx.DiGraph:
        """
        Initialize the institutional network structure.
        
        Returns:
            Directed graph representing institutional relationships
        """
        G = nx.DiGraph()
        
        # Add nodes (institutions)
        for i, inst in enumerate(self.institutions):
            G.add_node(inst, 
                       legitimacy=self.legitimacy[i],
                       denial=self.denial[i],
                       injury=self.injury[i])
        
        # Add edges (relationships between institutions)
        # This is a simplified model - real relationships would be more complex
        
        # CDC relationships
        G.add_edge('CDC', 'Media', weight=0.8, type='information')
        G.add_edge('CDC', 'Academia', weight=0.6, type='funding')
        G.add_edge('Industry', 'CDC', weight=0.7, type='influence')
        
        # FDA relationships
        G.add_edge('FDA', 'Media', weight=0.7, type='information')
        G.add_edge('Industry', 'FDA', weight=0.9, type='influence')
        
        # Media relationships
        G.add_edge('Media', 'Academia', weight=0.5, type='amplification')
        G.add_edge('Media', 'Industry', weight=0.3, type='scrutiny')
        
        # Academia relationships
        G.add_edge('Academia', 'CDC', weight=0.6, type='expertise')
        G.add_edge('Academia', 'FDA', weight=0.5, type='expertise')
        G.add_edge('Industry', 'Academia', weight=0.8, type='funding')
        
        return G
    
    def denial_injury_step(self, dt: float = 0.1):
        """
        Perform a single time step of the denial-injury-denial loop.
        
        Args:
            dt: Time step size
        """
        n_inst = len(self.institutions)
        
        # Parameters
        denial_strength = self.params['denial_strength']
        
        # Calculate injury based on current denial levels
        # Higher denial leads to more injury over time
        injury_rate = denial_strength * self.denial * (1 - self.injury)
        
        # Calculate denial response to injury
        # More injury leads to stronger denial
        denial_rate = self.injury * (1 - self.denial) - 0.1 * self.denial
        
        # Calculate legitimacy decay based on denial and injury
        # Both denial and injury reduce legitimacy
        legitimacy_rate = -0.1 * (self.denial + self.injury) * self.legitimacy
        
        # Update state variables
        self.injury += dt * injury_rate
        self.denial += dt * denial_rate
        self.legitimacy += dt * legitimacy_rate
        
        # Ensure values stay in valid range
        self.injury = np.clip(self.injury, 0, 1)
        self.denial = np.clip(self.denial, 0, 1)
        self.legitimacy = np.clip(self.legitimacy, 0, 1)
        
        # Update network if available
        if HAS_NETWORKX:
            for i, inst in enumerate(self.institutions):
                self.network.nodes[inst]['legitimacy'] = self.legitimacy[i]
                self.network.nodes[inst]['denial'] = self.denial[i]
                self.network.nodes[inst]['injury'] = self.injury[i]
        
        # Calculate system entropy (measure of disorder/legitimacy loss)
        entropy = -np.sum(self.legitimacy * np.log2(self.legitimacy + 1e-10))
        
        # Update time and iteration
        self.time += dt
        self.iteration += 1
        
        # Record history
        self.history['time'].append(self.time)
        self.history['legitimacy'].append(self.legitimacy.copy())
        self.history['denial'].append(self.denial.copy())
        self.history['injury'].append(self.injury.copy())
        self.history['entropy'].append(entropy)
    
    def regulatory_capture_step(self, dt: float = 0.1):
        """
        Perform a single time step of regulatory capture dynamics.
        
        Args:
            dt: Time step size
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX required for regulatory capture simulation")
            return
        
        # Parameters
        capture_factor = self.params['capture_factor']
        
        # Calculate influence propagation through the network
        # This is a simplified model of how industry influence affects regulatory bodies
        
        # Get industry influence edges
        industry_edges = [(u, v, d) for u, v, d in self.network.edges(data=True) 
                         if u == 'Industry' and d['type'] == 'influence']
        
        # Propagate influence
        for u, v, d in industry_edges:
            # Higher weight means more influence
            influence = d['weight'] * capture_factor
            
            # Increase denial proportional to influence
            v_idx = self.institutions.index(v)
            self.denial[v_idx] += dt * influence * (1 - self.denial[v_idx])
            
            # Decrease legitimacy proportional to influence
            self.legitimacy[v_idx] -= dt * influence * self.legitimacy[v_idx] * 0.2
        
        # Ensure values stay in valid range
        self.denial = np.clip(self.denial, 0, 1)
        self.legitimacy = np.clip(self.legitimacy, 0, 1)
        
        # Update network
        for i, inst in enumerate(self.institutions):
            self.network.nodes[inst]['legitimacy'] = self.legitimacy[i]
            self.network.nodes[inst]['denial'] = self.denial[i]
    
    def memetic_immunosuppression_step(self, dt: float = 0.1):
        """
        Perform a single time step of memetic immunosuppression dynamics.
        
        Args:
            dt: Time step size
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX required for memetic immunosuppression simulation")
            return
        
        # Parameters
        spread_rate = self.params['memetic_spread_rate']
        
        # Calculate memetic spread through the network
        # This models how narratives spread between institutions
        
        # Get information/amplification edges
        info_edges = [(u, v, d) for u, v, d in self.network.edges(data=True) 
                     if d['type'] in ['information', 'amplification']]
        
        # Temporary array to store memetic effects
        memetic_effect = np.zeros(len(self.institutions))
        
        # Propagate memetic effects
        for u, v, d in info_edges:
            u_idx = self.institutions.index(u)
            v_idx = self.institutions.index(v)
            
            # Higher denial in source increases memetic spread
            source_effect = self.denial[u_idx] * d['weight'] * spread_rate
            
            # Add to target's memetic effect
            memetic_effect[v_idx] += source_effect
        
        # Apply memetic effects
        for i in range(len(self.institutions)):
            # Memetic effects increase denial and reduce legitimacy
            self.denial[i] += dt * memetic_effect[i] * (1 - self.denial[i])
            self.legitimacy[i] -= dt * memetic_effect[i] * self.legitimacy[i] * 0.1
        
        # Ensure values stay in valid range
        self.denial = np.clip(self.denial, 0, 1)
        self.legitimacy = np.clip(self.legitimacy, 0, 1)
        
        # Update network
        if HAS_NETWORKX:
            for i, inst in enumerate(self.institutions):
                self.network.nodes[inst]['legitimacy'] = self.legitimacy[i]
                self.network.nodes[inst]['denial'] = self.denial[i]
    
    def simulation_step(self, dt: float = 0.1):
        """
        Perform a complete simulation step combining all dynamics.
        
        Args:
            dt: Time step size
        """
        # Run individual dynamics steps
        self.denial_injury_step(dt)
        self.regulatory_capture_step(dt)
        self.memetic_immunosuppression_step(dt)
    
    def run_simulation(self, time_steps: Optional[int] = None, dt: float = 0.1) -> Dict[str, Any]:
        """
        Run the full simulation for the specified number of time steps.
        
        Args:
            time_steps: Number of time steps to simulate
            dt: Time step size
            
        Returns:
            Dictionary with simulation results
        """
        if time_steps is None:
            time_steps = self.params['time_steps']
        
        logger.info(f"Starting institutional feedback simulation with {time_steps} time steps")
        
        # Reset state before simulation
        self.reset_state()
        
        # Run simulation
        for _ in range(time_steps):
            self.simulation_step(dt)
            
            # Log progress every 10% of steps
            if time_steps > 10 and _ % (time_steps // 10) == 0:
                logger.info(f"Simulation progress: {_ / time_steps * 100:.1f}%")
        
        logger.info("Simulation completed")
        
        # Prepare results
        results = {
            'params': self.params.copy(),
            'final_state': {
                'time': self.time,
                'legitimacy': self.legitimacy.tolist(),
                'denial': self.denial.tolist(),
                'injury': self.injury.tolist(),
                'entropy': self.history['entropy'][-1]
            },
            'history': {
                'time': self.history['time'],
                'legitimacy': [l.tolist() for l in self.history['legitimacy']],
                'denial': [d.tolist() for d in self.history['denial']],
                'injury': [i.tolist() for i in self.history['injury']],
                'entropy': self.history['entropy']
            },
            'institutions': self.institutions
        }
        
        return results
    
    def plot_legitimacy_entropy(self, save_path: Optional[str] = None):
        """
        Generate a plot of system legitimacy entropy over time.
        
        Args:
            save_path: Path to save the plot image
        """
        if len(self.history['time']) < 2:
            logger.error("Not enough data for plotting. Run simulation first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot entropy over time
        plt.plot(self.history['time'], self.history['entropy'], 
                 'r-', linewidth=2, label='System Legitimacy Entropy')
        
        # Add threshold line for critical entropy
        critical_entropy = 2.0  # Placeholder value
        plt.axhline(y=critical_entropy, color='k', linestyle='--', 
                   label='Critical Entropy Threshold')
        
        # Add annotations for key events
        # In a real implementation, these would be determined by the simulation
        key_points = [
            (20, 'Initial Denial Phase'),
            (50, 'Regulatory Capture Acceleration'),
            (80, 'Narrative Collapse Threshold')
        ]
        
        for t, label in key_points:
            if t < len(self.history['time']):
                plt.annotate(
                    label,
                    (self.history['time'][t], self.history['entropy'][t]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
                )
        
        plt.xlabel('Time')
        plt.ylabel('Legitimacy Entropy')
        plt.title('System Legitimacy Entropy Index')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entropy plot saved to {save_path}")
        
        plt.close()
    
    def plot_denial_recursion_map(self, save_path: Optional[str] = None):
        """
        Generate a denial recursion map visualization.
        
        Args:
            save_path: Path to save the plot image
        """
        if not HAS_NETWORKX or len(self.history['time']) < 2:
            logger.error("NetworkX required for denial recursion map or not enough data.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Get final state
        pos = nx.spring_layout(self.network, seed=42)
        
        # Node sizes based on legitimacy (inverse)
        node_sizes = [1000 * (1 - self.network.nodes[n]['legitimacy']) for n in self.network.nodes()]
        
        # Node colors based on denial level
        node_colors = [self.network.nodes[n]['denial'] for n in self.network.nodes()]
        
        # Edge widths based on weight
        edge_widths = [d['weight'] * 2 for u, v, d in self.network.edges(data=True)]
        
        # Draw the network
        nx.draw_networkx_nodes(self.network, pos, 
                              node_size=node_sizes,
                              node_color=node_colors, 
                              cmap=plt.cm.Reds,
                              alpha=0.8)
        
        nx.draw_networkx_edges(self.network, pos, 
                              width=edge_widths,
                              alpha=0.6, 
                              edge_color='gray',
                              arrows=True, 
                              arrowsize=15)
        
        nx.draw_networkx_labels(self.network, pos, font_size=12, font_weight='bold')
        
        # Add edge labels
        edge_labels = {(u, v): d['type'] for u, v, d in self.network.edges(data=True)}
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=edge_labels, font_size=8)
        
        # Add colorbar for denial levels
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Denial Level')
        
        plt.title('Institutional Denial Recursion Map')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Denial recursion map saved to {save_path}")
        
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
    # Initialize model
    model = InstitutionalFeedbackModel()
    
    # Run simulation
    results = model.run_simulation(time_steps=100)
    
    # Create output directory
    os.makedirs('output/institutional', exist_ok=True)
    
    # Generate and save visualizations
    model.plot_legitimacy_entropy(save_path='output/institutional/entropy_index.png')
    model.plot_denial_recursion_map(save_path='output/institutional/denial_recursion_map.png')
    
    # Save results
    model.save_results(results, 'output/institutional/simulation_results.json')
    
    return results


if __name__ == "__main__":
    # Run a sample simulation when executed directly
    run_sample_simulation()
