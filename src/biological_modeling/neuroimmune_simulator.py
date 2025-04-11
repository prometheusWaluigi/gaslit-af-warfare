#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuroimmune Simulator Module for GASLIT-AF WARSTACK

This module implements simulations of neuroimmune dynamics using KPZ/fKPZχ models,
ODE/PDE attractor maps, and phase portraits to model the core GASLIT-AF attractor states:
spike protein neurotoxicity → cerebellar trauma → behavioral and autonomic collapse.

The module leverages Intel OneAPI and SYCL for hardware acceleration on Intel Arc GPUs.
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
import json

# Optional imports for hardware acceleration
try:
    import dpctl
    import dpctl.tensor as dpt
    HAS_ONEAPI = True
except ImportError:
    HAS_ONEAPI = False
    logging.warning("Intel OneAPI not available. Using CPU-only mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuroimmuneDynamics:
    """
    Simulates neuroimmune dynamics using various mathematical models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the neuroimmune dynamics simulator.
        
        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.config = config or {}
        self.device = self._initialize_device()
        
        # Default parameters
        self.params = {
            'spike_toxicity': 0.75,  # Normalized toxicity factor (0-1)
            'cerebellar_vulnerability': 0.65,  # Vulnerability factor (0-1)
            'autonomic_resilience': 0.3,  # Resilience factor (0-1)
            'time_steps': 1000,  # Number of time steps for simulation
            'spatial_resolution': 100,  # Spatial grid resolution
            'diffusion_coefficient': 0.1,  # Diffusion coefficient
            'noise_amplitude': 0.05,  # Noise amplitude for KPZ
            'lambda_nonlinearity': 1.0,  # λ parameter for KPZ nonlinearity
            'nu_viscosity': 0.1,  # ν parameter for KPZ viscosity
        }
        
        # Update with user-provided parameters
        self.params.update(self.config.get('params', {}))
        
        # Initialize state variables
        self.reset_state()
    
    def _initialize_device(self) -> Optional[Any]:
        """
        Initialize the compute device (CPU or GPU via OneAPI).
        
        Returns:
            Device object or None if using CPU
        """
        if not HAS_ONEAPI:
            return None
            
        try:
            # Try to get a GPU device
            gpu_devices = [d for d in dpctl.get_devices() if d.is_gpu]
            if gpu_devices:
                logger.info(f"Using GPU device: {gpu_devices[0].name}")
                return gpu_devices[0]
            else:
                logger.info("No GPU devices found, falling back to CPU")
                return dpctl.select_default_device()
        except Exception as e:
            logger.error(f"Error initializing OneAPI device: {e}")
            return None
    
    def reset_state(self):
        """Reset the simulation state."""
        self.time = 0
        self.iteration = 0
        
        # Initialize spatial grid
        n = self.params['spatial_resolution']
        self.x = np.linspace(0, 1, n)
        self.y = np.linspace(0, 1, n)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize field variables
        self.h = np.zeros((n, n))  # Height field for KPZ
        self.immune_activation = np.zeros((n, n))  # Immune activation field
        self.neurotoxicity = np.zeros((n, n))  # Neurotoxicity field
        
        # Initialize with small random perturbations
        self.h += np.random.normal(0, 0.01, (n, n))
        self.immune_activation += np.random.normal(0, 0.01, (n, n))
        self.neurotoxicity += np.random.normal(0, 0.01, (n, n))
        
        # History for tracking attractor states
        self.history = {
            'time': [],
            'mean_h': [],
            'mean_immune': [],
            'mean_neurotox': [],
            'attractor_state': []
        }
    
    def kpz_step(self, dt: float = 0.01) -> np.ndarray:
        """
        Perform a single time step of the KPZ equation.
        
        The KPZ equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η
        
        Args:
            dt: Time step size
            
        Returns:
            Updated height field
        """
        n = self.params['spatial_resolution']
        ν = self.params['nu_viscosity']
        λ = self.params['lambda_nonlinearity']
        D = self.params['noise_amplitude']
        
        # Compute spatial derivatives using finite differences
        # Central difference for Laplacian (∇²h)
        laplacian = np.zeros_like(self.h)
        for i in range(1, n-1):
            for j in range(1, n-1):
                laplacian[i, j] = (self.h[i+1, j] + self.h[i-1, j] + 
                                  self.h[i, j+1] + self.h[i, j-1] - 4*self.h[i, j])
        
        # Forward difference for gradient squared ((∇h)²)
        grad_squared = np.zeros_like(self.h)
        for i in range(n-1):
            for j in range(n-1):
                dx = self.h[i+1, j] - self.h[i, j]
                dy = self.h[i, j+1] - self.h[i, j]
                grad_squared[i, j] = dx**2 + dy**2
        
        # Random noise term (η)
        noise = np.random.normal(0, np.sqrt(dt), (n, n))
        
        # Update height field using KPZ equation
        self.h += dt * (ν * laplacian + (λ/2) * grad_squared) + np.sqrt(2*D*dt) * noise
        
        return self.h
    
    def coupled_dynamics_step(self, dt: float = 0.01):
        """
        Perform a single time step of the coupled neuroimmune dynamics.
        
        Args:
            dt: Time step size
        """
        # Update KPZ height field
        self.kpz_step(dt)
        
        # Parameters
        spike_tox = self.params['spike_toxicity']
        cereb_vuln = self.params['cerebellar_vulnerability']
        auto_resil = self.params['autonomic_resilience']
        
        # Coupled dynamics equations
        # 1. Spike protein drives immune activation
        immune_activation_rate = spike_tox * (1 - self.immune_activation) - 0.1 * self.immune_activation
        
        # 2. Immune activation and spike toxicity drive neurotoxicity
        neurotox_rate = (spike_tox * self.immune_activation * (1 - self.neurotoxicity) - 
                         auto_resil * self.neurotoxicity)
        
        # 3. Cerebellar vulnerability amplifies neurotoxicity effects on KPZ dynamics
        kpz_coupling = cereb_vuln * self.neurotoxicity * (1 - np.abs(self.h))
        
        # Update fields
        self.immune_activation += dt * immune_activation_rate
        self.neurotoxicity += dt * neurotox_rate
        self.h += dt * kpz_coupling
        
        # Ensure values stay in valid range
        self.immune_activation = np.clip(self.immune_activation, 0, 1)
        self.neurotoxicity = np.clip(self.neurotoxicity, 0, 1)
        
        # Update time and iteration
        self.time += dt
        self.iteration += 1
        
        # Record history
        self.history['time'].append(self.time)
        self.history['mean_h'].append(np.mean(self.h))
        self.history['mean_immune'].append(np.mean(self.immune_activation))
        self.history['mean_neurotox'].append(np.mean(self.neurotoxicity))
        
        # Determine attractor state
        mean_neurotox = np.mean(self.neurotoxicity)
        if mean_neurotox < 0.3:
            state = "Resilient"
        elif mean_neurotox < 0.7:
            state = "Vulnerable"
        else:
            state = "Collapse"
        self.history['attractor_state'].append(state)
    
    def run_simulation(self, time_steps: Optional[int] = None, dt: float = 0.01) -> Dict[str, Any]:
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
        
        logger.info(f"Starting simulation with {time_steps} time steps")
        
        # Reset state before simulation
        self.reset_state()
        
        # Run simulation
        for _ in range(time_steps):
            self.coupled_dynamics_step(dt)
            
            # Log progress every 10% of steps
            if _ % (time_steps // 10) == 0:
                logger.info(f"Simulation progress: {_ / time_steps * 100:.1f}%")
        
        logger.info("Simulation completed")
        
        # Prepare results
        results = {
            'params': self.params.copy(),
            'final_state': {
                'time': self.time,
                'mean_height': float(np.mean(self.h)),
                'mean_immune_activation': float(np.mean(self.immune_activation)),
                'mean_neurotoxicity': float(np.mean(self.neurotoxicity)),
                'final_attractor_state': self.history['attractor_state'][-1]
            },
            'history': {
                'time': self.history['time'],
                'mean_height': self.history['mean_h'],
                'mean_immune_activation': self.history['mean_immune'],
                'mean_neurotoxicity': self.history['mean_neurotox'],
                'attractor_state': self.history['attractor_state']
            }
        }
        
        return results
    
    def plot_phase_portrait(self, save_path: Optional[str] = None):
        """
        Generate a phase portrait of the simulation results.
        
        Args:
            save_path: Path to save the plot image
        """
        if len(self.history['mean_immune']) < 2:
            logger.error("Not enough data for phase portrait. Run simulation first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create a colormap based on time
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.history['time'])))
        
        # Plot the trajectory in the phase space
        plt.scatter(
            self.history['mean_immune'], 
            self.history['mean_neurotox'],
            c=colors, 
            s=10, 
            alpha=0.7
        )
        
        # Add arrows to show direction
        step = max(1, len(self.history['mean_immune']) // 20)
        for i in range(0, len(self.history['mean_immune']) - step, step):
            plt.arrow(
                self.history['mean_immune'][i],
                self.history['mean_neurotox'][i],
                self.history['mean_immune'][i+step] - self.history['mean_immune'][i],
                self.history['mean_neurotox'][i+step] - self.history['mean_neurotox'][i],
                head_width=0.01, 
                head_length=0.02, 
                fc='black', 
                ec='black',
                alpha=0.7
            )
        
        # Add labels for attractor states
        attractor_states = set(self.history['attractor_state'])
        for state in attractor_states:
            indices = [i for i, s in enumerate(self.history['attractor_state']) if s == state]
            if indices:
                mid_idx = indices[len(indices)//2]
                plt.annotate(
                    state,
                    (self.history['mean_immune'][mid_idx], self.history['mean_neurotox'][mid_idx]),
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
                )
        
        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Time Progression')
        
        plt.xlabel('Immune Activation')
        plt.ylabel('Neurotoxicity')
        plt.title('Phase Portrait of Neuroimmune Dynamics')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Phase portrait saved to {save_path}")
        
        plt.close()
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save simulation results to a JSON file.
        
        Args:
            results: Simulation results dictionary
            filepath: Path to save the results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = json.loads(
            json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


def run_sample_simulation():
    """Run a sample simulation with default parameters."""
    simulator = NeuroimmuneDynamics()
    results = simulator.run_simulation(time_steps=500)
    
    # Generate and save phase portrait
    os.makedirs('output', exist_ok=True)
    simulator.plot_phase_portrait(save_path='output/phase_portrait.png')
    
    # Save results
    simulator.save_results(results, 'output/simulation_results.json')
    
    return results


if __name__ == "__main__":
    # Run a sample simulation when executed directly
    run_sample_simulation()
