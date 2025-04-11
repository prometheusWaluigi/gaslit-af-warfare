#!/usr/bin/env python3
"""
Neuroimmune Dynamics Simulator for GASLIT-AF WARSTACK

This module simulates the core GASLIT-AF attractor states:
- Spike protein neurotoxicity
- Cerebellar trauma
- Behavioral and autonomic collapse

Using KPZ / fKPZχ simulations of neuroimmune dynamics, ODE/PDE attractor maps,
and phase portraits of feedback loop entrapment.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# Optional imports for hardware acceleration
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import dpctl
    import dpctl.tensor as dpt
    HAS_ONEAPI = True
except ImportError:
    HAS_ONEAPI = False


class NeuroimmuneDynamics:
    """
    Simulates neuroimmune dynamics using KPZ / fKPZχ models and ODE/PDE systems.
    
    This class implements various models of neuroimmune dynamics, focusing on
    the interactions between spike proteins, cerebellar function, and autonomic
    nervous system responses.
    """
    
    def __init__(self, config=None):
        """
        Initialize the neuroimmune dynamics simulator.
        
        Args:
            config (dict, optional): Configuration parameters for the simulation.
        """
        # Default configuration
        self.config = {
            'grid_size': 100,
            'time_steps': 1000,
            'dt': 0.01,
            'noise_strength': 0.1,
            'diffusion_constant': 0.5,
            'reaction_rate': 1.0,
            'coupling_strength': 0.8,
            'initial_condition': 'random',
            'boundary_condition': 'periodic',
            'use_hardware_acceleration': True,
            'output_dir': 'results/biological_modeling',
            'random_seed': 42
        }
        
        # Update with user configuration if provided
        if config is not None:
            self.config.update(config)
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        # Initialize state variables
        self.t = 0.0
        self.iteration = 0
        self.grid = None
        self.history = []
        self.attractor_states = []
        self.phase_portrait = None
        
        # Initialize hardware acceleration if available
        self.device = self._initialize_device()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def _initialize_device(self):
        """Initialize hardware acceleration device if available."""
        device = None
        
        if self.config['use_hardware_acceleration']:
            if HAS_ONEAPI:
                try:
                    # Try to use Intel GPU if available
                    gpu_devices = [d for d in dpctl.get_devices() if d.is_gpu]
                    if gpu_devices:
                        device = gpu_devices[0]
                        print(f"Using Intel OneAPI with device: {device}")
                    else:
                        cpu_devices = [d for d in dpctl.get_devices() if d.is_cpu]
                        if cpu_devices:
                            device = cpu_devices[0]
                            print(f"Using Intel OneAPI with device: {device}")
                except Exception as e:
                    print(f"Error initializing Intel OneAPI: {e}")
            
            elif HAS_TORCH:
                try:
                    if torch.cuda.is_available():
                        device = "cuda"
                        print("Using PyTorch with CUDA")
                    else:
                        device = "cpu"
                        print("Using PyTorch with CPU")
                except Exception as e:
                    print(f"Error initializing PyTorch: {e}")
        
        if device is None:
            print("Using NumPy for computations (no hardware acceleration)")
        
        return device
    
    def initialize_grid(self):
        """Initialize the simulation grid based on configuration."""
        size = self.config['grid_size']
        
        if self.config['initial_condition'] == 'random':
            self.grid = np.random.rand(size, size)
        elif self.config['initial_condition'] == 'center_peak':
            self.grid = np.zeros((size, size))
            center = size // 2
            radius = size // 10
            for i in range(size):
                for j in range(size):
                    if (i - center)**2 + (j - center)**2 < radius**2:
                        self.grid[i, j] = 1.0
        elif self.config['initial_condition'] == 'gradient':
            x = np.linspace(0, 1, size)
            y = np.linspace(0, 1, size)
            X, Y = np.meshgrid(x, y)
            self.grid = X * Y
        else:
            # Default to random
            self.grid = np.random.rand(size, size)
        
        # Store initial state
        self.history.append(self.grid.copy())
        
        return self.grid
    
    def kpz_step(self):
        """
        Perform a single step of the KPZ (Kardar-Parisi-Zhang) equation.
        
        The KPZ equation models the growth of an interface and is given by:
        ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η
        
        where:
        - h is the height field (our grid)
        - ν is the diffusion constant
        - λ is the coupling strength
        - η is a noise term
        """
        # Extract parameters
        dt = self.config['dt']
        D = self.config['diffusion_constant']
        noise = self.config['noise_strength']
        coupling = self.config['coupling_strength']
        
        # Make a copy of the current grid
        new_grid = self.grid.copy()
        
        # Compute the Laplacian (∇²h)
        laplacian = np.zeros_like(self.grid)
        for i in range(1, self.grid.shape[0] - 1):
            for j in range(1, self.grid.shape[1] - 1):
                laplacian[i, j] = (
                    self.grid[i+1, j] + self.grid[i-1, j] +
                    self.grid[i, j+1] + self.grid[i, j-1] - 4 * self.grid[i, j]
                )
        
        # Compute the gradient squared term (∇h)²
        gradient_x = np.zeros_like(self.grid)
        gradient_y = np.zeros_like(self.grid)
        
        for i in range(1, self.grid.shape[0] - 1):
            for j in range(1, self.grid.shape[1] - 1):
                gradient_x[i, j] = (self.grid[i+1, j] - self.grid[i-1, j]) / 2
                gradient_y[i, j] = (self.grid[i, j+1] - self.grid[i, j-1]) / 2
        
        gradient_squared = gradient_x**2 + gradient_y**2
        
        # Generate noise term
        eta = np.random.normal(0, noise, self.grid.shape)
        
        # Update the grid using the KPZ equation
        new_grid = self.grid + dt * (
            D * laplacian + 
            coupling * gradient_squared / 2 + 
            eta
        )
        
        # Apply boundary conditions
        if self.config['boundary_condition'] == 'periodic':
            # Copy the edges to the opposite sides
            new_grid[0, :] = new_grid[-2, :]
            new_grid[-1, :] = new_grid[1, :]
            new_grid[:, 0] = new_grid[:, -2]
            new_grid[:, -1] = new_grid[:, 1]
        else:
            # Fixed boundary conditions (edges stay at 0)
            new_grid[0, :] = 0
            new_grid[-1, :] = 0
            new_grid[:, 0] = 0
            new_grid[:, -1] = 0
        
        # Update the grid
        self.grid = new_grid
        
        # Update time and iteration
        self.t += dt
        self.iteration += 1
        
        # Store the current state if needed
        if self.iteration % 10 == 0:
            self.history.append(self.grid.copy())
        
        return self.grid
    
    def run_simulation(self, steps=None):
        """
        Run the KPZ simulation for the specified number of steps.
        
        Args:
            steps (int, optional): Number of steps to run. If None, uses the
                                  value from the configuration.
        
        Returns:
            list: History of grid states during the simulation.
        """
        if steps is None:
            steps = self.config['time_steps']
        
        # Initialize the grid if not already done
        if self.grid is None:
            self.initialize_grid()
        
        # Run the simulation
        start_time = time.time()
        for _ in range(steps):
            self.kpz_step()
        
        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        
        return self.history
    
    def compute_attractor_states(self):
        """
        Compute the attractor states of the system.
        
        This method analyzes the simulation history to identify stable
        attractor states in the system dynamics.
        
        Returns:
            list: Identified attractor states.
        """
        if not self.history:
            raise ValueError("No simulation history available. Run the simulation first.")
        
        # Compute the average value of the grid over time
        avg_values = [np.mean(grid) for grid in self.history]
        
        # Find peaks in the average values
        peaks, _ = find_peaks(avg_values, height=0, distance=20)
        
        # Extract the attractor states
        self.attractor_states = [self.history[i] for i in peaks]
        
        return self.attractor_states
    
    def generate_phase_portrait(self, param1_range, param2_range, param1_name='diffusion_constant', param2_name='coupling_strength'):
        """
        Generate a phase portrait by varying two parameters.
        
        Args:
            param1_range (list): Range of values for the first parameter.
            param2_range (list): Range of values for the second parameter.
            param1_name (str): Name of the first parameter.
            param2_name (str): Name of the second parameter.
        
        Returns:
            numpy.ndarray: Phase portrait data.
        """
        # Initialize the phase portrait
        portrait = np.zeros((len(param1_range), len(param2_range)))
        
        # Iterate over parameter values
        for i, p1 in enumerate(param1_range):
            for j, p2 in enumerate(param2_range):
                # Update the configuration
                config = self.config.copy()
                config[param1_name] = p1
                config[param2_name] = p2
                
                # Create a new simulator with this configuration
                sim = NeuroimmuneDynamics(config)
                sim.initialize_grid()
                
                # Run a short simulation
                sim.run_simulation(steps=100)
                
                # Compute a metric for the phase portrait
                # (e.g., average value, variance, etc.)
                portrait[i, j] = np.var(sim.grid)
        
        self.phase_portrait = portrait
        return portrait
    
    def visualize_grid(self, grid=None, title=None, save_path=None):
        """
        Visualize the current state of the grid.
        
        Args:
            grid (numpy.ndarray, optional): Grid to visualize. If None, uses the current grid.
            title (str, optional): Title for the plot.
            save_path (str, optional): Path to save the visualization.
        """
        if grid is None:
            grid = self.grid
        
        if grid is None:
            raise ValueError("No grid available to visualize.")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='viridis', origin='lower')
        plt.colorbar(label='Value')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Grid State at t={self.t:.2f}, Iteration {self.iteration}')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.close()
    
    def visualize_history(self, interval=10, save_dir=None):
        """
        Visualize the history of the grid states.
        
        Args:
            interval (int): Interval between frames to visualize.
            save_dir (str, optional): Directory to save the visualizations.
        """
        if not self.history:
            raise ValueError("No simulation history available. Run the simulation first.")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i, grid in enumerate(self.history[::interval]):
            t = i * interval * self.config['dt']
            title = f'Grid State at t={t:.2f}, Iteration {i * interval}'
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f'grid_state_{i:04d}.png')
            
            self.visualize_grid(grid, title, save_path)
    
    def visualize_phase_portrait(self, param1_name='diffusion_constant', param2_name='coupling_strength', save_path=None):
        """
        Visualize the phase portrait.
        
        Args:
            param1_name (str): Name of the first parameter.
            param2_name (str): Name of the second parameter.
            save_path (str, optional): Path to save the visualization.
        """
        if self.phase_portrait is None:
            raise ValueError("No phase portrait available. Generate it first.")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(self.phase_portrait, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Variance')
        plt.title(f'Phase Portrait: {param1_name} vs {param2_name}')
        plt.xlabel(param1_name)
        plt.ylabel(param2_name)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Phase portrait saved to {save_path}")
        
        plt.close()
    
    def save_results(self, filename=None):
        """
        Save the simulation results to a file.
        
        Args:
            filename (str, optional): Name of the file to save the results.
                                     If None, a default name is generated.
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"neuroimmune_simulation_{timestamp}.json"
        
        filepath = os.path.join(self.config['output_dir'], filename)
        
        # Prepare the results
        results = {
            'config': self.config,
            'final_state': self.grid.tolist() if self.grid is not None else None,
            'time': self.t,
            'iterations': self.iteration,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        
        return filepath
    
    def load_results(self, filepath):
        """
        Load simulation results from a file.
        
        Args:
            filepath (str): Path to the results file.
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Update the configuration
        self.config = results['config']
        
        # Load the final state
        if results['final_state'] is not None:
            self.grid = np.array(results['final_state'])
        
        # Update time and iteration
        self.t = results['time']
        self.iteration = results['iterations']
        
        print(f"Results loaded from {filepath}")
        
        return results


def run_sample_simulation():
    """Run a sample simulation and visualize the results."""
    # Create a simulator with default configuration
    simulator = NeuroimmuneDynamics()
    
    # Initialize the grid
    simulator.initialize_grid()
    
    # Run the simulation
    simulator.run_simulation(steps=500)
    
    # Visualize the final state
    output_dir = simulator.config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    simulator.visualize_grid(save_path=os.path.join(output_dir, 'final_state.png'))
    
    # Generate and visualize a phase portrait
    param1_range = np.linspace(0.1, 1.0, 10)
    param2_range = np.linspace(0.1, 1.0, 10)
    simulator.generate_phase_portrait(param1_range, param2_range)
    simulator.visualize_phase_portrait(save_path=os.path.join(output_dir, 'phase_portrait.png'))
    
    # Save the results
    simulator.save_results()
    
    return simulator


if __name__ == "__main__":
    # Run a sample simulation if the script is executed directly
    simulator = run_sample_simulation()
    
    print("Sample simulation completed successfully.")
