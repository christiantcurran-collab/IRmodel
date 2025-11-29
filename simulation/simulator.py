"""
CPU-based Monte Carlo Simulator for Two-Factor Hull-White Model

This module implements efficient Monte Carlo simulation of the Two-Factor
Hull-White interest rate model using NumPy for CPU computation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import time
from datetime import datetime

import sys
sys.path.insert(0, '..')
from models.hull_white_2f import HullWhite2FModel, HullWhite2FParams


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation."""
    
    # Simulation paths
    times: np.ndarray                    # Time grid [n_steps + 1]
    short_rates: np.ndarray              # Short rate paths [n_trials, n_steps + 1]
    second_factor: np.ndarray            # Second factor paths [n_trials, n_steps + 1]
    
    # Yield curves at each time step
    yield_curve_times: np.ndarray        # Maturities for yield curves
    yield_curves: np.ndarray             # Yield curves [n_trials, n_steps + 1, n_maturities]
    
    # Summary statistics
    mean_short_rate: np.ndarray          # Mean short rate at each step
    std_short_rate: np.ndarray           # Std of short rate at each step
    mean_yield_curves: np.ndarray        # Mean yield curve at each step [n_steps + 1, n_maturities]
    
    # Metadata
    n_trials: int
    n_steps: int
    dt: float
    total_time: float
    execution_time: float                # Seconds
    device: str                          # 'cpu' or 'gpu'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to JSON-serializable dictionary."""
        return {
            'times': self.times.tolist(),
            'mean_short_rate': self.mean_short_rate.tolist(),
            'std_short_rate': self.std_short_rate.tolist(),
            'yield_curve_times': self.yield_curve_times.tolist(),
            'mean_yield_curves': self.mean_yield_curves.tolist(),
            'n_trials': self.n_trials,
            'n_steps': self.n_steps,
            'dt': self.dt,
            'total_time': self.total_time,
            'execution_time': self.execution_time,
            'device': self.device,
            'timestamp': self.timestamp,
            # Include some sample paths for visualization (first 10 trials)
            'sample_short_rates': self.short_rates[:min(10, self.n_trials)].tolist(),
            'sample_yield_curves': self.yield_curves[:min(5, self.n_trials)].tolist()
        }


class HullWhite2FSimulator:
    """
    CPU-based Monte Carlo simulator for Two-Factor Hull-White model.
    
    Uses NumPy for efficient vectorized simulation.
    """
    
    def __init__(self, model: HullWhite2FModel):
        """
        Initialize the simulator.
        
        Args:
            model: Calibrated Two-Factor Hull-White model
        """
        self.model = model
        self.params = model.params
    
    def simulate(
        self,
        n_trials: int = 1000,
        n_steps: int = 100,
        total_time: float = 10.0,
        yield_maturities: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation of the Two-Factor Hull-White model.
        
        Uses Euler-Maruyama discretization:
        r(t+dt) = r(t) + [θ(t) + u(t) - a*r(t)]*dt + σ₁*√dt*Z₁
        u(t+dt) = u(t) - b*u(t)*dt + σ₂*√dt*Z₂
        
        where Z₁, Z₂ are correlated standard normals.
        
        Args:
            n_trials: Number of Monte Carlo trials
            n_steps: Number of time steps
            total_time: Total simulation time in years
            yield_maturities: Maturities for yield curve computation
            seed: Random seed for reproducibility
            
        Returns:
            SimulationResult with all simulation data
        """
        start_time = time.perf_counter()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Default yield maturities
        if yield_maturities is None:
            yield_maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        
        dt = total_time / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Extract parameters
        a = self.params.a
        b = self.params.b
        sigma1 = self.params.sigma1
        sigma2 = self.params.sigma2
        rho = self.params.rho
        
        # Initialize arrays
        times = np.linspace(0, total_time, n_steps + 1)
        r = np.zeros((n_trials, n_steps + 1))
        u = np.zeros((n_trials, n_steps + 1))
        
        # Initial conditions
        r[:, 0] = self.model.initial_short_rate()
        u[:, 0] = 0.0  # Second factor starts at zero
        
        # Generate correlated random numbers
        # Z2 = rho * Z1 + sqrt(1-rho^2) * Z_independent
        Z1 = np.random.standard_normal((n_trials, n_steps))
        Z_indep = np.random.standard_normal((n_trials, n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_indep
        
        # Euler-Maruyama simulation
        for i in range(n_steps):
            t = times[i]
            theta_t = self.model.theta(np.array([t]))[0]
            
            # Short rate dynamics
            drift_r = (theta_t + u[:, i] - a * r[:, i]) * dt
            diffusion_r = sigma1 * sqrt_dt * Z1[:, i]
            r[:, i + 1] = r[:, i] + drift_r + diffusion_r
            
            # Second factor dynamics
            drift_u = -b * u[:, i] * dt
            diffusion_u = sigma2 * sqrt_dt * Z2[:, i]
            u[:, i + 1] = u[:, i] + drift_u + diffusion_u
        
        # VECTORIZED yield curve computation - much faster!
        yield_curves = self.model.yield_curves_vectorized(times, r, u, yield_maturities)
        
        # Compute statistics
        mean_short_rate = np.mean(r, axis=0)
        std_short_rate = np.std(r, axis=0)
        mean_yield_curves = np.mean(yield_curves, axis=0)
        
        execution_time = time.perf_counter() - start_time
        
        return SimulationResult(
            times=times,
            short_rates=r,
            second_factor=u,
            yield_curve_times=yield_maturities,
            yield_curves=yield_curves,
            mean_short_rate=mean_short_rate,
            std_short_rate=std_short_rate,
            mean_yield_curves=mean_yield_curves,
            n_trials=n_trials,
            n_steps=n_steps,
            dt=dt,
            total_time=total_time,
            execution_time=execution_time,
            device='cpu'
        )
    
    def simulate_antithetic(
        self,
        n_trials: int = 1000,
        n_steps: int = 100,
        total_time: float = 10.0,
        yield_maturities: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run simulation with antithetic variates for variance reduction.
        
        For each set of random numbers Z, we also use -Z, which reduces
        variance in the Monte Carlo estimates.
        """
        start_time = time.perf_counter()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Half the trials since we double with antithetic
        half_trials = n_trials // 2
        
        if yield_maturities is None:
            yield_maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        
        dt = total_time / n_steps
        sqrt_dt = np.sqrt(dt)
        
        a = self.params.a
        b = self.params.b
        sigma1 = self.params.sigma1
        sigma2 = self.params.sigma2
        rho = self.params.rho
        
        times = np.linspace(0, total_time, n_steps + 1)
        
        # Arrays for both original and antithetic paths
        r = np.zeros((n_trials, n_steps + 1))
        u = np.zeros((n_trials, n_steps + 1))
        
        r[:, 0] = self.model.initial_short_rate()
        u[:, 0] = 0.0
        
        # Generate random numbers
        Z1 = np.random.standard_normal((half_trials, n_steps))
        Z_indep = np.random.standard_normal((half_trials, n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_indep
        
        # Antithetic versions
        Z1_full = np.vstack([Z1, -Z1])
        Z2_full = np.vstack([Z2, -Z2])
        
        # Simulation loop
        for i in range(n_steps):
            t = times[i]
            theta_t = self.model.theta(np.array([t]))[0]
            
            drift_r = (theta_t + u[:, i] - a * r[:, i]) * dt
            diffusion_r = sigma1 * sqrt_dt * Z1_full[:, i]
            r[:, i + 1] = r[:, i] + drift_r + diffusion_r
            
            drift_u = -b * u[:, i] * dt
            diffusion_u = sigma2 * sqrt_dt * Z2_full[:, i]
            u[:, i + 1] = u[:, i] + drift_u + diffusion_u
        
        # Compute yield curves
        n_maturities = len(yield_maturities)
        yield_curves = np.zeros((n_trials, n_steps + 1, n_maturities))
        
        for step in range(n_steps + 1):
            t = times[step]
            for trial in range(n_trials):
                yield_curves[trial, step, :] = self.model.yield_from_rates(
                    t, r[trial, step], u[trial, step], yield_maturities
                )
        
        mean_short_rate = np.mean(r, axis=0)
        std_short_rate = np.std(r, axis=0)
        mean_yield_curves = np.mean(yield_curves, axis=0)
        
        execution_time = time.perf_counter() - start_time
        
        return SimulationResult(
            times=times,
            short_rates=r,
            second_factor=u,
            yield_curve_times=yield_maturities,
            yield_curves=yield_curves,
            mean_short_rate=mean_short_rate,
            std_short_rate=std_short_rate,
            mean_yield_curves=mean_yield_curves,
            n_trials=n_trials,
            n_steps=n_steps,
            dt=dt,
            total_time=total_time,
            execution_time=execution_time,
            device='cpu-antithetic'
        )

