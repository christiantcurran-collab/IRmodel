"""
GPU-accelerated Monte Carlo Simulator for Two-Factor Hull-White Model

This module implements GPU-accelerated simulation using CuPy (NVIDIA CUDA).
Designed to work in Google Colab with GPU runtime.

Falls back to CPU (NumPy) if GPU is not available.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time
from datetime import datetime

import sys
sys.path.insert(0, '..')
from models.hull_white_2f import HullWhite2FModel
from .simulator import SimulationResult

# Try to import CuPy for GPU acceleration
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU available: {cp.cuda.Device().name}")
except ImportError:
    cp = None
    print("CuPy not installed. GPU acceleration unavailable.")
except Exception as e:
    cp = None
    print(f"GPU initialization failed: {e}")


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPU."""
    if not GPU_AVAILABLE or cp is None:
        return {
            'available': False,
            'reason': 'CuPy not installed or no CUDA GPU available'
        }
    
    try:
        device = cp.cuda.Device()
        return {
            'available': True,
            'name': device.name,
            'compute_capability': device.compute_capability,
            'total_memory': device.mem_info[1] / (1024**3),  # GB
            'free_memory': device.mem_info[0] / (1024**3)    # GB
        }
    except Exception as e:
        return {
            'available': False,
            'reason': str(e)
        }


class HullWhite2FSimulatorGPU:
    """
    GPU-accelerated Monte Carlo simulator for Two-Factor Hull-White model.
    
    Uses CuPy for NVIDIA GPU acceleration. Falls back to NumPy if GPU unavailable.
    """
    
    def __init__(self, model: HullWhite2FModel, force_cpu: bool = False):
        """
        Initialize the GPU simulator.
        
        Args:
            model: Calibrated Two-Factor Hull-White model
            force_cpu: If True, use CPU even if GPU is available
        """
        self.model = model
        self.params = model.params
        self.use_gpu = GPU_AVAILABLE and not force_cpu
        
        # Select array library
        self.xp = cp if self.use_gpu else np
        
        if self.use_gpu:
            print("Using GPU acceleration (CuPy)")
        else:
            print("Using CPU (NumPy)")
    
    def simulate(
        self,
        n_trials: int = 1000,
        n_steps: int = 100,
        total_time: float = 10.0,
        yield_maturities: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run GPU-accelerated Monte Carlo simulation.
        
        The simulation kernel runs entirely on GPU, with data transferred
        back to CPU only at the end.
        
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
        xp = self.xp
        
        if seed is not None:
            if self.use_gpu:
                cp.random.seed(seed)
            else:
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
        
        # Pre-compute theta values on CPU and transfer to GPU
        times = np.linspace(0, total_time, n_steps + 1)
        theta_values = self.model.theta(times[:-1])  # theta at each step
        
        if self.use_gpu:
            theta_gpu = cp.asarray(theta_values)
        else:
            theta_gpu = theta_values
        
        # Initialize arrays on GPU
        r = xp.zeros((n_trials, n_steps + 1), dtype=xp.float32)
        u = xp.zeros((n_trials, n_steps + 1), dtype=xp.float32)
        
        # Initial conditions
        r0 = self.model.initial_short_rate()
        r[:, 0] = r0
        u[:, 0] = 0.0
        
        # Generate all random numbers at once (more efficient on GPU)
        Z1 = xp.random.standard_normal((n_trials, n_steps), dtype=xp.float32)
        Z_indep = xp.random.standard_normal((n_trials, n_steps), dtype=xp.float32)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_indep
        
        # Vectorized Euler-Maruyama simulation
        # This loop is necessary but each step is fully vectorized across trials
        for i in range(n_steps):
            theta_t = theta_gpu[i]
            
            # Short rate dynamics (vectorized across all trials)
            drift_r = (theta_t + u[:, i] - a * r[:, i]) * dt
            diffusion_r = sigma1 * sqrt_dt * Z1[:, i]
            r[:, i + 1] = r[:, i] + drift_r + diffusion_r
            
            # Second factor dynamics
            drift_u = -b * u[:, i] * dt
            diffusion_u = sigma2 * sqrt_dt * Z2[:, i]
            u[:, i + 1] = u[:, i] + drift_u + diffusion_u
        
        # Synchronize GPU before timing yield curve computation
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
        
        path_time = time.perf_counter() - start_time
        
        # Transfer results back to CPU for yield curve computation
        if self.use_gpu:
            r_cpu = cp.asnumpy(r)
            u_cpu = cp.asnumpy(u)
        else:
            r_cpu = r
            u_cpu = u
        
        # VECTORIZED yield curve computation - much faster!
        yield_curves = self.model.yield_curves_vectorized(times, r_cpu, u_cpu, yield_maturities)
        
        # Compute statistics
        mean_short_rate = np.mean(r_cpu, axis=0)
        std_short_rate = np.std(r_cpu, axis=0)
        mean_yield_curves = np.mean(yield_curves, axis=0)
        
        execution_time = time.perf_counter() - start_time
        
        device_name = 'gpu' if self.use_gpu else 'cpu'
        if self.use_gpu:
            try:
                device_name = f"gpu ({cp.cuda.Device().name})"
            except:
                device_name = 'gpu'
        
        return SimulationResult(
            times=times,
            short_rates=r_cpu.astype(np.float64),
            second_factor=u_cpu.astype(np.float64),
            yield_curve_times=yield_maturities,
            yield_curves=yield_curves.astype(np.float64),
            mean_short_rate=mean_short_rate.astype(np.float64),
            std_short_rate=std_short_rate.astype(np.float64),
            mean_yield_curves=mean_yield_curves.astype(np.float64),
            n_trials=n_trials,
            n_steps=n_steps,
            dt=dt,
            total_time=total_time,
            execution_time=execution_time,
            device=device_name
        )
    
    def simulate_batch(
        self,
        n_trials: int = 10000,
        n_steps: int = 100,
        total_time: float = 10.0,
        yield_maturities: Optional[np.ndarray] = None,
        batch_size: int = 5000,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run simulation in batches to manage GPU memory.
        
        For very large simulations, this prevents GPU memory exhaustion.
        
        Args:
            n_trials: Total number of Monte Carlo trials
            n_steps: Number of time steps
            total_time: Total simulation time in years
            yield_maturities: Maturities for yield curve computation
            batch_size: Number of trials per batch
            seed: Random seed for reproducibility
            
        Returns:
            SimulationResult with combined results from all batches
        """
        start_time = time.perf_counter()
        
        if yield_maturities is None:
            yield_maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        
        n_batches = (n_trials + batch_size - 1) // batch_size
        
        # Accumulators for results
        all_r = []
        all_u = []
        all_yield_curves = []
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_trials)
            batch_n = batch_end - batch_start
            
            # Set seed for reproducibility across batches
            batch_seed = seed + batch_idx if seed is not None else None
            
            result = self.simulate(
                n_trials=batch_n,
                n_steps=n_steps,
                total_time=total_time,
                yield_maturities=yield_maturities,
                seed=batch_seed
            )
            
            all_r.append(result.short_rates)
            all_u.append(result.second_factor)
            all_yield_curves.append(result.yield_curves)
            
            # Clear GPU memory between batches
            if self.use_gpu:
                cp.get_default_memory_pool().free_all_blocks()
        
        # Combine results
        r_combined = np.vstack(all_r)
        u_combined = np.vstack(all_u)
        yield_curves_combined = np.vstack(all_yield_curves)
        
        times = result.times
        mean_short_rate = np.mean(r_combined, axis=0)
        std_short_rate = np.std(r_combined, axis=0)
        mean_yield_curves = np.mean(yield_curves_combined, axis=0)
        
        execution_time = time.perf_counter() - start_time
        
        device_name = 'gpu-batched' if self.use_gpu else 'cpu-batched'
        
        return SimulationResult(
            times=times,
            short_rates=r_combined,
            second_factor=u_combined,
            yield_curve_times=yield_maturities,
            yield_curves=yield_curves_combined,
            mean_short_rate=mean_short_rate,
            std_short_rate=std_short_rate,
            mean_yield_curves=mean_yield_curves,
            n_trials=n_trials,
            n_steps=n_steps,
            dt=result.dt,
            total_time=total_time,
            execution_time=execution_time,
            device=device_name
        )


def run_comparison(
    model: HullWhite2FModel,
    n_trials: int = 5000,
    n_steps: int = 100,
    total_time: float = 10.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run simulation on both CPU and GPU and compare performance.
    
    Args:
        model: Calibrated Two-Factor Hull-White model
        n_trials: Number of Monte Carlo trials
        n_steps: Number of time steps
        total_time: Total simulation time in years
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results and timing comparison
    """
    from .simulator import HullWhite2FSimulator
    
    results = {
        'n_trials': n_trials,
        'n_steps': n_steps,
        'total_time': total_time,
        'gpu_info': get_gpu_info()
    }
    
    # CPU simulation
    cpu_sim = HullWhite2FSimulator(model)
    cpu_result = cpu_sim.simulate(
        n_trials=n_trials,
        n_steps=n_steps,
        total_time=total_time,
        seed=seed
    )
    results['cpu'] = {
        'execution_time': cpu_result.execution_time,
        'device': cpu_result.device
    }
    
    # GPU simulation
    gpu_sim = HullWhite2FSimulatorGPU(model)
    gpu_result = gpu_sim.simulate(
        n_trials=n_trials,
        n_steps=n_steps,
        total_time=total_time,
        seed=seed
    )
    results['gpu'] = {
        'execution_time': gpu_result.execution_time,
        'device': gpu_result.device
    }
    
    # Speedup
    if gpu_result.execution_time > 0:
        results['speedup'] = cpu_result.execution_time / gpu_result.execution_time
    else:
        results['speedup'] = 0
    
    results['cpu_result'] = cpu_result
    results['gpu_result'] = gpu_result
    
    return results

