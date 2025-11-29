"""
Two-Factor Hull-White Interest Rate Model

The Two-Factor Hull-White model extends the one-factor model by adding a second
stochastic factor that allows for richer yield curve dynamics and better fitting
of the volatility term structure.

Model Dynamics:
    dr(t) = [θ(t) + u(t) - a*r(t)]dt + σ₁*dW₁(t)
    du(t) = -b*u(t)dt + σ₂*dW₂(t)

Where:
    - r(t): Short rate
    - u(t): Stochastic mean level (second factor)
    - θ(t): Time-dependent drift calibrated to initial yield curve
    - a: Mean reversion speed for short rate
    - b: Mean reversion speed for second factor
    - σ₁: Volatility of short rate
    - σ₂: Volatility of second factor
    - ρ: Correlation between W₁ and W₂
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from scipy.interpolate import CubicSpline


@dataclass
class HullWhite2FParams:
    """Parameters for the Two-Factor Hull-White model."""
    a: float = 0.1          # Mean reversion speed for r
    b: float = 0.05         # Mean reversion speed for u
    sigma1: float = 0.01    # Volatility of r
    sigma2: float = 0.008   # Volatility of u
    rho: float = -0.3       # Correlation between factors
    
    def validate(self):
        """Validate parameter constraints."""
        assert self.a > 0, "Mean reversion 'a' must be positive"
        assert self.b > 0, "Mean reversion 'b' must be positive"
        assert self.sigma1 > 0, "Volatility 'sigma1' must be positive"
        assert self.sigma2 > 0, "Volatility 'sigma2' must be positive"
        assert -1 <= self.rho <= 1, "Correlation 'rho' must be in [-1, 1]"


class YieldCurve:
    """Represents an initial yield curve for calibration."""
    
    def __init__(self, maturities: np.ndarray, yields: np.ndarray):
        """
        Initialize yield curve.
        
        Args:
            maturities: Array of maturities in years
            yields: Array of continuously compounded yields
        """
        self.maturities = np.array(maturities)
        self.yields = np.array(yields)
        
        # Fit cubic spline for interpolation
        self._spline = CubicSpline(maturities, yields, bc_type='natural')
        
        # Compute instantaneous forward rates
        self._compute_forward_rates()
    
    def _compute_forward_rates(self):
        """Compute instantaneous forward rates from yield curve."""
        # f(0,T) = y(T) + T * dy/dT
        fine_maturities = np.linspace(0.001, self.maturities[-1], 1000)
        yields_fine = self._spline(fine_maturities)
        dy_dt = self._spline(fine_maturities, 1)  # First derivative
        forward_rates = yields_fine + fine_maturities * dy_dt
        self._forward_spline = CubicSpline(fine_maturities, forward_rates, bc_type='natural')
    
    def yield_at(self, T: np.ndarray) -> np.ndarray:
        """Get yield at maturity T."""
        T = np.atleast_1d(T)
        return self._spline(np.clip(T, self.maturities[0], self.maturities[-1]))
    
    def forward_rate(self, T: np.ndarray) -> np.ndarray:
        """Get instantaneous forward rate f(0,T)."""
        T = np.atleast_1d(T)
        T_clipped = np.clip(T, 0.001, self.maturities[-1])
        return self._forward_spline(T_clipped)
    
    def discount_factor(self, T: np.ndarray) -> np.ndarray:
        """Get discount factor P(0,T) = exp(-y(T)*T)."""
        T = np.atleast_1d(T)
        return np.exp(-self.yield_at(T) * T)


class HullWhite2FModel:
    """Two-Factor Hull-White Interest Rate Model."""
    
    def __init__(self, params: HullWhite2FParams, initial_curve: YieldCurve):
        """
        Initialize the Two-Factor Hull-White model.
        
        Args:
            params: Model parameters
            initial_curve: Initial yield curve for calibration
        """
        params.validate()
        self.params = params
        self.initial_curve = initial_curve
        
        # Calibrate theta(t) to match initial yield curve
        self._calibrate_theta()
    
    def _calibrate_theta(self):
        """
        Calibrate θ(t) to fit the initial yield curve.
        
        In the Two-Factor HW model:
        θ(t) = ∂f(0,t)/∂t + a*f(0,t) + σ₁²/(2a)*(1 - e^(-2at))
        """
        # Create fine grid for theta
        self.theta_grid = np.linspace(0.001, self.initial_curve.maturities[-1], 500)
        
        # Get forward rates and their derivatives
        f = self.initial_curve.forward_rate(self.theta_grid)
        
        # Numerical derivative of forward rate
        df_dt = np.gradient(f, self.theta_grid)
        
        a = self.params.a
        sigma1 = self.params.sigma1
        
        # Compute theta
        self.theta_values = df_dt + a * f + (sigma1**2 / (2 * a)) * (1 - np.exp(-2 * a * self.theta_grid))
        
        # Create spline for theta(t)
        self._theta_spline = CubicSpline(self.theta_grid, self.theta_values, bc_type='natural')
    
    def theta(self, t: np.ndarray) -> np.ndarray:
        """Get calibrated theta at time t."""
        t = np.atleast_1d(t)
        t_clipped = np.clip(t, self.theta_grid[0], self.theta_grid[-1])
        return self._theta_spline(t_clipped)
    
    def initial_short_rate(self) -> float:
        """Get the initial short rate r(0) from the yield curve."""
        return float(self.initial_curve.forward_rate(np.array([0.001]))[0])
    
    def B_factor(self, t: float, T: float, mean_rev: float) -> float:
        """
        Compute B(t,T) factor for bond pricing.
        B(t,T) = (1 - e^(-mean_rev*(T-t))) / mean_rev
        """
        if T <= t:
            return 0.0
        return (1 - np.exp(-mean_rev * (T - t))) / mean_rev
    
    def zero_bond_price(self, t: float, T: float, r_t: float, u_t: float) -> float:
        """
        Compute zero-coupon bond price P(t,T) under the Two-Factor HW model.
        
        P(t,T) = A(t,T) * exp(-B₁(t,T)*r(t) - B₂(t,T)*u(t))
        """
        if T <= t:
            return 1.0
        
        a, b = self.params.a, self.params.b
        sigma1, sigma2 = self.params.sigma1, self.params.sigma2
        rho = self.params.rho
        
        B1 = self.B_factor(t, T, a)
        B2 = self.B_factor(t, T, b)
        
        # Compute A(t,T) using the initial curve
        P_0_T = float(self.initial_curve.discount_factor(np.array([T]))[0])
        P_0_t = float(self.initial_curve.discount_factor(np.array([t]))[0]) if t > 0 else 1.0
        f_0_t = float(self.initial_curve.forward_rate(np.array([max(t, 0.001)]))[0])
        
        # Variance terms
        tau = T - t
        V = self._compute_variance(t, T)
        
        ln_A = np.log(P_0_T / P_0_t) - B1 * f_0_t + 0.5 * V
        A = np.exp(ln_A)
        
        return A * np.exp(-B1 * r_t - B2 * u_t)
    
    def _compute_variance(self, t: float, T: float) -> float:
        """Compute the variance term for bond pricing."""
        a, b = self.params.a, self.params.b
        sigma1, sigma2 = self.params.sigma1, self.params.sigma2
        rho = self.params.rho
        
        tau = T - t
        
        # Simplified variance approximation
        V1 = (sigma1**2 / (2 * a**3)) * (
            2 * a * tau - 3 + 4 * np.exp(-a * tau) - np.exp(-2 * a * tau)
        )
        
        V2 = (sigma2**2 / (2 * b**3)) * (
            2 * b * tau - 3 + 4 * np.exp(-b * tau) - np.exp(-2 * b * tau)
        )
        
        # Cross-variance term
        V12 = 2 * rho * sigma1 * sigma2 / (a * b * (a + b)) * (
            tau - (1 - np.exp(-a * tau)) / a - (1 - np.exp(-b * tau)) / b +
            (1 - np.exp(-(a + b) * tau)) / (a + b)
        )
        
        return V1 + V2 + V12
    
    def yield_from_rates(self, t: float, r_t: float, u_t: float, 
                         maturities: np.ndarray) -> np.ndarray:
        """
        Compute yield curve at time t given short rate r(t) and factor u(t).
        
        Args:
            t: Current time
            r_t: Current short rate
            u_t: Current value of second factor
            maturities: Maturities (from time t) for which to compute yields
            
        Returns:
            Array of yields for given maturities
        """
        yields = np.zeros(len(maturities))
        
        for i, tau in enumerate(maturities):
            if tau <= 0:
                yields[i] = r_t
            else:
                T = t + tau
                P = self.zero_bond_price(t, T, r_t, u_t)
                yields[i] = -np.log(P) / tau
        
        return yields
    
    def yield_curves_vectorized(self, times: np.ndarray, r_all: np.ndarray, 
                                 u_all: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        """
        VECTORIZED yield curve computation for all trials and time steps.
        
        Args:
            times: (n_steps+1,) array of time points
            r_all: (n_trials, n_steps+1) array of short rates
            u_all: (n_trials, n_steps+1) array of second factor
            maturities: (n_mats,) array of maturities
        
        Returns:
            (n_trials, n_steps+1, n_mats) array of yields
        """
        n_trials, n_steps_plus1 = r_all.shape
        n_mats = len(maturities)
        a, b = self.params.a, self.params.b
        
        # Pre-compute B factors and A terms for all (time, maturity) pairs
        B1 = np.zeros((n_steps_plus1, n_mats))
        B2 = np.zeros((n_steps_plus1, n_mats))
        ln_A = np.zeros((n_steps_plus1, n_mats))
        
        for i, t in enumerate(times):
            for j, tau in enumerate(maturities):
                if tau > 0:
                    B1[i, j] = (1 - np.exp(-a * tau)) / a
                    B2[i, j] = (1 - np.exp(-b * tau)) / b
                    
                    T = t + tau
                    P_0_T = self.initial_curve.discount_factor(np.array([T]))[0]
                    P_0_t = self.initial_curve.discount_factor(np.array([max(t, 0.001)]))[0]
                    f_0_t = self.initial_curve.forward_rate(np.array([max(t, 0.001)]))[0]
                    
                    ln_A[i, j] = np.log(P_0_T / P_0_t) - B1[i, j] * f_0_t
        
        # Compute yields using broadcasting
        r_expanded = r_all[:, :, np.newaxis]
        u_expanded = u_all[:, :, np.newaxis]
        
        B1_expanded = B1[np.newaxis, :, :]
        B2_expanded = B2[np.newaxis, :, :]
        ln_A_expanded = ln_A[np.newaxis, :, :]
        
        ln_P = ln_A_expanded - B1_expanded * r_expanded - B2_expanded * u_expanded
        tau_expanded = maturities[np.newaxis, np.newaxis, :]
        
        yields = np.where(tau_expanded > 0, -ln_P / tau_expanded, r_expanded)
        
        return yields


def create_default_model() -> HullWhite2FModel:
    """Create a default Two-Factor Hull-White model with typical US Treasury curve."""
    from .treasury_curve import get_us_treasury_curve
    
    params = HullWhite2FParams(
        a=0.1,          # Mean reversion for short rate (~10 year half-life)
        b=0.05,         # Slower mean reversion for second factor
        sigma1=0.01,    # 1% short rate volatility
        sigma2=0.008,   # 0.8% second factor volatility
        rho=-0.3        # Negative correlation (typical for rates)
    )
    
    curve = get_us_treasury_curve()
    return HullWhite2FModel(params, curve)

