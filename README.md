# Two-Factor Hull-White Interest Rate Model Simulator

A comprehensive implementation of the Two-Factor Hull-White interest rate model with Monte Carlo simulation, calibrated to the US Treasury yield curve. Features a modern web interface for visualization and supports both CPU and GPU execution.

![Hull-White 2F Simulator](https://via.placeholder.com/800x400?text=Hull-White+2F+Simulator)

## Features

- **Two-Factor Hull-White Model**: Full implementation of the 2F-HW model with calibration to initial yield curve
- **US Treasury Curve**: Automatic calibration to current US Treasury rates
- **Monte Carlo Simulation**: Efficient path simulation with configurable trials and time steps
- **CPU/GPU Support**: Run simulations on CPU (NumPy) or GPU (CuPy) for comparison
- **Modern Web Interface**: Interactive visualization with Chart.js
- **Real-time Yield Curve Evolution**: Animate yield curve changes through time
- **Performance Benchmarking**: Compare CPU vs GPU execution times

## Model Description

The Two-Factor Hull-White model extends the classic one-factor model by adding a second stochastic factor:

$$dr(t) = [\theta(t) + u(t) - a \cdot r(t)]dt + \sigma_1 dW_1(t)$$

$$du(t) = -b \cdot u(t)dt + \sigma_2 dW_2(t)$$

Where:
- $r(t)$: Short rate
- $u(t)$: Stochastic mean level (second factor)
- $\theta(t)$: Time-dependent drift calibrated to initial yield curve
- $a, b$: Mean reversion speeds
- $\sigma_1, \sigma_2$: Volatilities
- $\rho$: Correlation between $W_1$ and $W_2$

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a | 0.10 | Short rate mean reversion (~7 year half-life) |
| b | 0.04 | Second factor mean reversion |
| σ₁ | 0.01 | Short rate volatility (100 bps/year) |
| σ₂ | 0.008 | Second factor volatility (80 bps/year) |
| ρ | -0.30 | Factor correlation |

## Installation

### Local Installation

```bash
# Clone or navigate to the project directory
cd InterestRate

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
```

Open your browser to `http://localhost:5000`

### Google Colab (with GPU)

1. Upload the project files to Google Colab or mount Google Drive
2. Use the provided `colab_runner.ipynb` notebook
3. Enable GPU runtime: Runtime → Change runtime type → GPU

## Usage

### Web Interface

1. **Configure Simulation**:
   - Set number of Monte Carlo trials (10 - 100,000)
   - Set time steps (5 - 500)
   - Set simulation horizon (0.5 - 50 years)
   - Choose CPU or GPU execution

2. **Adjust Model Parameters** (optional):
   - Mean reversion speeds (a, b)
   - Volatilities (σ₁, σ₂)
   - Factor correlation (ρ)

3. **Run Simulation**: Click "Run Simulation" to generate paths

4. **Visualize Results**:
   - Use the time slider to see yield curve evolution
   - Click play button for animation
   - View short rate paths with confidence bands

5. **Compare Performance**: Click "Compare CPU/GPU" to benchmark

### Python API

```python
from models import HullWhite2FModel, HullWhite2FParams, get_us_treasury_curve
from simulation import HullWhite2FSimulator, HullWhite2FSimulatorGPU

# Create model with custom parameters
params = HullWhite2FParams(
    a=0.1,
    b=0.04,
    sigma1=0.01,
    sigma2=0.008,
    rho=-0.3
)

# Get US Treasury curve
curve = get_us_treasury_curve()

# Create model
model = HullWhite2FModel(params, curve)

# Run CPU simulation
simulator = HullWhite2FSimulator(model)
result = simulator.simulate(
    n_trials=5000,
    n_steps=100,
    total_time=10.0
)

# Access results
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Final mean short rate: {result.mean_short_rate[-1]*100:.2f}%")

# GPU simulation (in Colab)
gpu_sim = HullWhite2FSimulatorGPU(model)
gpu_result = gpu_sim.simulate(n_trials=50000, n_steps=100, total_time=10.0)
print(f"GPU speedup: {result.execution_time / gpu_result.execution_time:.2f}x")
```

## Project Structure

```
InterestRate/
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
├── colab_runner.ipynb       # Google Colab notebook
├── models/
│   ├── __init__.py
│   ├── hull_white_2f.py     # Two-Factor HW model
│   └── treasury_curve.py    # US Treasury curve utilities
├── simulation/
│   ├── __init__.py
│   ├── simulator.py         # CPU simulator
│   └── gpu_simulator.py     # GPU simulator (CuPy)
├── templates/
│   └── index.html           # Web interface
└── static/
    ├── style.css            # Styles
    └── app.js               # Frontend JavaScript
```

## GPU Acceleration

For GPU acceleration in Google Colab:

```python
# Install CuPy for your CUDA version
!pip install cupy-cuda12x  # For CUDA 12.x (check with !nvidia-smi)

# The simulator will automatically use GPU when available
from simulation import HullWhite2FSimulatorGPU
gpu_sim = HullWhite2FSimulatorGPU(model)
result = gpu_sim.simulate(n_trials=100000, n_steps=100)
```

Expected speedups:
- 10x-50x for large simulations (>10,000 trials)
- 2x-10x for medium simulations (1,000-10,000 trials)
- May be slower for small simulations due to data transfer overhead

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/parameters` | GET | Get default model parameters |
| `/api/gpu-info` | GET | Get GPU availability info |
| `/api/initial-curve` | GET | Get initial US Treasury curve |
| `/api/simulate` | POST | Run Monte Carlo simulation |
| `/api/compare` | POST | Run CPU vs GPU comparison |

## Technical Notes

### Calibration

The model is calibrated to fit the initial yield curve by computing θ(t):

$$\theta(t) = \frac{\partial f(0,t)}{\partial t} + a \cdot f(0,t) + \frac{\sigma_1^2}{2a}(1 - e^{-2at})$$

Where $f(0,t)$ is the instantaneous forward rate derived from the yield curve.

### Numerical Methods

- **Euler-Maruyama**: Used for path discretization
- **Correlated Brownian Motions**: Generated using Cholesky decomposition
- **Antithetic Variates**: Available for variance reduction

## License

MIT License - See LICENSE file for details.

## References

1. Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure Models II: Two-Factor Models"
2. Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice"

