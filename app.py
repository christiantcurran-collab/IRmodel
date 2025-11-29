"""
Flask Web Application for Two-Factor Hull-White Model Simulation

This application provides a web interface for:
- Configuring model parameters
- Running Monte Carlo simulations (CPU/GPU)
- Visualizing yield curve evolution
- Comparing CPU vs GPU performance
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.hull_white_2f import HullWhite2FModel, HullWhite2FParams, YieldCurve
from models.treasury_curve import get_us_treasury_curve, get_model_parameters, get_parameter_descriptions
from simulation.simulator import HullWhite2FSimulator
from simulation.gpu_simulator import HullWhite2FSimulatorGPU, get_gpu_info

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global model cache
_model_cache = {}


def get_model(params_dict: dict = None) -> HullWhite2FModel:
    """Get or create a Hull-White model with given parameters."""
    if params_dict is None:
        params_dict = get_model_parameters()
    
    cache_key = json.dumps(params_dict, sort_keys=True)
    
    if cache_key not in _model_cache:
        params = HullWhite2FParams(
            a=params_dict.get('a', 0.1),
            b=params_dict.get('b', 0.04),
            sigma1=params_dict.get('sigma1', 0.01),
            sigma2=params_dict.get('sigma2', 0.008),
            rho=params_dict.get('rho', -0.3)
        )
        curve = get_us_treasury_curve()
        _model_cache[cache_key] = HullWhite2FModel(params, curve)
    
    return _model_cache[cache_key]


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """Get default model parameters and their descriptions."""
    return jsonify({
        'parameters': get_model_parameters(),
        'descriptions': get_parameter_descriptions()
    })


@app.route('/api/gpu-info', methods=['GET'])
def gpu_info():
    """Get GPU availability and information."""
    return jsonify(get_gpu_info())


@app.route('/api/initial-curve', methods=['GET'])
def initial_curve():
    """Get the initial US Treasury yield curve."""
    curve = get_us_treasury_curve()
    
    maturities = np.array([0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = curve.yield_at(maturities) * 100  # Convert to percentage
    
    return jsonify({
        'maturities': maturities.tolist(),
        'yields': yields.tolist(),
        'labels': ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    })


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Run Monte Carlo simulation.
    
    Request body:
    {
        "n_trials": 1000,
        "n_steps": 50,
        "total_time": 5.0,
        "device": "cpu" | "gpu",
        "parameters": {
            "a": 0.1,
            "b": 0.04,
            "sigma1": 0.01,
            "sigma2": 0.008,
            "rho": -0.3
        }
    }
    """
    try:
        data = request.get_json()
        
        n_trials = int(data.get('n_trials', 1000))
        n_steps = int(data.get('n_steps', 50))
        total_time = float(data.get('total_time', 5.0))
        device = data.get('device', 'cpu')
        params_dict = data.get('parameters', get_model_parameters())
        seed = data.get('seed', None)
        
        # Validate inputs
        n_trials = max(10, min(100000, n_trials))
        n_steps = max(5, min(500, n_steps))
        total_time = max(0.5, min(50.0, total_time))
        
        # Get model
        model = get_model(params_dict)
        
        # Select simulator
        if device == 'gpu':
            simulator = HullWhite2FSimulatorGPU(model)
        else:
            simulator = HullWhite2FSimulator(model)
        
        # Run simulation
        yield_maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        result = simulator.simulate(
            n_trials=n_trials,
            n_steps=n_steps,
            total_time=total_time,
            yield_maturities=yield_maturities,
            seed=seed
        )
        
        return jsonify({
            'success': True,
            'data': result.to_dict()
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare_devices():
    """
    Run simulation on both CPU and GPU and compare performance.
    
    Request body:
    {
        "n_trials": 5000,
        "n_steps": 100,
        "total_time": 10.0,
        "parameters": {...}
    }
    """
    try:
        data = request.get_json()
        
        n_trials = int(data.get('n_trials', 5000))
        n_steps = int(data.get('n_steps', 100))
        total_time = float(data.get('total_time', 10.0))
        params_dict = data.get('parameters', get_model_parameters())
        seed = data.get('seed', 42)
        
        # Validate inputs
        n_trials = max(10, min(50000, n_trials))
        n_steps = max(5, min(500, n_steps))
        total_time = max(0.5, min(50.0, total_time))
        
        model = get_model(params_dict)
        yield_maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        
        # CPU simulation
        cpu_sim = HullWhite2FSimulator(model)
        cpu_result = cpu_sim.simulate(
            n_trials=n_trials,
            n_steps=n_steps,
            total_time=total_time,
            yield_maturities=yield_maturities,
            seed=seed
        )
        
        # GPU simulation
        gpu_sim = HullWhite2FSimulatorGPU(model)
        gpu_result = gpu_sim.simulate(
            n_trials=n_trials,
            n_steps=n_steps,
            total_time=total_time,
            yield_maturities=yield_maturities,
            seed=seed
        )
        
        speedup = cpu_result.execution_time / gpu_result.execution_time if gpu_result.execution_time > 0 else 0
        
        return jsonify({
            'success': True,
            'cpu': {
                'execution_time': cpu_result.execution_time,
                'device': cpu_result.device,
                'data': cpu_result.to_dict()
            },
            'gpu': {
                'execution_time': gpu_result.execution_time,
                'device': gpu_result.device,
                'data': gpu_result.to_dict()
            },
            'speedup': speedup,
            'gpu_info': get_gpu_info()
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/yield-curve-snapshot', methods=['POST'])
def yield_curve_snapshot():
    """
    Get yield curve at specific time step from simulation results.
    
    Request body:
    {
        "time_step": 25,
        "simulation_data": {...}  // Previous simulation result
    }
    """
    try:
        data = request.get_json()
        time_step = int(data.get('time_step', 0))
        sim_data = data.get('simulation_data', {})
        
        if not sim_data:
            return jsonify({'success': False, 'error': 'No simulation data provided'})
        
        mean_curves = np.array(sim_data.get('mean_yield_curves', []))
        times = np.array(sim_data.get('times', []))
        yield_times = np.array(sim_data.get('yield_curve_times', []))
        
        if time_step >= len(mean_curves):
            time_step = len(mean_curves) - 1
        
        return jsonify({
            'success': True,
            'time': times[time_step],
            'maturities': yield_times.tolist(),
            'yields': (mean_curves[time_step] * 100).tolist()  # Convert to percentage
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Two-Factor Hull-White Model Simulation Server")
    print("="*60)
    print(f"\nGPU Info: {get_gpu_info()}")
    print("\nStarting server at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

