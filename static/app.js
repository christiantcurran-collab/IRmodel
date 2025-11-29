/**
 * Hull-White 2F Simulator - Frontend Application
 * Handles UI interactions, API calls, and chart rendering
 */

// Global state
let simulationData = null;
let currentTimeStep = 0;
let isPlaying = false;
let playInterval = null;
let charts = {};

// Chart.js color scheme
const chartColors = {
    primary: '#00d9a5',
    secondary: '#00b4d8',
    tertiary: '#7c3aed',
    warning: '#f59e0b',
    grid: 'rgba(255, 255, 255, 0.05)',
    text: '#8b949e',
    textLight: '#e6edf3'
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    checkGPUStatus();
    loadInitialCurve();
    setupEventListeners();
});

/**
 * Initialize Chart.js instances
 */
function initializeCharts() {
    // Common chart options
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    color: chartColors.text,
                    font: { family: 'Outfit', size: 12 },
                    padding: 15,
                    usePointStyle: true
                }
            }
        },
        scales: {
            x: {
                grid: { color: chartColors.grid },
                ticks: { color: chartColors.text, font: { family: 'JetBrains Mono', size: 11 } }
            },
            y: {
                grid: { color: chartColors.grid },
                ticks: { color: chartColors.text, font: { family: 'JetBrains Mono', size: 11 } }
            }
        }
    };

    // Yield Curve Chart
    const yieldCtx = document.getElementById('yieldCurveChart').getContext('2d');
    charts.yieldCurve = new Chart(yieldCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Yield Curve',
                data: [],
                borderColor: chartColors.primary,
                backgroundColor: 'rgba(0, 217, 165, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: 'Initial Curve',
                data: [],
                borderColor: chartColors.text,
                borderDash: [5, 5],
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                x: {
                    ...commonOptions.scales.x,
                    title: { display: true, text: 'Maturity (Years)', color: chartColors.text }
                },
                y: {
                    ...commonOptions.scales.y,
                    title: { display: true, text: 'Yield (%)', color: chartColors.text }
                }
            }
        }
    });

    // Short Rate Chart
    const rateCtx = document.getElementById('shortRateChart').getContext('2d');
    charts.shortRate = new Chart(rateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Mean Short Rate',
                data: [],
                borderColor: chartColors.primary,
                backgroundColor: 'rgba(0, 217, 165, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0
            }, {
                label: '+/- 1 Std Dev',
                data: [],
                borderColor: 'rgba(0, 217, 165, 0.3)',
                backgroundColor: 'rgba(0, 217, 165, 0.05)',
                fill: '+1',
                tension: 0.3,
                pointRadius: 0
            }, {
                label: '',
                data: [],
                borderColor: 'rgba(0, 217, 165, 0.3)',
                fill: false,
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                legend: {
                    ...commonOptions.plugins.legend,
                    labels: {
                        ...commonOptions.plugins.legend.labels,
                        filter: (item) => item.text !== ''
                    }
                }
            },
            scales: {
                ...commonOptions.scales,
                x: {
                    ...commonOptions.scales.x,
                    title: { display: true, text: 'Time (Years)', color: chartColors.text }
                },
                y: {
                    ...commonOptions.scales.y,
                    title: { display: true, text: 'Short Rate (%)', color: chartColors.text }
                }
            }
        }
    });

    // Surface Chart (simplified 2D heatmap representation)
    const surfaceCtx = document.getElementById('surfaceChart').getContext('2d');
    charts.surface = new Chart(surfaceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Yield curves over time (each line = different time step)',
                    color: chartColors.text,
                    font: { size: 12 }
                }
            },
            scales: {
                ...commonOptions.scales,
                x: {
                    ...commonOptions.scales.x,
                    title: { display: true, text: 'Maturity (Years)', color: chartColors.text }
                },
                y: {
                    ...commonOptions.scales.y,
                    title: { display: true, text: 'Yield (%)', color: chartColors.text }
                }
            }
        }
    });
}

/**
 * Check GPU availability
 */
async function checkGPUStatus() {
    try {
        const response = await fetch('/api/gpu-info');
        const gpuInfo = await response.json();
        
        const statusEl = document.getElementById('gpuStatus');
        const indicator = statusEl.querySelector('.status-indicator');
        const text = statusEl.querySelector('.status-text');
        
        if (gpuInfo.available) {
            indicator.classList.add('available');
            text.textContent = `GPU: ${gpuInfo.name || 'Available'}`;
        } else {
            indicator.classList.add('unavailable');
            text.textContent = 'GPU: Not Available';
        }
    } catch (error) {
        console.error('Failed to check GPU status:', error);
    }
}

/**
 * Load and display initial yield curve
 */
async function loadInitialCurve() {
    try {
        const response = await fetch('/api/initial-curve');
        const data = await response.json();
        
        charts.yieldCurve.data.labels = data.labels;
        charts.yieldCurve.data.datasets[0].data = data.yields;
        charts.yieldCurve.data.datasets[1].data = data.yields;
        charts.yieldCurve.update();
    } catch (error) {
        console.error('Failed to load initial curve:', error);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Device toggle
    document.querySelectorAll('.device-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.device-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Time slider
    document.getElementById('timeSlider').addEventListener('input', (e) => {
        if (simulationData) {
            currentTimeStep = parseInt(e.target.value);
            updateVisualization();
        }
    });
}

/**
 * Toggle collapsible section
 */
function toggleSection(element) {
    const section = element.closest('.panel-section');
    section.classList.toggle('collapsed');
}

/**
 * Get current parameters from UI
 */
function getParameters() {
    return {
        a: parseFloat(document.getElementById('paramA').value),
        b: parseFloat(document.getElementById('paramB').value),
        sigma1: parseFloat(document.getElementById('sigma1').value),
        sigma2: parseFloat(document.getElementById('sigma2').value),
        rho: parseFloat(document.getElementById('rho').value)
    };
}

/**
 * Get simulation settings from UI
 */
function getSimulationSettings() {
    return {
        n_trials: parseInt(document.getElementById('nTrials').value),
        n_steps: parseInt(document.getElementById('nSteps').value),
        total_time: parseFloat(document.getElementById('totalTime').value),
        device: document.querySelector('.device-btn.active').dataset.device,
        parameters: getParameters()
    };
}

/**
 * Show loading overlay
 */
function showLoading(message = 'Running simulation...') {
    const overlay = document.getElementById('loadingOverlay');
    overlay.querySelector('.loading-text').textContent = message;
    overlay.classList.add('active');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

/**
 * Run simulation
 */
async function runSimulation() {
    showLoading();
    
    try {
        const settings = getSimulationSettings();
        
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        
        const result = await response.json();
        
        if (result.success) {
            simulationData = result.data;
            currentTimeStep = 0;
            
            // Show time control
            document.getElementById('timeControl').style.display = 'flex';
            document.getElementById('statsPanel').style.display = 'flex';
            
            // Setup time slider
            const slider = document.getElementById('timeSlider');
            slider.max = simulationData.n_steps;
            slider.value = 0;
            
            // Update stats
            updateStats();
            
            // Update charts
            updateVisualization();
            updateShortRateChart();
            updateSurfaceChart();
            
        } else {
            alert('Simulation failed: ' + result.error);
        }
    } catch (error) {
        console.error('Simulation error:', error);
        alert('Simulation failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Update visualization for current time step
 */
function updateVisualization() {
    if (!simulationData) return;
    
    const time = simulationData.times[currentTimeStep];
    document.getElementById('currentTime').textContent = time.toFixed(2);
    
    // Update yield curve
    const yields = simulationData.mean_yield_curves[currentTimeStep].map(y => y * 100);
    const maturities = simulationData.yield_curve_times;
    
    charts.yieldCurve.data.labels = maturities.map(m => m.toString());
    charts.yieldCurve.data.datasets[0].data = yields;
    
    // Color based on time
    const hue = 160 - (currentTimeStep / simulationData.n_steps) * 60; // Green to teal
    charts.yieldCurve.data.datasets[0].borderColor = `hsl(${hue}, 80%, 50%)`;
    charts.yieldCurve.data.datasets[0].backgroundColor = `hsla(${hue}, 80%, 50%, 0.1)`;
    
    document.getElementById('yieldCurveSubtitle').textContent = `t = ${time.toFixed(2)} years`;
    
    charts.yieldCurve.update('none');
}

/**
 * Update short rate chart with full simulation data
 */
function updateShortRateChart() {
    if (!simulationData) return;
    
    const times = simulationData.times;
    const meanRates = simulationData.mean_short_rate.map(r => r * 100);
    const stdRates = simulationData.std_short_rate.map(r => r * 100);
    
    const upperBand = meanRates.map((m, i) => m + stdRates[i]);
    const lowerBand = meanRates.map((m, i) => m - stdRates[i]);
    
    charts.shortRate.data.labels = times.map(t => t.toFixed(1));
    charts.shortRate.data.datasets[0].data = meanRates;
    charts.shortRate.data.datasets[1].data = upperBand;
    charts.shortRate.data.datasets[2].data = lowerBand;
    
    // Add sample paths
    if (simulationData.sample_short_rates) {
        // Remove old sample path datasets
        while (charts.shortRate.data.datasets.length > 3) {
            charts.shortRate.data.datasets.pop();
        }
        
        // Add a few sample paths
        const numPaths = Math.min(5, simulationData.sample_short_rates.length);
        for (let i = 0; i < numPaths; i++) {
            const pathData = simulationData.sample_short_rates[i].map(r => r * 100);
            const alpha = 0.3 - (i * 0.05);
            charts.shortRate.data.datasets.push({
                label: '',
                data: pathData,
                borderColor: `rgba(0, 180, 216, ${alpha})`,
                borderWidth: 1,
                fill: false,
                tension: 0.1,
                pointRadius: 0
            });
        }
    }
    
    document.getElementById('shortRateSubtitle').textContent = 
        `${simulationData.n_trials.toLocaleString()} trials, ${simulationData.device.toUpperCase()}`;
    
    charts.shortRate.update();
}

/**
 * Update surface chart with yield curves over time
 */
function updateSurfaceChart() {
    if (!simulationData) return;
    
    const maturities = simulationData.yield_curve_times;
    const nSteps = simulationData.n_steps;
    
    // Sample every nth time step
    const step = Math.max(1, Math.floor(nSteps / 10));
    const datasets = [];
    
    for (let i = 0; i <= nSteps; i += step) {
        const time = simulationData.times[i];
        const yields = simulationData.mean_yield_curves[i].map(y => y * 100);
        
        // Color gradient from green to blue over time
        const progress = i / nSteps;
        const hue = 160 - progress * 40;
        const saturation = 70 + progress * 10;
        
        datasets.push({
            label: `t=${time.toFixed(1)}y`,
            data: yields,
            borderColor: `hsl(${hue}, ${saturation}%, 50%)`,
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.4,
            pointRadius: 0
        });
    }
    
    charts.surface.data.labels = maturities.map(m => m.toString());
    charts.surface.data.datasets = datasets;
    charts.surface.update();
}

/**
 * Update stats panel
 */
function updateStats() {
    if (!simulationData) return;
    
    document.getElementById('execTime').textContent = 
        simulationData.execution_time.toFixed(3) + 's';
    document.getElementById('deviceUsed').textContent = 
        simulationData.device.toUpperCase();
    document.getElementById('trialsRun').textContent = 
        simulationData.n_trials.toLocaleString();
    document.getElementById('finalRate').textContent = 
        (simulationData.mean_short_rate[simulationData.n_steps] * 100).toFixed(2) + '%';
}

/**
 * Toggle playback of time evolution
 */
function togglePlayback() {
    const btn = document.getElementById('playBtn');
    
    if (isPlaying) {
        clearInterval(playInterval);
        isPlaying = false;
        btn.classList.remove('playing');
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
    } else {
        isPlaying = true;
        btn.classList.add('playing');
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
        
        playInterval = setInterval(() => {
            currentTimeStep++;
            if (currentTimeStep > simulationData.n_steps) {
                currentTimeStep = 0;
            }
            document.getElementById('timeSlider').value = currentTimeStep;
            updateVisualization();
        }, 100);
    }
}

/**
 * Run CPU/GPU comparison
 */
async function runComparison() {
    showLoading('Running CPU vs GPU comparison...');
    
    try {
        const settings = getSimulationSettings();
        
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update modal with results
            document.getElementById('cpuTime').textContent = 
                result.cpu.execution_time.toFixed(3) + 's';
            document.getElementById('cpuDevice').textContent = 
                result.cpu.device;
            
            document.getElementById('gpuTime').textContent = 
                result.gpu.execution_time.toFixed(3) + 's';
            document.getElementById('gpuDevice').textContent = 
                result.gpu.device;
            
            document.getElementById('speedupValue').textContent = 
                result.speedup.toFixed(2) + 'x';
            
            // Show modal
            document.getElementById('comparisonModal').classList.add('active');
            
            // Update main visualization with GPU results
            simulationData = result.gpu.data;
            currentTimeStep = 0;
            
            document.getElementById('timeControl').style.display = 'flex';
            document.getElementById('statsPanel').style.display = 'flex';
            
            const slider = document.getElementById('timeSlider');
            slider.max = simulationData.n_steps;
            slider.value = 0;
            
            updateStats();
            updateVisualization();
            updateShortRateChart();
            updateSurfaceChart();
            
        } else {
            alert('Comparison failed: ' + result.error);
        }
    } catch (error) {
        console.error('Comparison error:', error);
        alert('Comparison failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Close modal
 */
function closeModal() {
    document.getElementById('comparisonModal').classList.remove('active');
}

// Close modal on outside click
document.getElementById('comparisonModal').addEventListener('click', (e) => {
    if (e.target.id === 'comparisonModal') {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

