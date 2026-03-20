# Liquid Metal Antenna Optimizer

A Python toolkit for designing and optimising reconfigurable patch antennas with liquid metal microfluidic channels, combining analytical EM modelling with a neural network surrogate and evolutionary optimisation.

## Features

| Component | Description |
|-----------|-------------|
| `AntennaGeometry` | Parameterised patch antenna (length, width, channel positions) |
| `EMSolver` | Hammerstad fringe-corrected resonant frequency + Lorentzian S11 |
| `NeuralSurrogate` | 2-layer MLP (numpy-only) for fast frequency prediction |
| `DifferentiableOptimizer` | Simplified CMA-ES optimiser using surrogate model |

## Quick Start

```bash
pip install numpy pytest
python demo.py
```

### Demo output (example)

```
Liquid Metal Antenna Optimizer — 2.4 GHz WiFi Demo
Initial geometry : AntennaGeometry(length=0.0312, width=0.0400, channels=0)
Initial resonant frequency : 3.7142 GHz
Running CMA-ES optimisation (50 iterations)…
Best geometry found : AntennaGeometry(length=0.0624, width=0.0400, channels=0)
Resonant frequency  : 2.4001 GHz
Error from target   : 0.01 MHz
```

## Usage

```python
from liquid_metal_antenna import AntennaGeometry, EMSolver, DifferentiableOptimizer

# Define a patch antenna
geom = AntennaGeometry(length=0.031, width=0.040, channel_positions=[(0.01, 0.02)])

# Query EM properties
solver = EMSolver(substrate_height=1.6e-3)
f0 = solver.resonant_frequency(geom)
print(f"Resonant frequency: {f0/1e9:.3f} GHz")

# Optimise toward a target frequency
opt = DifferentiableOptimizer(em_solver=solver)
best = opt.optimize(target_freq=2.4e9, n_iterations=50)
print(best)
```

## Physics

The resonant frequency is computed using the Hammerstad fringe-field correction:

```
ΔL = 0.824 · h · (w + 0.264·h) / (w − 0.8·h)
L_eff = L + ΔL
f₀ = c / (2 · L_eff)
```

S11 is modelled as a Lorentzian dip at resonance:

```
Γ(f) = 1 − γ² / ((f − f₀)² + γ²),  γ = f₀ / Q
S11(dB) = 20 log₁₀ |Γ(f)|
```

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

## Architecture

```
liquid_metal_antenna/
├── __init__.py                  # public exports
├── antenna_geometry.py          # geometry parameterisation
├── em_solver.py                 # analytical EM model
├── neural_surrogate.py          # MLP surrogate (numpy)
└── differentiable_optimizer.py  # CMA-ES optimiser
tests/
└── test_antenna_optimizer.py    # 14 unit/integration tests
demo.py                          # 2.4 GHz WiFi demo
```

## Dependencies

- **numpy** — all numerical computation
- **pytest** — test runner (dev dependency)

## License

MIT
