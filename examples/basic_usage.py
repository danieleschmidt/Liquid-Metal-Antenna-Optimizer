#!/usr/bin/env python3
"""
Basic usage example for Liquid Metal Antenna Optimizer.

This example demonstrates the core functionality of the library,
including antenna specification, optimization, and basic analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from liquid_metal_antenna import AntennaSpec, LMAOptimizer, DifferentiableFDTD
from liquid_metal_antenna.designs import ReconfigurablePatch
from liquid_metal_antenna.liquid_metal import GalinStanModel


def basic_optimization_example():
    """Example 1: Basic antenna optimization."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Antenna Optimization")
    print("=" * 60)
    
    # Define antenna specifications
    spec = AntennaSpec(
        frequency_range=(2.4e9, 2.5e9),  # 2.4-2.5 GHz (WiFi band)
        substrate='rogers_4003c',
        metal='galinstan',
        size_constraint=(30, 30, 3),  # 30x30x3 mm
        min_gain=5.0,  # dBi
        max_vswr=2.0,
        min_efficiency=0.8
    )
    
    print(f"Antenna Specification: {spec}")
    print(f"Center frequency: {spec.frequency_range.center/1e9:.2f} GHz")
    print(f"Bandwidth: {spec.frequency_range.bandwidth/1e6:.1f} MHz")
    print(f"Substrate wavelength: {spec.get_substrate_wavelength_at_center():.1f} mm")
    
    # Create optimizer
    optimizer = LMAOptimizer(
        spec=spec,
        solver='differentiable_fdtd',
        device='cpu'  # Use CPU for compatibility
    )
    
    # Run optimization
    print("\nStarting optimization...")
    result = optimizer.optimize(
        objective='max_gain',
        constraints={
            'vswr': '<2.0',
            'bandwidth': '>50e6',
            'efficiency': '>0.8'
        },
        n_iterations=200  # Reduced for quick demo
    )
    
    # Display results
    print(f"\nOptimization Results:")
    print(f"Final gain: {result.gain_dbi:.1f} dBi")
    print(f"VSWR: {result.vswr:.2f}")
    print(f"Bandwidth: {result.bandwidth_hz/1e6:.1f} MHz")
    print(f"Efficiency: {result.efficiency:.1%}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Optimization time: {result.optimization_time:.1f} seconds")
    
    # Plot convergence
    if len(result.objective_history) > 0:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(result.objective_history)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Optimization Convergence')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(result.constraint_violations)
        plt.xlabel('Iteration')
        plt.ylabel('Constraint Violation')
        plt.title('Constraint Satisfaction')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return result


def reconfigurable_patch_example():
    """Example 2: Reconfigurable patch antenna design."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Reconfigurable Patch Antenna")
    print("=" * 60)
    
    # Create reconfigurable patch antenna
    patch = ReconfigurablePatch(
        substrate_height=1.6,  # mm
        dielectric_constant=4.4,  # FR4
        n_channels=6,
        channel_width=0.8  # mm
    )
    
    print(f"Reconfigurable Patch: {patch}")
    print(f"Base resonant frequency: {patch.get_resonant_frequency()/1e9:.2f} GHz")
    print(f"Estimated bandwidth: {patch.estimate_bandwidth()/1e6:.1f} MHz")
    
    # Get predefined configurations
    states = patch.get_reconfiguration_states()
    
    print(f"\nReconfiguration States:")
    for state_name, state_info in states.items():
        print(f"  {state_name}: {state_info['description']}")
        print(f"    Target frequency: {state_info['target_frequency']/1e9:.2f} GHz")
        print(f"    Channel fill: {state_info['channel_fill']}")
    
    # Test different configurations
    frequencies = []
    configurations = []
    
    for i in range(1, patch.n_channels + 1):
        # Fill first i channels
        config = [True] * i + [False] * (patch.n_channels - i)
        patch.set_configuration(config)
        
        freq = patch.get_resonant_frequency()
        frequencies.append(freq)
        configurations.append(f"{i} channels")
        
        print(f"\n{i} channels active:")
        print(f"  Configuration: {config}")
        print(f"  Resonant frequency: {freq/1e9:.3f} GHz")
    
    # Plot frequency tuning range
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, patch.n_channels + 1), np.array(frequencies) / 1e9, 'o-')
    plt.xlabel('Number of Active Channels')
    plt.ylabel('Resonant Frequency (GHz)')
    plt.title('Frequency Reconfiguration Range')
    plt.grid(True)
    plt.show()
    
    # Export configuration
    patch.export_config('patch_config.json')
    
    return patch


def material_properties_example():
    """Example 3: Liquid metal material properties."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Liquid Metal Properties")
    print("=" * 60)
    
    # Create Galinstan model
    galinstan = GalinStanModel()
    
    print(f"Galinstan Model: {galinstan}")
    
    # Temperature range analysis
    temperatures = np.linspace(20, 80, 61)
    conductivities = galinstan.conductivity(temperatures)
    viscosities = galinstan.viscosity(temperatures)
    densities = galinstan.density(temperatures)
    
    # Properties at room temperature
    room_temp = 25.0
    props = galinstan.get_properties_dict(room_temp)
    
    print(f"\nProperties at {room_temp}°C:")
    for key, value in props.items():
        if isinstance(value, float):
            if 'conductivity' in key:
                print(f"  {key}: {value/1e6:.2f} MS/m")
            elif 'viscosity' in key:
                print(f"  {key}: {value*1000:.2f} mPa·s")
            elif 'density' in key:
                print(f"  {key}: {value:.0f} kg/m³")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Frequency-dependent properties
    frequencies = np.array([1e9, 2.45e9, 5.8e9, 10e9])  # 1, 2.45, 5.8, 10 GHz
    
    print(f"\nSkin depth at different frequencies:")
    for freq in frequencies:
        skin_depth = galinstan.skin_depth(freq, room_temp)
        print(f"  {freq/1e9:.1f} GHz: {skin_depth*1e6:.2f} μm")
    
    # Flow properties
    channel_diameter = 1e-3  # 1mm
    flow_velocity = 0.01  # 1 cm/s
    
    re_number = galinstan.reynolds_number(flow_velocity, channel_diameter, room_temp)
    print(f"\nFlow Analysis (v={flow_velocity*100:.0f} cm/s, D={channel_diameter*1000:.0f} mm):")
    print(f"  Reynolds number: {re_number:.1f}")
    print(f"  Flow regime: {'Laminar' if re_number < 2300 else 'Turbulent'}")
    
    # Plot temperature dependence
    galinstan.plot_temperature_dependence()
    
    return galinstan


def solver_comparison_example():
    """Example 4: FDTD solver demonstration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: FDTD Solver Demonstration")
    print("=" * 60)
    
    # Create simple antenna geometry
    geometry_size = (32, 32, 16)  # Small for quick simulation
    geometry = np.zeros(geometry_size)
    
    # Add simple patch antenna
    patch_size = 8
    center_x, center_y = geometry_size[0] // 2, geometry_size[1] // 2
    patch_z = geometry_size[2] - 2
    
    geometry[
        center_x - patch_size // 2 : center_x + patch_size // 2,
        center_y - patch_size // 2 : center_y + patch_size // 2,
        patch_z
    ] = 1.0
    
    print(f"Created test geometry: {geometry_size}")
    print(f"Patch size: {patch_size}x{patch_size} cells")
    print(f"Total conductor cells: {np.sum(geometry):.0f}")
    
    # Create solver
    solver = DifferentiableFDTD(
        resolution=1.0e-3,  # 1mm resolution
        precision='float32'
    )
    
    print(f"FDTD Solver: resolution={solver.resolution*1000:.1f}mm")
    print(f"Grid size: {solver.grid_size}")
    print(f"Estimated memory: {solver.estimate_memory_usage():.3f} GB")
    
    # Run simulation
    frequency = 2.45e9
    print(f"\nRunning FDTD simulation at {frequency/1e9:.2f} GHz...")
    
    result = solver.simulate(
        geometry=geometry,
        frequency=frequency,
        compute_gradients=False,
        max_time_steps=500  # Reduced for quick demo
    )
    
    print(f"Simulation Results:")
    print(f"  Gain: {result.gain_dbi:.1f} dBi")
    print(f"  VSWR: {result.get_vswr_at_frequency(frequency):.2f}")
    print(f"  Computation time: {result.computation_time:.2f} seconds")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    
    # Plot radiation pattern
    if result.radiation_pattern is not None:
        theta = result.theta_angles
        pattern = result.radiation_pattern
        
        # Main cut (phi=0)
        main_cut = pattern[:, 0] if pattern.ndim > 1 else pattern
        pattern_db = 10 * np.log10(np.maximum(main_cut, 1e-10))
        pattern_db = pattern_db - np.max(pattern_db)  # Normalize
        
        plt.figure(figsize=(8, 6))
        plt.subplot(111, projection='polar')
        plt.plot(theta, pattern_db)
        plt.ylim([-40, 0])
        plt.title(f'Radiation Pattern (Gain: {result.gain_dbi:.1f} dBi)')
        plt.show()
    
    return result


def main():
    """Run all examples."""
    print("Liquid Metal Antenna Optimizer - Basic Usage Examples")
    print("====================================================")
    
    try:
        # Example 1: Basic optimization
        opt_result = basic_optimization_example()
        
        # Example 2: Reconfigurable patch
        patch = reconfigurable_patch_example()
        
        # Example 3: Material properties
        galinstan = material_properties_example()
        
        # Example 4: FDTD solver
        sim_result = solver_comparison_example()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return {
            'optimization_result': opt_result,
            'patch_antenna': patch,
            'material_model': galinstan,
            'simulation_result': sim_result
        }
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()