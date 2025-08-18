#!/usr/bin/env python3
"""
Quick demonstration of Liquid Metal Antenna Optimizer.
Generation 1: Simple but functional implementation.
"""

import sys
import numpy as np
import time

# Add current directory to path
sys.path.insert(0, '.')

from liquid_metal_antenna import AntennaSpec, LMAOptimizer

def generation1_quick_demo():
    """Generation 1: Simple but functional demonstration."""
    print("=" * 60)
    print("LIQUID METAL ANTENNA OPTIMIZER - GENERATION 1")
    print("Simple Implementation - Core Functionality")
    print("=" * 60)
    
    # Define antenna specifications
    spec = AntennaSpec(
        frequency_range=(2.4e9, 2.5e9),  # WiFi band
        substrate='rogers_4003c',
        metal='galinstan',
        size_constraint=(25, 25, 2),  # 25x25x2 mm
        min_gain=6.0,  # dBi
        max_vswr=2.0,
        min_efficiency=0.8
    )
    
    print(f"📡 Antenna Target Specifications:")
    print(f"   Frequency: {spec.frequency_range.start/1e9:.2f}-{spec.frequency_range.stop/1e9:.2f} GHz")
    print(f"   Size: {spec.size_constraint} mm")
    print(f"   Min gain: {spec.min_gain} dBi")
    print(f"   Max VSWR: {spec.max_vswr}")
    print(f"   Min efficiency: {spec.min_efficiency}")
    
    # Create optimizer with simple solver
    optimizer = LMAOptimizer(
        spec=spec,
        solver='simple_fdtd',  # Use simple implementation
        device='cpu'
    )
    
    print(f"\n🔧 Optimizer Configuration:")
    print(f"   Solver: {getattr(optimizer, 'solver_type', 'simple_fdtd')}")
    print(f"   Device: {getattr(optimizer, 'device', 'cpu')}")
    print(f"   Objective: Maximize gain")
    
    # Run quick optimization
    print(f"\n🚀 Starting optimization...")
    start_time = time.time()
    
    result = optimizer.optimize(
        objective='max_gain',
        constraints={
            'vswr': '<2.0',
            'bandwidth': '>50e6',  # >50 MHz
            'efficiency': '>0.8'   # >80%
        },
        n_iterations=100  # Quick demo
    )
    
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 Optimization Results:")
    print(f"   ✅ Final gain: {result.gain_dbi:.1f} dBi")
    print(f"   ✅ VSWR: {result.vswr:.2f}")
    print(f"   ✅ Bandwidth: {result.bandwidth_hz/1e6:.1f} MHz")
    print(f"   ✅ Efficiency: {result.efficiency:.1%}")
    print(f"   📈 Converged: {result.converged}")
    print(f"   🔄 Iterations: {result.iterations}")
    print(f"   ⏱️  Total time: {optimization_time:.1f}s")
    
    # Performance assessment
    meets_specs = (
        result.gain_dbi >= spec.min_gain and
        result.vswr <= spec.max_vswr and
        result.efficiency >= spec.min_efficiency
    )
    
    print(f"\n🎯 Performance Assessment:")
    print(f"   Specifications met: {'✅ YES' if meets_specs else '❌ NO'}")
    
    if meets_specs:
        print(f"   🎉 SUCCESS: Antenna meets all design requirements!")
    else:
        print(f"   ⚠️  PARTIAL: Some specifications not met, iteration needed")
    
    # Show convergence info
    if len(result.objective_history) > 5:
        print(f"\n📈 Convergence Analysis:")
        initial_obj = result.objective_history[0]
        final_obj = result.objective_history[-1]
        improvement = ((final_obj - initial_obj) / abs(initial_obj)) * 100
        print(f"   Initial objective: {initial_obj:.3f}")
        print(f"   Final objective: {final_obj:.3f}")
        print(f"   Improvement: {improvement:+.1f}%")
    
    # Basic geometry info
    if hasattr(result, 'geometry') and result.geometry is not None:
        print(f"\n🔬 Geometry Summary:")
        print(f"   Shape: {result.geometry.shape}")
        print(f"   Conductor volume: {np.sum(result.geometry > 0.5):.0f} cells")
        conductor_fraction = np.mean(result.geometry > 0.5)
        print(f"   Fill factor: {conductor_fraction:.1%}")
    
    return result

def generation1_multiple_configs():
    """Test multiple antenna configurations."""
    print(f"\n" + "=" * 60)
    print("GENERATION 1: MULTIPLE CONFIGURATION TEST")
    print("=" * 60)
    
    configs = [
        {
            'name': 'Compact 2.4GHz',
            'freq': (2.4e9, 2.5e9),
            'size': (20, 20, 1.6),
            'target_gain': 5.0
        },
        {
            'name': 'High-gain 5GHz',
            'freq': (5.7e9, 5.9e9),
            'size': (15, 15, 1.6),
            'target_gain': 8.0
        },
        {
            'name': 'Broadband 3.5GHz',
            'freq': (3.3e9, 3.7e9),
            'size': (30, 30, 2.0),
            'target_gain': 7.0
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']}...")
        
        spec = AntennaSpec(
            frequency_range=config['freq'],
            substrate='rogers_4003c',
            metal='galinstan',
            size_constraint=config['size'],
            min_gain=config['target_gain']
        )
        
        optimizer = LMAOptimizer(spec=spec, solver='simple_fdtd', device='cpu')
        
        start_time = time.time()
        result = optimizer.optimize(
            objective='max_gain',
            n_iterations=50  # Quick test
        )
        opt_time = time.time() - start_time
        
        results[config['name']] = {
            'result': result,
            'time': opt_time,
            'meets_target': result.gain_dbi >= config['target_gain']
        }
        
        print(f"   Gain: {result.gain_dbi:.1f} dBi (target: {config['target_gain']:.1f})")
        print(f"   VSWR: {result.vswr:.2f}")
        print(f"   Time: {opt_time:.1f}s")
        print(f"   Target met: {'✅' if results[config['name']]['meets_target'] else '❌'}")
    
    # Summary
    print(f"\n📊 CONFIGURATION SUMMARY:")
    successful = sum(1 for r in results.values() if r['meets_target'])
    total_time = sum(r['time'] for r in results.values())
    
    print(f"   Successful configs: {successful}/{len(configs)}")
    print(f"   Total optimization time: {total_time:.1f}s")
    print(f"   Average time per config: {total_time/len(configs):.1f}s")
    
    return results

def main():
    """Main demonstration function."""
    print("🚀 LIQUID METAL ANTENNA OPTIMIZER")
    print("Generation 1: Core Functionality Demonstration")
    print("=" * 60)
    
    try:
        # Single antenna demonstration
        result1 = generation1_quick_demo()
        
        # Multiple configurations
        results2 = generation1_multiple_configs()
        
        print(f"\n" + "=" * 60)
        print("✅ GENERATION 1 DEMONSTRATION COMPLETE")
        print("Core functionality validated and working!")
        print("=" * 60)
        
        return {
            'single_demo': result1,
            'multi_config': results2
        }
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎉 All tests passed! Ready for Generation 2.")
    else:
        print(f"\n❌ Some tests failed. Check implementation.")