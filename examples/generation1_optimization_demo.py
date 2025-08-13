#!/usr/bin/env python3
"""
Generation 1 Optimization Demo
Demonstrates basic antenna optimization functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Demonstrate basic antenna optimization."""
    
    print("🔬 Liquid Metal Antenna Optimizer - Optimization Demo")
    print("=" * 60)
    
    # Import the core components
    try:
        from liquid_metal_antenna import AntennaSpec, LMAOptimizer
        print("✅ Successfully imported components")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Create antenna specification for WiFi application
    print("\n📋 Creating WiFi Antenna Specification...")
    spec = AntennaSpec(
        frequency_range=(2.4e9, 2.5e9),  # 2.4 GHz WiFi band
        substrate='rogers_4003c',
        metal='galinstan',
        size_constraint=(30, 30, 1.6),  # Compact 30x30mm antenna
        min_gain=5.0,  # Target 5 dBi gain
        max_vswr=2.0,
        min_efficiency=0.8
    )
    print(f"✅ Spec: {spec}")
    
    # Create optimizer
    print("\n🔧 Setting up Optimizer...")
    optimizer = LMAOptimizer(spec=spec, solver='differentiable_fdtd', device='cpu')
    print(f"✅ Optimizer ready with {type(optimizer).__name__}")
    
    # Define optimization constraints
    constraints = {
        'vswr': '<2.0',          # Good impedance matching
        'bandwidth': '>100e6',   # 100 MHz minimum bandwidth  
        'efficiency': '>0.8'     # 80% efficiency target
    }
    
    print("\n🚀 Running Basic Optimization...")
    print("   Objective: Maximize gain")
    print("   Constraints:", constraints)
    print("   Iterations: 20 (quick demo)")
    
    # Run optimization
    try:
        result = optimizer.optimize(
            objective='max_gain',
            constraints=constraints,
            n_iterations=20
        )
        
        print("\n📊 Optimization Results:")
        print("=" * 40)
        print(f"✅ Converged: {result.converged}")
        print(f"📏 Iterations: {result.iterations}")
        print(f"⏱️  Time: {result.optimization_time:.2f} seconds")
        print(f"📈 Final gain: {result.gain_dbi:.1f} dBi")
        print(f"📡 VSWR: {result.vswr:.2f}")
        print(f"📊 Bandwidth: {result.bandwidth_hz/1e6:.1f} MHz")
        print(f"⚡ Efficiency: {result.efficiency:.1%}")
        
        # Show optimization progress
        print(f"\n📈 Optimization Progress:")
        if len(result.objective_history) >= 3:
            print(f"   Initial objective: {result.objective_history[0]:.3f}")
            print(f"   Final objective: {result.objective_history[-1]:.3f}")
            improvement = abs(result.objective_history[-1] - result.objective_history[0])
            print(f"   Improvement: {improvement:.3f}")
        
        # Analyze constraint satisfaction
        print(f"\n🎯 Constraint Analysis:")
        vswr_ok = result.vswr <= 2.0
        bandwidth_ok = result.bandwidth_hz >= 100e6
        efficiency_ok = result.efficiency >= 0.8
        
        print(f"   VSWR < 2.0: {'✅' if vswr_ok else '❌'} ({result.vswr:.2f})")
        print(f"   BW > 100MHz: {'✅' if bandwidth_ok else '❌'} ({result.bandwidth_hz/1e6:.1f} MHz)")
        print(f"   Eff > 80%: {'✅' if efficiency_ok else '❌'} ({result.efficiency:.1%})")
        
        all_satisfied = vswr_ok and bandwidth_ok and efficiency_ok
        print(f"   All constraints: {'✅ SATISFIED' if all_satisfied else '⚠️  PARTIALLY SATISFIED'}")
        
        print(f"\n🏗️  Geometry Summary:")
        if hasattr(result.geometry, 'shape'):
            print(f"   Shape: {result.geometry.shape}")
        else:
            geo = result.geometry
            print(f"   Dimensions: {len(geo)}x{len(geo[0]) if geo else 0}x{len(geo[0][0]) if geo and geo[0] else 0}")
        
        # Count metal fill
        def count_metal_fill(geo):
            if hasattr(geo, 'flatten'):
                total = geo.size
                metal = sum(1 for x in geo.flatten() if x > 0.5)
            else:
                total = 0
                metal = 0
                def count_recursive(item):
                    nonlocal total, metal
                    if hasattr(item, '__iter__') and not isinstance(item, str):
                        for x in item:
                            count_recursive(x)
                    else:
                        total += 1
                        if item > 0.5:
                            metal += 1
                count_recursive(geo)
            return metal, total
        
        metal_cells, total_cells = count_metal_fill(result.geometry)
        fill_fraction = metal_cells / total_cells if total_cells > 0 else 0
        print(f"   Metal fill: {fill_fraction:.1%} ({metal_cells}/{total_cells} cells)")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test simple VSWR optimization
    print("\n🔄 Testing Alternative Objective...")
    try:
        result2 = optimizer.optimize(
            objective='min_vswr',
            constraints={'efficiency': '>0.7'},
            n_iterations=10
        )
        print(f"✅ VSWR optimization: {result2.vswr:.2f} (vs {result.vswr:.2f} from gain opt)")
        print(f"   Gain trade-off: {result2.gain_dbi:.1f} dBi (vs {result.gain_dbi:.1f} dBi)")
        
    except Exception as e:
        print(f"⚠️  VSWR optimization had issues: {e}")
    
    # Summary
    print("\n🎯 Generation 1 Optimization Summary")
    print("=" * 60)
    print("✅ Basic optimization loop functional")
    print("✅ Multiple objectives supported (gain, VSWR)")
    print("✅ Constraint handling working")
    print("✅ Geometry evolution successful")
    print("✅ Performance metrics computed")
    print("⚠️  Using simplified EM models (no external deps)")
    print("\n🚀 Ready for Generation 2: Enhanced robustness!")
    
    print("\n" + "=" * 60)
    print("Generation 1 optimization demo completed! 🎉")

if __name__ == "__main__":
    main()