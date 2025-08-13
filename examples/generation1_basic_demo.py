#!/usr/bin/env python3
"""
Generation 1 Basic Demo - Liquid Metal Antenna Optimizer
Demonstrates core functionality without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Demonstrate basic antenna specification and optimizer setup."""
    
    print("🚀 Liquid Metal Antenna Optimizer - Generation 1 Demo")
    print("=" * 60)
    
    # Import the core components
    try:
        from liquid_metal_antenna import AntennaSpec, LMAOptimizer
        print("✅ Successfully imported core components")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Create antenna specification
    print("\n📋 Creating Antenna Specification...")
    try:
        spec = AntennaSpec(
            frequency_range=(2.4e9, 5.8e9),  # 2.4-5.8 GHz WiFi band
            substrate='rogers_4003c',
            metal='galinstan',
            size_constraint=(50, 50, 3),  # 50x50x3 mm
            min_gain=5.0,  # 5 dBi minimum
            max_vswr=2.0,
            min_efficiency=0.8
        )
        print(f"✅ Antenna spec created: {spec}")
        
        # Display key properties
        print(f"   Center frequency: {spec.frequency_range.center/1e9:.2f} GHz")
        print(f"   Bandwidth: {spec.frequency_range.bandwidth/1e6:.0f} MHz")
        print(f"   Fractional BW: {spec.frequency_range.fractional_bandwidth:.1%}")
        print(f"   Free-space wavelength: {spec.get_wavelength_at_center():.1f} mm")
        print(f"   Substrate wavelength: {spec.get_substrate_wavelength_at_center():.1f} mm")
        print(f"   Electrically small: {spec.is_electrically_small()}")
        print(f"   Liquid metal conductivity: {spec.get_liquid_metal_conductivity():.2e} S/m")
        
    except Exception as e:
        print(f"❌ Antenna spec creation failed: {e}")
        return
    
    # Create optimizer
    print("\n🔧 Creating Optimizer...")
    try:
        # Check if torch is available for full implementation
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"✅ PyTorch available, using device: {device}")
            has_torch = True
        except ImportError:
            print("⚠️  PyTorch not available, using fallback implementation")
            device = 'cpu'
            has_torch = False
        
        optimizer = LMAOptimizer(
            spec=spec,
            solver='differentiable_fdtd',
            device=device
        )
        print("✅ Optimizer created successfully")
        
        # Display optimizer configuration
        print(f"   Solver: {type(optimizer.solver).__name__}")
        print(f"   Device: {device}")
        print(f"   Max iterations: {optimizer.max_iterations}")
        if hasattr(optimizer, 'learning_rate'):
            print(f"   Learning rate: {optimizer.learning_rate}")
        else:
            print(f"   Step size: {getattr(optimizer, 'step_size', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        return
    
    # Test basic geometry creation
    print("\n🏗️  Testing Geometry Creation...")
    try:
        initial_geometry = optimizer.create_initial_geometry(spec)
        if hasattr(initial_geometry, 'shape'):
            print(f"✅ Initial geometry created: {initial_geometry.shape}")
        else:
            print(f"✅ Initial geometry created: {len(initial_geometry)}x{len(initial_geometry[0]) if initial_geometry else 0}x{len(initial_geometry[0][0]) if initial_geometry and initial_geometry[0] else 0}")
        print(f"   Geometry type: {type(initial_geometry)}")
        
        # Count non-zero elements
        def count_nonzero(geo):
            count = 0
            if hasattr(geo, 'flatten'):
                return sum(1 for x in geo.flatten() if x > 0)
            elif hasattr(geo, '__iter__'):
                for item in geo:
                    if hasattr(item, '__iter__') and not isinstance(item, str):
                        count += count_nonzero(item)
                    elif item > 0:
                        count += 1
            return count
        
        print(f"   Non-zero elements: {count_nonzero(initial_geometry)}")
        
    except Exception as e:
        print(f"❌ Geometry creation failed: {e}")
        return
    
    # Test specification serialization
    print("\n💾 Testing Specification Serialization...")
    try:
        spec_dict = spec.to_dict()
        print("✅ Specification serialized to dictionary")
        
        spec_restored = AntennaSpec.from_dict(spec_dict)
        print("✅ Specification restored from dictionary")
        print(f"   Specs match: {spec_restored.frequency_range.center == spec.frequency_range.center}")
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return
    
    # Summary
    print("\n🎯 Generation 1 Summary")
    print("=" * 60)
    print("✅ Core antenna specification system working")
    print("✅ Optimizer initialization successful")
    print("✅ Basic geometry creation functional")
    print("✅ Specification serialization working")
    print(f"✅ PyTorch support: {'Available' if has_torch else 'Fallback mode'}")
    
    if has_torch:
        print("\n🚀 Ready for Generation 2: Robustness & Reliability")
    else:
        print("\n⚠️  Limited functionality without PyTorch")
        print("   Install PyTorch for full optimization capabilities")
    
    print("\n" + "=" * 60)
    print("Generation 1 demo completed successfully! 🎉")

if __name__ == "__main__":
    main()