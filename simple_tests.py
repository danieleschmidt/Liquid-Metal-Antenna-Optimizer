#!/usr/bin/env python3
"""
Simple tests focused on components that work without PyTorch.
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports that should work without torch."""
    print("🔧 Testing basic imports...")
    try:
        # Core components
        from liquid_metal_antenna.core.antenna_spec import (
            AntennaSpec, SubstrateMaterial, LiquidMetalType
        )
        print("   ✅ Core antenna_spec imports successful")
        
        # Utils should work
        from liquid_metal_antenna.utils.logging_config import get_logger
        from liquid_metal_antenna.utils.validation import ValidationError
        print("   ✅ Utils imports successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {str(e)}")
        return False

def test_antenna_spec():
    """Test AntennaSpec functionality."""
    print("🧪 Testing AntennaSpec...")
    
    try:
        from liquid_metal_antenna.core.antenna_spec import (
            AntennaSpec, SubstrateMaterial, LiquidMetalType
        )
        
        # Create antenna spec
        spec = AntennaSpec(
            frequency_range=(2e9, 3e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            size_constraint=(30, 30, 3)
        )
        
        # Test properties
        print(f"      Frequency range: {spec.frequency_range}, expected: {(2e9, 3e9)}")
        print(f"      Substrate: {spec.substrate}, expected: {SubstrateMaterial.ROGERS_4003C}")
        print(f"      Metal: {spec.metal}, expected: {LiquidMetalType.GALINSTAN}")
        
        # Check frequency range properties
        assert spec.frequency_range.start == 2e9
        assert spec.frequency_range.stop == 3e9
        # Check substrate properties (it's a MaterialProperties object)
        assert spec.substrate.dielectric_constant == 3.38  # Rogers 4003C
        # Check metal type
        assert spec.metal == LiquidMetalType.GALINSTAN
        
        # Test computed properties  
        center_freq = (spec.frequency_range.start + spec.frequency_range.stop) / 2
        expected_center = 2.5e9
        print(f"      Center frequency: {center_freq}, expected: {expected_center}")
        assert abs(center_freq - expected_center) < 1e6  # Allow small numerical differences
        
        print("   ✅ AntennaSpec creation and properties work")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"   ❌ AntennaSpec test failed: {str(e)}")
        print(f"      Traceback: {traceback.format_exc()}")
        return False

def test_fallback_optimizer():
    """Test the fallback optimizer."""
    print("🚀 Testing fallback optimizer...")
    
    try:
        from liquid_metal_antenna.core.optimizer_fallback import SimpleLMAOptimizer
        from liquid_metal_antenna.core.antenna_spec import (
            AntennaSpec, SubstrateMaterial, LiquidMetalType
        )
        
        # Create spec
        spec = AntennaSpec(
            frequency_range=(2e9, 3e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            size_constraint=(30, 30, 3)
        )
        
        # Create optimizer
        optimizer = SimpleLMAOptimizer(spec)
        
        # Test geometry creation
        geometry = optimizer.create_initial_geometry(spec)
        assert geometry.shape == (32, 32, 8)
        
        print("   ✅ Fallback optimizer initialization and geometry creation work")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback optimizer test failed: {str(e)}")
        return False

def test_multi_objective_basic():
    """Test basic multi-objective imports without torch dependencies."""
    print("🎯 Testing multi-objective basics...")
    
    try:
        # Test if we can at least import the module structure
        import liquid_metal_antenna.optimization.multi_objective as mo
        
        # Test objective creation
        objectives = mo.create_standard_objectives()
        assert len(objectives) == 4  # Should have 4 standard objectives
        assert objectives[0].name == "gain"
        assert objectives[1].name == "vswr"
        
        print("   ✅ Multi-objective structure imports work")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Multi-objective test failed: {str(e)}")
        return False

def main():
    """Run simple tests."""
    print("🧪 LIQUID METAL ANTENNA OPTIMIZER - SIMPLE TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("AntennaSpec", test_antenna_spec), 
        ("Fallback Optimizer", test_fallback_optimizer),
        ("Multi-Objective Basics", test_multi_objective_basic),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == '__main__':
    sys.exit(main())