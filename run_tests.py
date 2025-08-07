#!/usr/bin/env python3
"""
Simple test runner for the liquid metal antenna optimizer.
"""

import sys
import time
import traceback
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

def run_import_tests():
    """Test basic imports."""
    print("🔧 Testing basic imports...")
    try:
        import liquid_metal_antenna
        from liquid_metal_antenna import (
            AntennaSpec, LMAOptimizer, DifferentiableFDTD, 
            ReconfigurablePatch, LiquidMetalArray
        )
        from liquid_metal_antenna.optimization import (
            MultiObjectiveOptimizer, BayesianOptimizer, NeuralSurrogate
        )
        from liquid_metal_antenna.designs import BeamformingArray
        
        print("   ✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {str(e)}")
        traceback.print_exc()
        return False

def run_basic_functionality_tests():
    """Test basic functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test AntennaSpec creation
        from liquid_metal_antenna import AntennaSpec
        from liquid_metal_antenna.core.antenna_spec import SubstrateMaterial, LiquidMetalType
        
        spec = AntennaSpec(
            frequency_range=(2e9, 3e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            size_constraint=(30, 30, 3)
        )
        print("   ✅ AntennaSpec creation successful")
        
        # Test FDTD solver initialization
        from liquid_metal_antenna import DifferentiableFDTD
        
        solver = DifferentiableFDTD(
            resolution=1e-3,
            domain_size=(50e-3, 50e-3, 10e-3),
            pml_thickness=5e-3
        )
        print("   ✅ FDTD solver initialization successful")
        
        # Test multi-objective optimizer
        from liquid_metal_antenna.optimization import MultiObjectiveOptimizer
        
        optimizer = MultiObjectiveOptimizer(algorithm='nsga3', population_size=20)
        print("   ✅ Multi-objective optimizer initialization successful")
        
        # Test BeamformingArray
        from liquid_metal_antenna.designs import BeamformingArray
        
        array = BeamformingArray(
            n_elements=(4, 4),
            element_spacing=(0.5, 0.5),
            frequency=2.45e9
        )
        print("   ✅ BeamformingArray initialization successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {str(e)}")
        traceback.print_exc()
        return False

def run_optimization_tests():
    """Test optimization algorithms."""
    print("🚀 Testing optimization algorithms...")
    
    try:
        # Test Bayesian optimization
        from liquid_metal_antenna.optimization import BayesianOptimizer
        
        optimizer = BayesianOptimizer(
            acquisition_function='ei',
            kernel='matern52',
            n_initial_points=3
        )
        print("   ✅ Bayesian optimizer initialization successful")
        
        # Test NSGA-III
        from liquid_metal_antenna.optimization import (
            NSGA3Optimizer, create_standard_objectives
        )
        
        objectives = create_standard_objectives()
        nsga3 = NSGA3Optimizer(objectives, population_size=10, n_generations=5)
        print("   ✅ NSGA-III optimizer initialization successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Optimization test failed: {str(e)}")
        traceback.print_exc()
        return False

def run_neural_surrogate_tests():
    """Test neural surrogate models."""
    print("🧠 Testing neural surrogate models...")
    
    try:
        from liquid_metal_antenna.optimization import NeuralSurrogate, SurrogateTrainer
        import numpy as np
        
        # Test surrogate model creation
        surrogate = NeuralSurrogate(
            model_type='fourier_neural_operator',
            input_resolution=(16, 16, 4)
        )
        print("   ✅ Neural surrogate initialization successful")
        
        # Test prediction (without actual training)
        from liquid_metal_antenna.core.antenna_spec import AntennaSpec, SubstrateMaterial, LiquidMetalType
        
        spec = AntennaSpec(
            frequency_range=(2e9, 3e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            size_constraint=(30, 30, 3)
        )
        
        geometry = np.random.random((16, 16, 4))
        result = surrogate.predict(geometry, 2.45e9, spec)
        print("   ✅ Neural surrogate prediction successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Neural surrogate test failed: {str(e)}")
        traceback.print_exc()
        return False

def run_beam_steering_tests():
    """Test beam steering functionality."""
    print("📡 Testing beam steering functionality...")
    
    try:
        from liquid_metal_antenna.designs import (
            BeamformingArray, LiquidMetalPhaseShifter
        )
        import numpy as np
        
        # Test phase shifter
        phase_shifter = LiquidMetalPhaseShifter(
            max_delay=1e-9,
            frequency=2.45e9
        )
        
        phase = phase_shifter.calculate_phase_shift(0.5)
        fill_ratio = phase_shifter.calculate_required_fill_ratio(np.pi/4)
        print("   ✅ Liquid metal phase shifter successful")
        
        # Test beamforming array
        array = BeamformingArray(
            n_elements=(4, 4),
            frequency=2.45e9
        )
        
        # Test array factor calculation
        theta = np.linspace(0, np.pi, 10)
        phi = np.linspace(0, 2*np.pi, 20)
        array_factor = array.calculate_array_factor(theta, phi)
        print("   ✅ Array factor calculation successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Beam steering test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 LIQUID METAL ANTENNA OPTIMIZER - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Tests", run_import_tests),
        ("Basic Functionality Tests", run_basic_functionality_tests),
        ("Optimization Tests", run_optimization_tests),
        ("Neural Surrogate Tests", run_neural_surrogate_tests),
        ("Beam Steering Tests", run_beam_steering_tests)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        test_start = time.time()
        success = test_func()
        test_time = time.time() - test_start
        
        results.append((test_name, success, test_time))
        
        if success:
            print(f"   ✅ PASSED ({test_time:.3f}s)")
        else:
            print(f"   ❌ FAILED ({test_time:.3f}s)")
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {total_time:.3f}s")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == '__main__':
    sys.exit(main())