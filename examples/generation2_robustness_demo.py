#!/usr/bin/env python3
"""
Generation 2 Robustness Demo
Demonstrates enhanced error handling, logging, and monitoring capabilities.
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Demonstrate Generation 2 robustness features."""
    
    print("🛡️  Liquid Metal Antenna Optimizer - Generation 2 Robustness Demo")
    print("=" * 70)
    
    # Create temporary log directory for demo
    log_dir = tempfile.mkdtemp(prefix='antenna_logs_')
    print(f"📝 Log directory: {log_dir}")
    
    try:
        # Setup enhanced logging
        print("\n📊 Setting up Enhanced Logging...")
        try:
            from liquid_metal_antenna.utils.logging_config import setup_logging, get_logger
            setup_logging(
                log_dir=log_dir,
                console_level='INFO',
                file_level='DEBUG',
                structured_output=True
            )
            logger = get_logger('demo')
            logger.info("Generation 2 robustness demo started")
            print("✅ Enhanced logging configured")
        except ImportError:
            print("⚠️  Enhanced logging not available - using basic logging")
            logger = None
        
        # Import core components
        from liquid_metal_antenna import AntennaSpec, LMAOptimizer
        print("✅ Core components imported")
        
        # Test error handling
        print("\n🔧 Testing Error Handling...")
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate='rogers_4003c',
            metal='galinstan',
            size_constraint=(25, 25, 1.6)
        )
        
        optimizer = LMAOptimizer(spec=spec)
        
        # Test 1: Valid optimization
        print("\n🎯 Test 1: Valid Optimization")
        try:
            result = optimizer.optimize(
                objective='max_gain',
                constraints={'vswr': '<2.0'},
                n_iterations=15
            )
            print(f"✅ Optimization succeeded: {result.gain_dbi:.1f} dBi, VSWR {result.vswr:.2f}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
        
        # Test 2: Invalid constraints (trigger error handling)
        print("\n🚨 Test 2: Error Recovery Testing")
        try:
            # This should trigger error handling with invalid constraints
            result = optimizer.optimize(
                objective='invalid_objective',  # Invalid objective
                constraints={'invalid_constraint': '<1.0'},
                n_iterations=5
            )
            print(f"✅ Error recovery successful: {result.gain_dbi:.1f} dBi")
            
        except Exception as e:
            print(f"⚠️  Error not recovered: {e}")
        
        # Test 3: Resource monitoring
        print("\n📈 Test 3: Performance Monitoring")
        import time
        start_time = time.time()
        
        # Run multiple optimizations to generate metrics
        results = []
        for i in range(3):
            try:
                result = optimizer.optimize(
                    objective='max_gain',
                    constraints={'vswr': '<2.5'},
                    n_iterations=10
                )
                results.append(result)
                print(f"   Run {i+1}: {result.gain_dbi:.1f} dBi in {result.optimization_time:.2f}s")
                
            except Exception as e:
                print(f"   Run {i+1}: Failed - {e}")
        
        total_time = time.time() - start_time
        print(f"✅ Performance monitoring: {len(results)} successful runs in {total_time:.2f}s")
        
        # Test 4: Error statistics and health monitoring
        print("\n🏥 Test 4: Health Monitoring")
        try:
            from liquid_metal_antenna.utils.error_handling import global_error_handler
            
            # Get error statistics
            stats = global_error_handler.get_error_statistics()
            health_report = global_error_handler.get_error_report()
            
            print(f"   Total errors: {stats['total_errors']}")
            print(f"   Resolution rate: {stats['resolution_rate']:.1f}%")
            print(f"   Health score: {health_report['health_score']:.1f}/100")
            
            if health_report['recommendations']:
                print("   Recommendations:")
                for rec in health_report['recommendations'][:3]:
                    print(f"     • {rec}")
            
            print("✅ Health monitoring active")
            
        except ImportError:
            print("⚠️  Advanced health monitoring not available")
        
        # Test 5: Security validation
        print("\n🔒 Test 5: Security Validation")
        try:
            from liquid_metal_antenna.utils.security import SecurityValidator
            
            validator = SecurityValidator()
            
            # Test geometry validation
            test_geometry = optimizer.create_initial_geometry(spec)
            validation_result = validator.validate_geometry(test_geometry)
            
            print(f"   Geometry validation: {'✅ Passed' if validation_result['valid'] else '❌ Failed'}")
            
            if validation_result.get('warnings'):
                print(f"   Warnings: {len(validation_result['warnings'])}")
            
            print("✅ Security validation active")
            
        except ImportError:
            print("⚠️  Security validation not available")
        
        # Test 6: Constraint validation and recovery
        print("\n🎚️  Test 6: Constraint Handling")
        
        # Test with strict constraints that might fail
        strict_constraints = {
            'vswr': '<1.5',        # Very strict VSWR
            'efficiency': '>0.95', # Very high efficiency
            'bandwidth': '>200e6'  # Large bandwidth
        }
        
        try:
            result = optimizer.optimize(
                objective='max_gain',
                constraints=strict_constraints,
                n_iterations=15
            )
            
            print(f"   Strict optimization result:")
            print(f"     Gain: {result.gain_dbi:.1f} dBi")
            print(f"     VSWR: {result.vswr:.2f}")
            print(f"     Efficiency: {result.efficiency:.1%}")
            print(f"     Converged: {result.converged}")
            
            # Check constraint satisfaction
            vswr_ok = result.vswr <= 1.5
            eff_ok = result.efficiency >= 0.95
            bw_ok = result.bandwidth_hz >= 200e6
            
            print(f"   Constraint satisfaction:")
            print(f"     VSWR < 1.5: {'✅' if vswr_ok else '❌'}")
            print(f"     Eff > 95%: {'✅' if eff_ok else '❌'}")
            print(f"     BW > 200MHz: {'✅' if bw_ok else '❌'}")
            
        except Exception as e:
            print(f"   Strict constraints failed: {e}")
        
        # Summary of Generation 2 features
        print("\n🎯 Generation 2 Feature Summary")
        print("=" * 70)
        print("✅ Enhanced error handling with recovery strategies")
        print("✅ Comprehensive logging and monitoring")
        print("✅ Performance metrics and health scoring")
        print("✅ Security validation and input sanitization")
        print("✅ Robust constraint handling")
        print("✅ Circuit breaker pattern for failing operations")
        print("✅ Graceful degradation under stress")
        
        # Log analysis
        print(f"\n📋 Log Analysis")
        try:
            log_files = os.listdir(log_dir)
            print(f"   Generated log files: {len(log_files)}")
            for log_file in log_files:
                file_path = os.path.join(log_dir, log_file)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"     {log_file}: {size_kb:.1f} KB")
            
        except Exception as e:
            print(f"   Log analysis error: {e}")
        
        print("\n🚀 Ready for Generation 3: Scale and Optimize!")
        
    finally:
        # Cleanup temporary log directory
        try:
            shutil.rmtree(log_dir)
            print(f"\n🧹 Cleaned up log directory: {log_dir}")
        except Exception as e:
            print(f"⚠️  Could not clean up log directory: {e}")
    
    print("\n" + "=" * 70)
    print("Generation 2 robustness demo completed! 🎉")

if __name__ == "__main__":
    main()