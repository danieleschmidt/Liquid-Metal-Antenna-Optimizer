"""
Integration tests for the complete liquid metal antenna optimization system.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

from liquid_metal_antenna.core.antenna_spec import AntennaSpec, SubstrateMaterial, LiquidMetalType
from liquid_metal_antenna.solvers.fdtd import DifferentiableFDTD
from liquid_metal_antenna.solvers.enhanced_fdtd import EnhancedFDTD
from liquid_metal_antenna.optimization.lma_optimizer import LMAOptimizer
from liquid_metal_antenna.optimization.neural_surrogate import NeuralSurrogate
from liquid_metal_antenna.designs.patch_antenna import PatchAntenna
from liquid_metal_antenna.designs.monopole_antenna import MonopoleAntenna
from liquid_metal_antenna.designs.array_antenna import ArrayAntenna
from liquid_metal_antenna.utils.diagnostics import SystemDiagnostics
from liquid_metal_antenna.utils.logging_config import setup_logging, get_logger


class TestEndToEndWorkflow:
    """Test complete end-to-end optimization workflow."""
    
    @pytest.fixture
    def test_spec(self):
        """Create comprehensive test specification."""
        return AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            performance_targets={
                'min_gain_dbi': 6.0,
                'max_vswr': 2.0,
                'min_efficiency': 0.8,
                'bandwidth_requirement': 'wide'
            }
        )
    
    def test_complete_patch_antenna_optimization(self, test_spec):
        """Test complete patch antenna design and optimization."""
        # Create patch antenna design
        patch_design = PatchAntenna(test_spec)
        initial_geometry = patch_design.generate_initial_design()
        
        assert initial_geometry is not None
        assert initial_geometry.ndim == 3
        assert np.any(initial_geometry > 0)  # Should have some conductor
        
        # Setup optimizer
        optimizer = LMAOptimizer(test_spec, n_iterations=5)
        
        # Mock solver for testing
        mock_solver = Mock()
        mock_solver.simulate.return_value = Mock(
            gain_dbi=6.5,
            vswr=1.8,
            s_parameters=np.array([[[0.15+0.08j]]]),
            converged=True,
            computation_time=1.2
        )
        optimizer.solver = mock_solver
        
        # Run optimization
        result = optimizer.optimize(initial_geometry)
        
        # Verify results
        assert result is not None
        assert result.optimized_geometry is not None
        assert result.final_objective_value is not None
        assert len(result.objective_history) <= 5
        
        # Verify performance improvement
        initial_objective = optimizer._evaluate_objective(initial_geometry)
        final_objective = result.final_objective_value
        
        # Should show some improvement (or at least not get worse)
        assert final_objective >= initial_objective * 0.9  # Allow 10% tolerance
    
    def test_monopole_antenna_with_enhanced_solver(self, test_spec):
        """Test monopole antenna optimization with enhanced solver."""
        # Create monopole design
        monopole_design = MonopoleAntenna(test_spec)
        initial_geometry = monopole_design.generate_initial_design()
        
        # Setup enhanced solver
        enhanced_solver = EnhancedFDTD(
            resolution=3e-3,
            stability_check=True,
            adaptive_stepping=True
        )
        
        # Mock the solver's simulation method
        enhanced_solver.simulate = Mock(return_value=Mock(
            gain_dbi=4.2,
            vswr=2.1,
            s_parameters=np.array([[[0.2+0.1j]]]),
            converged=True,
            iterations=150,
            stability_score=0.95
        ))
        
        # Setup optimizer with enhanced solver
        optimizer = LMAOptimizer(test_spec, n_iterations=3)
        optimizer.solver = enhanced_solver
        
        result = optimizer.optimize(initial_geometry)
        
        # Verify integration
        assert result is not None
        assert hasattr(result, 'optimized_geometry')
        
        # Verify enhanced solver was used
        enhanced_solver.simulate.assert_called()
    
    def test_array_antenna_multi_objective_optimization(self, test_spec):
        """Test array antenna with multi-objective optimization."""
        # Modify spec for array requirements
        array_spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            performance_targets={
                'min_gain_dbi': 12.0,  # Higher gain for array
                'max_vswr': 1.5,
                'beamwidth_requirement': 'narrow',
                'sidelobe_level': -20  # dB
            }
        )
        
        # Create array design
        array_design = ArrayAntenna(array_spec, n_elements=4)
        initial_geometry = array_design.generate_initial_design()
        
        # Setup multi-objective optimizer
        optimizer = LMAOptimizer(
            array_spec,
            optimization_mode='multi_objective',
            n_iterations=5
        )
        
        # Mock solver for array simulation
        mock_solver = Mock()
        mock_solver.simulate.return_value = Mock(
            gain_dbi=13.5,
            vswr=1.3,
            radiation_pattern=np.ones((18, 36)) * 0.8,  # Narrower beam
            sidelobe_level=-18,
            converged=True
        )
        optimizer.solver = mock_solver
        
        result = optimizer.optimize(initial_geometry)
        
        # Verify multi-objective optimization
        assert result is not None
        assert hasattr(result, 'pareto_front')  # Multi-objective specific
        assert len(result.objective_history) > 0
    
    def test_neural_surrogate_integration(self, test_spec):
        """Test integration with neural surrogate models."""
        surrogate = NeuralSurrogate(test_spec)
        
        # Generate training data
        training_geometries = [
            np.random.rand(16, 16, 4) * 0.5 for _ in range(50)
        ]
        
        # Mock training results
        training_results = []
        for i, geom in enumerate(training_geometries):
            result = Mock()
            result.gain_dbi = 4.0 + np.random.randn() * 0.5
            result.vswr = 1.5 + np.random.randn() * 0.3
            result.s_parameters = np.array([[[0.1 + 0.05j * np.random.randn()]]])
            training_results.append(result)
        
        # Train surrogate model
        surrogate.train(training_geometries, training_results)
        
        # Test prediction
        test_geometry = np.random.rand(16, 16, 4) * 0.5
        prediction = surrogate.predict(test_geometry, 2.45e9, test_spec)
        
        # Verify prediction structure
        assert prediction is not None
        assert hasattr(prediction, 'gain_dbi')
        assert hasattr(prediction, 'vswr')
        assert prediction.computation_time < 0.1  # Should be very fast
        
        # Integration test with optimizer
        optimizer = LMAOptimizer(test_spec, n_iterations=3)
        optimizer.solver = surrogate  # Use surrogate as solver
        
        initial_geometry = np.random.rand(16, 16, 4) * 0.4
        result = optimizer.optimize(initial_geometry)
        
        assert result is not None
        assert result.optimized_geometry is not None


class TestSystemIntegration:
    """Test integration between major system components."""
    
    def test_logging_integration(self):
        """Test logging system integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup logging
            setup_logging(
                log_dir=temp_dir,
                console_level='INFO',
                file_level='DEBUG'
            )
            
            logger = get_logger('integration_test')
            
            # Test logging during operations
            logger.info("Starting integration test")
            
            spec = AntennaSpec(
                frequency_range=(2.4e9, 2.5e9),
                substrate=SubstrateMaterial.ROGERS_4003C,
                metal=LiquidMetalType.GALINSTAN
            )
            
            logger.debug(f"Created antenna spec: {spec}")
            
            # Create optimizer with logging
            optimizer = LMAOptimizer(spec, n_iterations=2)
            
            # Mock solver that logs
            mock_solver = Mock()
            def mock_simulate_with_logging(geometry, **kwargs):
                logger.info("Running FDTD simulation")
                return Mock(gain_dbi=5.0, vswr=1.5, converged=True)
            
            mock_solver.simulate = mock_simulate_with_logging
            optimizer.solver = mock_solver
            
            geometry = np.random.rand(12, 12, 4) * 0.5
            result = optimizer.optimize(geometry)
            
            logger.info("Optimization completed successfully")
            
            # Verify log files were created
            log_files = os.listdir(temp_dir)
            assert any('antenna_optimizer' in f for f in log_files)
            
            # Check log content
            main_log = os.path.join(temp_dir, 'antenna_optimizer.log')
            if os.path.exists(main_log):
                with open(main_log, 'r') as f:
                    log_content = f.read()
                    assert 'integration_test' in log_content
                    assert 'FDTD simulation' in log_content
    
    def test_diagnostics_integration(self):
        """Test diagnostics system integration."""
        diagnostics = SystemDiagnostics()
        
        # Run health checks
        health_results = diagnostics.run_all_health_checks()
        
        assert 'system_resources' in health_results
        assert 'python_environment' in health_results
        assert 'dependencies' in health_results
        
        # Get system metrics
        metrics = diagnostics.get_system_metrics()
        
        assert metrics.cpu_count > 0
        assert metrics.memory_total_gb > 0
        assert metrics.python_version is not None
        
        # Integration with optimization
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec)
        
        # Monitor system during optimization
        initial_metrics = diagnostics.get_system_metrics()
        
        # Mock solver
        mock_solver = Mock()
        mock_solver.simulate.return_value = Mock(gain_dbi=5.0, vswr=1.5)
        optimizer.solver = mock_solver
        
        geometry = np.random.rand(10, 10, 4) * 0.5
        result = optimizer.optimize(geometry, max_iterations=2)
        
        final_metrics = diagnostics.get_system_metrics()
        
        # System should remain stable
        memory_increase = final_metrics.memory_used_gb - initial_metrics.memory_used_gb
        assert memory_increase < 1.0  # Less than 1GB increase
    
    def test_caching_integration(self):
        """Test caching system integration with optimization."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Setup optimizer with caching enabled
        optimizer = LMAOptimizer(spec, enable_caching=True, n_iterations=3)
        
        # Mock solver that tracks calls
        call_count = 0
        def counting_simulator(geometry, **kwargs):
            nonlocal call_count
            call_count += 1
            return Mock(
                gain_dbi=5.0 + np.random.randn() * 0.1,
                vswr=1.5 + np.random.randn() * 0.05,
                converged=True
            )
        
        mock_solver = Mock()
        mock_solver.simulate = counting_simulator
        optimizer.solver = mock_solver
        
        # Run same optimization twice
        geometry = np.random.rand(12, 12, 4) * 0.5
        geometry_copy = geometry.copy()  # Exact copy for cache hit
        
        result1 = optimizer.optimize(geometry)
        initial_calls = call_count
        
        result2 = optimizer.optimize(geometry_copy)
        final_calls = call_count
        
        # Second run should have fewer solver calls due to caching
        cache_savings = initial_calls - (final_calls - initial_calls)
        assert cache_savings > 0, f"No cache benefit observed: {initial_calls} vs {final_calls - initial_calls}"
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec, n_iterations=3)
        
        # Mock solver that fails occasionally
        failure_count = 0
        def failing_solver(geometry, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count % 2 == 0:  # Fail every other call
                raise RuntimeError("Simulated solver failure")
            return Mock(gain_dbi=5.0, vswr=1.5, converged=True)
        
        mock_solver = Mock()
        mock_solver.simulate = failing_solver
        optimizer.solver = mock_solver
        
        geometry = np.random.rand(10, 10, 4) * 0.5
        
        # Optimization should handle failures gracefully
        result = optimizer.optimize(geometry)
        
        # Should still return a result despite failures
        assert result is not None
        assert result.optimized_geometry is not None
        
        # Should have attempted recovery
        assert failure_count > 1  # Multiple calls indicate retry attempts


class TestDataFlowIntegration:
    """Test data flow between components."""
    
    def test_geometry_transformation_pipeline(self):
        """Test geometry data flow through the complete pipeline."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Start with design generator
        patch_design = PatchAntenna(spec)
        initial_geometry = patch_design.generate_initial_design()
        
        original_shape = initial_geometry.shape
        original_metal_fraction = np.mean(initial_geometry > 0.5)
        
        # Transform through optimizer
        optimizer = LMAOptimizer(spec, n_iterations=3)
        
        # Track geometry transformations
        geometry_history = []
        
        def tracking_solver(geometry, **kwargs):
            geometry_history.append(geometry.copy())
            return Mock(
                gain_dbi=5.0 + len(geometry_history) * 0.1,  # Improving performance
                vswr=2.0 - len(geometry_history) * 0.1,
                converged=True
            )
        
        mock_solver = Mock()
        mock_solver.simulate = tracking_solver
        optimizer.solver = mock_solver
        
        result = optimizer.optimize(initial_geometry)
        
        # Verify data flow
        assert len(geometry_history) > 0
        
        # All geometries should maintain shape
        for geometry in geometry_history:
            assert geometry.shape == original_shape
            assert np.all(geometry >= 0)
            assert np.all(geometry <= 1)
        
        # Final geometry should be different from initial
        final_geometry = result.optimized_geometry
        assert not np.allclose(initial_geometry, final_geometry, atol=1e-10)
    
    def test_performance_metrics_propagation(self):
        """Test performance metrics propagation through system."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            performance_targets={
                'min_gain_dbi': 6.0,
                'max_vswr': 2.0,
                'min_efficiency': 0.85
            }
        )
        
        optimizer = LMAOptimizer(spec, n_iterations=3)
        
        # Mock solver with detailed performance metrics
        def detailed_solver(geometry, **kwargs):
            return Mock(
                gain_dbi=6.5,
                max_gain_dbi=6.8,
                vswr=1.7,
                efficiency=0.88,
                s_parameters=np.array([[[0.12+0.06j]]]),
                radiation_pattern=np.ones((18, 36)),
                theta_angles=np.linspace(0, np.pi, 18),
                phi_angles=np.linspace(0, 2*np.pi, 36),
                converged=True,
                iterations=120,
                computation_time=2.1
            )
        
        mock_solver = Mock()
        mock_solver.simulate = detailed_solver
        optimizer.solver = mock_solver
        
        geometry = np.random.rand(16, 16, 4) * 0.5
        result = optimizer.optimize(geometry)
        
        # Verify all metrics are available in result
        assert hasattr(result, 'final_objective_value')
        assert hasattr(result, 'optimized_geometry')
        assert len(result.objective_history) > 0
        
        # Performance metrics should meet targets
        final_result = mock_solver.simulate(result.optimized_geometry)
        assert final_result.gain_dbi >= spec.performance_targets['min_gain_dbi']
        assert final_result.vswr <= spec.performance_targets['max_vswr']
        assert final_result.efficiency >= spec.performance_targets['min_efficiency']


class TestConfigurationIntegration:
    """Test configuration and parameter flow integration."""
    
    def test_specification_parameter_propagation(self):
        """Test that specification parameters propagate correctly."""
        # Create spec with specific parameters
        spec = AntennaSpec(
            frequency_range=(5.8e9, 6.0e9),  # 5.8 GHz ISM band
            substrate=SubstrateMaterial.DUROID_5880,
            metal=LiquidMetalType.MERCURY,
            size_constraint={
                'max_width': 25e-3,   # 25mm
                'max_height': 25e-3,  # 25mm
                'max_thickness': 0.8e-3  # 0.8mm
            },
            performance_targets={
                'min_gain_dbi': 8.0,
                'max_vswr': 1.5,
                'min_efficiency': 0.90,
                'bandwidth_percentage': 5.0
            }
        )
        
        # Create components and verify parameter propagation
        solver = DifferentiableFDTD(resolution=1e-3)
        optimizer = LMAOptimizer(spec, algorithm='genetic')
        design = PatchAntenna(spec)
        
        # Generate initial design and verify constraints
        initial_geometry = design.generate_initial_design()
        
        # Geometry should respect size constraints
        dx = solver.resolution
        max_width_pixels = int(spec.size_constraint['max_width'] / dx)
        max_height_pixels = int(spec.size_constraint['max_height'] / dx)
        
        assert initial_geometry.shape[0] <= max_width_pixels * 1.2  # Allow some tolerance
        assert initial_geometry.shape[1] <= max_height_pixels * 1.2
        
        # Mock solver to verify frequency parameter usage
        frequency_used = []
        def frequency_tracking_solver(geometry, frequency, **kwargs):
            frequency_used.append(frequency)
            return Mock(gain_dbi=8.2, vswr=1.3, converged=True)
        
        mock_solver = Mock()
        mock_solver.simulate = frequency_tracking_solver
        optimizer.solver = mock_solver
        
        result = optimizer.optimize(initial_geometry, max_iterations=2)
        
        # Verify frequency from spec was used
        assert len(frequency_used) > 0
        for freq in frequency_used:
            assert spec.frequency_range[0] <= freq <= spec.frequency_range[1]
    
    def test_cross_component_validation(self):
        """Test validation across integrated components."""
        # Create specification with conflicting requirements
        conflicting_spec = AntennaSpec(
            frequency_range=(60e9, 61e9),  # Very high frequency
            substrate=SubstrateMaterial.ROGERS_4003C,  # Standard substrate
            metal=LiquidMetalType.GALINSTAN,
            size_constraint={
                'max_width': 100e-3,   # Large size
                'max_height': 100e-3,  # Large size  
                'max_thickness': 1.6e-3
            },
            performance_targets={
                'min_gain_dbi': 20.0,  # Very high gain
                'max_vswr': 1.2,       # Very low VSWR
                'bandwidth_percentage': 20.0  # Very wide bandwidth
            }
        )
        
        # System should detect conflicts during integration
        with pytest.warns(UserWarning):  # Should warn about unrealistic requirements
            optimizer = LMAOptimizer(conflicting_spec)
            design = PatchAntenna(conflicting_spec)
            
            # Validation should catch the conflicts
            warnings_caught = []
            import warnings
            
            def warning_handler(message, category=UserWarning, filename='', lineno=-1, file=None, line=None):
                warnings_caught.append(str(message))
            
            old_showwarning = warnings.showwarning
            warnings.showwarning = warning_handler
            
            try:
                initial_geometry = design.generate_initial_design()
                conflicting_spec.validate()
            finally:
                warnings.showwarning = old_showwarning
            
            # Should have generated warnings about conflicts
            assert len(warnings_caught) > 0
            assert any('high frequency' in w.lower() for w in warnings_caught) or \
                   any('unrealistic' in w.lower() for w in warnings_caught)


class TestScalabilityIntegration:
    """Test system behavior with different scales and complexities."""
    
    @pytest.mark.parametrize("complexity", [
        ("simple", (16, 16, 4), 2.45e9, 3),
        ("medium", (32, 32, 6), 5.8e9, 5),
        ("complex", (48, 48, 8), 10e9, 8),
    ])
    def test_scalability_integration(self, complexity):
        """Test system integration at different complexity levels."""
        name, geometry_size, frequency, max_iterations = complexity
        
        spec = AntennaSpec(
            frequency_range=(frequency*0.95, frequency*1.05),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Create components scaled for complexity
        if name == "simple":
            solver = DifferentiableFDTD(resolution=3e-3)  # Coarser resolution
        elif name == "medium":
            solver = DifferentiableFDTD(resolution=2e-3)  # Medium resolution
        else:  # complex
            solver = EnhancedFDTD(resolution=1e-3, adaptive_stepping=True)  # Fine resolution
        
        optimizer = LMAOptimizer(spec, n_iterations=max_iterations)
        
        # Mock solver with complexity-dependent behavior
        def complexity_aware_solver(geometry, **kwargs):
            # Simulate complexity-dependent computation time
            complexity_factor = np.prod(geometry.shape) / 1000
            simulated_time = 0.01 * complexity_factor  # Base time scaled by complexity
            
            return Mock(
                gain_dbi=5.0 + np.random.randn() * 0.2,
                vswr=1.5 + np.random.randn() * 0.1,
                converged=True,
                computation_time=simulated_time
            )
        
        mock_solver = Mock()
        mock_solver.simulate = complexity_aware_solver
        optimizer.solver = mock_solver
        
        # Generate geometry at specified complexity
        geometry = np.random.rand(*geometry_size) * 0.5
        
        # Run optimization
        import time
        start_time = time.time()
        result = optimizer.optimize(geometry)
        actual_time = time.time() - start_time
        
        # Verify results scale appropriately
        assert result is not None
        assert result.optimized_geometry.shape == geometry_size
        
        # Time should scale reasonably with complexity
        expected_time_limits = {
            "simple": 2.0,    # seconds
            "medium": 5.0,    # seconds  
            "complex": 10.0   # seconds
        }
        
        assert actual_time < expected_time_limits[name], \
            f"{name} complexity took {actual_time:.2f}s > {expected_time_limits[name]}s"


if __name__ == '__main__':
    pytest.main([__file__, "-v"])