"""
Performance benchmarks and quality gates for liquid metal antenna optimizer.
"""

import pytest
import time
import numpy as np
import psutil
import threading
from unittest.mock import Mock, patch
from contextlib import contextmanager

from liquid_metal_antenna.core.antenna_spec import AntennaSpec, SubstrateMaterial, LiquidMetalType
from liquid_metal_antenna.solvers.fdtd import DifferentiableFDTD
from liquid_metal_antenna.optimization.lma_optimizer import LMAOptimizer
from liquid_metal_antenna.optimization.caching import OptimizationCache
from liquid_metal_antenna.optimization.concurrent import ConcurrentProcessor
from liquid_metal_antenna.utils.diagnostics import SystemDiagnostics, PerformanceMonitor


@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage during operations."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    max_memory = initial_memory
    
    def monitor():
        nonlocal max_memory
        while True:
            try:
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                time.sleep(0.1)
            except:
                break
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    
    start_time = time.time()
    
    try:
        yield {
            'get_peak_memory': lambda: max_memory,
            'get_memory_increase': lambda: max_memory - initial_memory,
            'get_duration': lambda: time.time() - start_time
        }
    finally:
        monitor_thread.join(timeout=0.1)


class TestPerformanceBenchmarks:
    """Performance benchmarks for core components."""
    
    @pytest.fixture
    def standard_spec(self):
        """Standard antenna specification for benchmarking."""
        return AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
    
    @pytest.fixture
    def benchmark_geometry(self):
        """Standard geometry for benchmarking."""
        return np.random.rand(32, 32, 6) * 0.5
    
    def test_fdtd_solver_performance_benchmark(self, standard_spec, benchmark_geometry):
        """Benchmark FDTD solver performance."""
        solver = DifferentiableFDTD(resolution=2e-3, precision='float32')
        
        # Performance targets
        TARGET_SETUP_TIME = 0.5  # seconds
        TARGET_MEMORY_USAGE = 500  # MB
        TARGET_SIMULATION_RATE = 1000  # time steps per second (estimated)
        
        with memory_monitor() as monitor:
            # Test solver setup performance
            setup_start = time.time()
            solver.set_grid_size(benchmark_geometry, standard_spec)
            setup_time = time.time() - setup_start
            
            assert setup_time < TARGET_SETUP_TIME, f"Setup too slow: {setup_time:.3f}s > {TARGET_SETUP_TIME}s"
            
            # Test memory usage during setup
            setup_memory = monitor['get_memory_increase']()
            assert setup_memory < TARGET_MEMORY_USAGE, f"Setup uses too much memory: {setup_memory:.1f}MB > {TARGET_MEMORY_USAGE}MB"
    
    @pytest.mark.slow
    def test_optimization_performance_benchmark(self, standard_spec):
        """Benchmark optimization performance."""
        optimizer = LMAOptimizer(standard_spec, n_iterations=20)
        
        # Mock solver for consistent timing
        mock_solver = Mock()
        mock_solver.simulate.return_value = Mock(
            gain_dbi=5.0, vswr=1.5, converged=True
        )
        optimizer.solver = mock_solver
        
        # Performance targets
        TARGET_ITERATION_TIME = 0.1  # seconds per iteration
        TARGET_CONVERGENCE_ITERATIONS = 15  # iterations to convergence
        
        geometry = np.random.rand(16, 16, 4) * 0.5
        
        with memory_monitor() as monitor:
            start_time = time.time()
            result = optimizer.optimize(geometry)
            total_time = time.time() - start_time
            
            iterations_performed = len(result.objective_history)
            avg_iteration_time = total_time / iterations_performed
            
            assert avg_iteration_time < TARGET_ITERATION_TIME, \
                f"Optimization too slow: {avg_iteration_time:.3f}s/iter > {TARGET_ITERATION_TIME}s/iter"
            
            assert iterations_performed <= TARGET_CONVERGENCE_ITERATIONS, \
                f"Slow convergence: {iterations_performed} > {TARGET_CONVERGENCE_ITERATIONS} iterations"
    
    def test_caching_performance_benchmark(self, standard_spec):
        """Benchmark caching system performance."""
        cache = OptimizationCache(max_size=1000)
        
        # Performance targets
        TARGET_CACHE_HIT_TIME = 0.001  # seconds
        TARGET_CACHE_MISS_TIME = 0.01  # seconds
        CACHE_HIT_RATIO_TARGET = 0.8  # 80% hit rate after warmup
        
        # Simulate cache operations
        test_geometries = [np.random.rand(16, 16, 4) for _ in range(100)]
        test_results = [Mock(gain_dbi=5.0 + i*0.1) for i in range(100)]
        
        # Cache miss benchmark (first access)
        cache_miss_times = []
        for i in range(10):
            start_time = time.time()
            cache.get(f"key_{i}", test_geometries[i])  # Cache miss
            cache.set(f"key_{i}", test_geometries[i], test_results[i])
            cache_miss_times.append(time.time() - start_time)
        
        avg_cache_miss_time = np.mean(cache_miss_times)
        assert avg_cache_miss_time < TARGET_CACHE_MISS_TIME, \
            f"Cache miss too slow: {avg_cache_miss_time:.4f}s > {TARGET_CACHE_MISS_TIME}s"
        
        # Cache hit benchmark (repeat access)
        cache_hit_times = []
        for i in range(10):
            start_time = time.time()
            result = cache.get(f"key_{i}", test_geometries[i])  # Cache hit
            cache_hit_times.append(time.time() - start_time)
            assert result is not None, "Expected cache hit"
        
        avg_cache_hit_time = np.mean(cache_hit_times)
        assert avg_cache_hit_time < TARGET_CACHE_HIT_TIME, \
            f"Cache hit too slow: {avg_cache_hit_time:.4f}s > {TARGET_CACHE_HIT_TIME}s"
    
    def test_concurrent_processing_performance(self):
        """Benchmark concurrent processing performance."""
        processor = ConcurrentProcessor(n_workers=4)
        
        # Performance targets
        TARGET_SPEEDUP = 2.0  # At least 2x speedup with 4 workers
        TARGET_OVERHEAD = 0.1  # Less than 10% overhead
        
        def mock_simulation(geometry):
            # Simulate computation time
            time.sleep(0.01)  # 10ms per simulation
            return np.sum(geometry)
        
        test_geometries = [np.random.rand(8, 8, 4) for _ in range(20)]
        
        # Sequential timing
        start_time = time.time()
        sequential_results = [mock_simulation(geom) for geom in test_geometries]
        sequential_time = time.time() - start_time
        
        # Concurrent timing
        start_time = time.time()
        concurrent_futures = []
        for geometry in test_geometries:
            future_id = processor.submit_task(mock_simulation, geometry)
            concurrent_futures.append(future_id)
        
        concurrent_results = []
        for future_id in concurrent_futures:
            result = processor.get_result(future_id, timeout=10.0)
            concurrent_results.append(result)
        
        concurrent_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sequential_time / concurrent_time
        assert speedup > TARGET_SPEEDUP, \
            f"Insufficient speedup: {speedup:.2f}x < {TARGET_SPEEDUP}x"
        
        # Verify results correctness
        assert len(concurrent_results) == len(sequential_results)
        for i, (seq, conc) in enumerate(zip(sequential_results, concurrent_results)):
            assert abs(seq - conc) < 1e-10, f"Result mismatch at index {i}"
    
    def test_memory_efficiency_benchmark(self, standard_spec):
        """Benchmark memory efficiency."""
        # Memory targets
        TARGET_BASE_MEMORY = 100  # MB base memory usage
        TARGET_PER_GEOMETRY_MEMORY = 10  # MB per geometry in memory
        
        geometries = []
        
        with memory_monitor() as monitor:
            base_memory = monitor['get_peak_memory']()
            
            # Add geometries and measure memory growth
            for i in range(10):
                geometry = np.random.rand(32, 32, 8)  # ~64KB per geometry
                geometries.append(geometry)
                
                if i == 0:
                    first_geometry_memory = monitor['get_peak_memory']()
                
            final_memory = monitor['get_peak_memory']()
        
        # Calculate memory usage
        base_overhead = first_geometry_memory - base_memory
        per_geometry_overhead = (final_memory - first_geometry_memory) / 9  # 9 additional geometries
        
        assert base_overhead < TARGET_BASE_MEMORY, \
            f"Base memory usage too high: {base_overhead:.1f}MB > {TARGET_BASE_MEMORY}MB"
        
        assert per_geometry_overhead < TARGET_PER_GEOMETRY_MEMORY, \
            f"Per-geometry overhead too high: {per_geometry_overhead:.1f}MB > {TARGET_PER_GEOMETRY_MEMORY}MB"


class TestScalabilityBenchmarks:
    """Test system scalability with increasing problem sizes."""
    
    @pytest.mark.parametrize("geometry_size", [
        (16, 16, 4),   # Small
        (32, 32, 6),   # Medium  
        (64, 64, 8),   # Large
    ])
    def test_geometry_size_scaling(self, geometry_size):
        """Test performance scaling with geometry size."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec, n_iterations=5)
        
        # Mock solver for consistent behavior
        mock_solver = Mock()
        mock_solver.simulate.return_value = Mock(gain_dbi=5.0, vswr=1.5)
        optimizer.solver = mock_solver
        
        geometry = np.random.rand(*geometry_size) * 0.5
        
        with memory_monitor() as monitor:
            start_time = time.time()
            result = optimizer.optimize(geometry)
            duration = time.time() - start_time
            peak_memory = monitor['get_peak_memory']()
        
        # Scaling targets (approximate)
        volume = np.prod(geometry_size)
        expected_memory = 50 + volume * 0.001  # Base + volume-dependent
        expected_time = 0.1 + volume * 1e-6   # Base + volume-dependent
        
        # Allow some tolerance for scaling
        assert peak_memory < expected_memory * 2, \
            f"Memory usage {peak_memory:.1f}MB exceeds 2x expected {expected_memory:.1f}MB for size {geometry_size}"
        
        assert duration < expected_time * 3, \
            f"Duration {duration:.2f}s exceeds 3x expected {expected_time:.2f}s for size {geometry_size}"
    
    def test_frequency_range_scaling(self):
        """Test performance scaling with frequency range complexity."""
        base_spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),  # Narrow band
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        wide_spec = AntennaSpec(
            frequency_range=(1e9, 10e9),  # Wide band
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        geometry = np.random.rand(20, 20, 4) * 0.5
        
        # Mock solver with frequency-dependent simulation time
        def mock_simulate_frequency_dependent(geom, frequency, **kwargs):
            # Simulate frequency-dependent computation time
            freq_ghz = frequency / 1e9
            time.sleep(0.001 * freq_ghz)  # 1ms per GHz
            return Mock(gain_dbi=5.0, vswr=1.5)
        
        # Test narrow band performance
        narrow_optimizer = LMAOptimizer(base_spec, n_iterations=3)
        narrow_optimizer.solver = Mock()
        narrow_optimizer.solver.simulate = mock_simulate_frequency_dependent
        
        start_time = time.time()
        narrow_result = narrow_optimizer.optimize(geometry)
        narrow_time = time.time() - start_time
        
        # Test wide band performance
        wide_optimizer = LMAOptimizer(wide_spec, n_iterations=3)
        wide_optimizer.solver = Mock()
        wide_optimizer.solver.simulate = mock_simulate_frequency_dependent
        
        start_time = time.time()
        wide_result = wide_optimizer.optimize(geometry)
        wide_time = time.time() - start_time
        
        # Wide band should take more time but not excessively
        time_ratio = wide_time / narrow_time
        assert 1.0 <= time_ratio <= 10.0, \
            f"Wide band time ratio {time_ratio:.2f} outside expected range [1.0, 10.0]"


class TestResourceUtilizationBenchmarks:
    """Test efficient resource utilization."""
    
    def test_cpu_utilization_efficiency(self):
        """Test CPU utilization efficiency."""
        processor = ConcurrentProcessor(n_workers=psutil.cpu_count())
        
        def cpu_intensive_task(data):
            # CPU-intensive computation
            return np.fft.fft2(data).sum()
        
        test_data = [np.random.rand(64, 64) for _ in range(psutil.cpu_count() * 2)]
        
        # Monitor CPU usage during processing
        cpu_before = psutil.cpu_percent(interval=0.1)
        
        start_time = time.time()
        futures = []
        for data in test_data:
            future_id = processor.submit_task(cpu_intensive_task, data)
            futures.append(future_id)
        
        # Wait for completion
        results = []
        for future_id in futures:
            result = processor.get_result(future_id, timeout=30.0)
            results.append(result)
        
        duration = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=0.1)
        
        # CPU should be well utilized during processing
        expected_min_cpu = 50  # At least 50% utilization expected
        assert cpu_after > expected_min_cpu or duration < 1.0, \
            f"Low CPU utilization: {cpu_after:.1f}% < {expected_min_cpu}% and duration {duration:.2f}s >= 1.0s"
        
        # Should complete in reasonable time
        expected_max_duration = len(test_data) * 0.5  # 0.5s per task maximum
        assert duration < expected_max_duration, \
            f"Processing too slow: {duration:.2f}s > {expected_max_duration:.2f}s"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations that might leak memory
        for iteration in range(50):
            # Create and optimize geometry
            geometry = np.random.rand(16, 16, 4) * 0.5
            
            spec = AntennaSpec(
                frequency_range=(2.4e9, 2.5e9),
                substrate=SubstrateMaterial.ROGERS_4003C,
                metal=LiquidMetalType.GALINSTAN
            )
            
            optimizer = LMAOptimizer(spec, n_iterations=2)
            
            # Mock solver to avoid actual computation
            mock_solver = Mock()
            mock_solver.simulate.return_value = Mock(gain_dbi=5.0, vswr=1.5)
            optimizer.solver = mock_solver
            
            result = optimizer.optimize(geometry)
            
            # Force garbage collection periodically
            if iteration % 10 == 0:
                import gc
                gc.collect()
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                max_acceptable_growth = 100 + iteration * 2  # MB, allowing some growth
                assert memory_growth < max_acceptable_growth, \
                    f"Potential memory leak: {memory_growth:.1f}MB growth at iteration {iteration}"


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.fixture
    def performance_baseline(self):
        """Performance baseline measurements."""
        return {
            'fdtd_setup_time': 0.5,      # seconds
            'optimization_iteration': 0.1, # seconds per iteration
            'cache_hit_time': 0.001,      # seconds
            'memory_per_geometry': 10,    # MB
            'concurrent_speedup': 2.0,    # times
        }
    
    def test_fdtd_performance_regression(self, performance_baseline):
        """Test for FDTD performance regression."""
        solver = DifferentiableFDTD(resolution=2e-3)
        geometry = np.random.rand(32, 32, 6) * 0.5
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        start_time = time.time()
        solver.set_grid_size(geometry, spec)
        setup_time = time.time() - start_time
        
        baseline = performance_baseline['fdtd_setup_time']
        regression_threshold = baseline * 1.5  # 50% slower is regression
        
        assert setup_time < regression_threshold, \
            f"FDTD setup regression: {setup_time:.3f}s > {regression_threshold:.3f}s (baseline: {baseline:.3f}s)"
    
    def test_optimization_performance_regression(self, performance_baseline):
        """Test for optimization performance regression."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        optimizer = LMAOptimizer(spec, n_iterations=10)
        
        # Mock consistent solver
        mock_solver = Mock()
        mock_solver.simulate.return_value = Mock(gain_dbi=5.0, vswr=1.5)
        optimizer.solver = mock_solver
        
        geometry = np.random.rand(16, 16, 4) * 0.5
        
        start_time = time.time()
        result = optimizer.optimize(geometry)
        total_time = time.time() - start_time
        
        iterations = len(result.objective_history)
        avg_iteration_time = total_time / iterations if iterations > 0 else float('inf')
        
        baseline = performance_baseline['optimization_iteration']
        regression_threshold = baseline * 2.0  # 100% slower is regression
        
        assert avg_iteration_time < regression_threshold, \
            f"Optimization regression: {avg_iteration_time:.3f}s/iter > {regression_threshold:.3f}s/iter (baseline: {baseline:.3f}s/iter)"


class TestSystemHealthBenchmarks:
    """Test system health and stability under load."""
    
    def test_system_stability_under_load(self):
        """Test system stability under sustained load."""
        diagnostics = SystemDiagnostics()
        
        # Get baseline metrics
        baseline_metrics = diagnostics.get_system_metrics()
        
        # Apply sustained load for a period
        load_duration = 10  # seconds
        processor = ConcurrentProcessor(n_workers=4)
        
        def sustained_task():
            # CPU and memory intensive task
            data = np.random.rand(100, 100)
            result = np.fft.fft2(data)
            return np.sum(result.real)
        
        start_time = time.time()
        futures = []
        
        # Submit tasks continuously
        while time.time() - start_time < load_duration:
            future_id = processor.submit_task(sustained_task)
            futures.append(future_id)
            time.sleep(0.1)  # Small delay between submissions
        
        # Wait for completion
        for future_id in futures:
            try:
                processor.get_result(future_id, timeout=5.0)
            except:
                pass  # Ignore timeout errors for this test
        
        # Get final metrics
        final_metrics = diagnostics.get_system_metrics()
        
        # System should remain stable
        memory_increase = final_metrics.memory_used_gb - baseline_metrics.memory_used_gb
        assert memory_increase < 1.0, \
            f"Excessive memory increase under load: {memory_increase:.2f}GB"
        
        # CPU usage should return to reasonable levels after load
        time.sleep(2)  # Allow system to settle
        settle_metrics = diagnostics.get_system_metrics()
        
        assert settle_metrics.cpu_usage_percent < 80, \
            f"System did not settle after load: {settle_metrics.cpu_usage_percent:.1f}% CPU"
    
    def test_resource_cleanup_effectiveness(self):
        """Test that resources are properly cleaned up."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create and destroy many objects
        for _ in range(100):
            # Create large temporary objects
            large_arrays = [np.random.rand(100, 100) for _ in range(10)]
            
            # Process them
            results = [np.sum(arr) for arr in large_arrays]
            
            # Explicitly delete references
            del large_arrays
            del results
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Wait for cleanup
        time.sleep(1)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal after cleanup
        max_acceptable_growth = 50  # MB
        assert memory_growth < max_acceptable_growth, \
            f"Poor resource cleanup: {memory_growth:.1f}MB growth > {max_acceptable_growth}MB"


if __name__ == '__main__':
    # Run with performance markers
    pytest.main([__file__, "-m", "not slow", "-v"])