"""
Pytest configuration and fixtures for liquid metal antenna optimizer tests.
"""

import pytest
import numpy as np
import tempfile
import os
import warnings
from unittest.mock import Mock

# Configure numpy for consistent testing
np.random.seed(42)  # Reproducible random numbers for testing


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security-related"
    )


@pytest.fixture(scope="session")
def temp_directory():
    """Create temporary directory for test session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_torch():
    """Mock torch module for tests that don't require actual PyTorch."""
    torch_mock = Mock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.zeros.return_value = Mock()
    torch_mock.ones.return_value = Mock()
    torch_mock.from_numpy.return_value = Mock()
    torch_mock.tensor.return_value = Mock()
    torch_mock.float32 = 'float32'
    torch_mock.float64 = 'float64'
    return torch_mock


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during tests unless explicitly testing warning behavior."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


@pytest.fixture
def sample_geometry_small():
    """Small geometry for fast testing."""
    return np.random.rand(8, 8, 4) * 0.5


@pytest.fixture
def sample_geometry_medium():
    """Medium geometry for standard testing."""
    return np.random.rand(16, 16, 4) * 0.5


@pytest.fixture
def sample_geometry_large():
    """Large geometry for scalability testing."""
    return np.random.rand(32, 32, 6) * 0.5


@pytest.fixture
def patch_antenna_geometry():
    """Simple patch antenna geometry."""
    geometry = np.zeros((20, 20, 4))
    geometry[8:12, 8:12, 2] = 1.0  # Metal patch on layer 2
    return geometry


@pytest.fixture
def monopole_antenna_geometry():
    """Simple monopole antenna geometry."""
    geometry = np.zeros((16, 16, 8))
    geometry[8, 8, :] = 1.0  # Vertical monopole
    geometry[6:11, 6:11, 0] = 1.0  # Ground plane
    return geometry


@pytest.fixture
def mock_solver_result():
    """Mock solver result with typical antenna performance values."""
    result = Mock()
    result.s_parameters = np.array([[[0.1+0.05j]]])
    result.frequencies = np.array([2.45e9])
    result.radiation_pattern = np.ones((18, 36))
    result.theta_angles = np.linspace(0, np.pi, 18)
    result.phi_angles = np.linspace(0, 2*np.pi, 36)
    result.gain_dbi = 5.2
    result.max_gain_dbi = 5.2
    result.vswr = 1.8
    result.efficiency = 0.85
    result.converged = True
    result.iterations = 100
    result.computation_time = 1.5
    return result


@pytest.fixture
def mock_fdtd_solver(mock_solver_result):
    """Mock FDTD solver that returns consistent results."""
    solver = Mock()
    solver.simulate.return_value = mock_solver_result
    solver.resolution = 2e-3
    solver.precision = 'float32'
    solver.device = 'cpu'
    return solver


@pytest.fixture
def performance_targets_basic():
    """Basic performance targets for testing."""
    return {
        'min_gain_dbi': 5.0,
        'max_vswr': 2.0,
        'min_efficiency': 0.8,
        'bandwidth_requirement': 'standard'
    }


@pytest.fixture
def performance_targets_high():
    """High performance targets for challenging tests."""
    return {
        'min_gain_dbi': 10.0,
        'max_vswr': 1.5,
        'min_efficiency': 0.9,
        'bandwidth_requirement': 'wide',
        'sidelobe_level': -20
    }


@pytest.fixture
def frequency_ranges():
    """Various frequency ranges for testing."""
    return {
        'ism_2_4': (2.4e9, 2.485e9),
        'ism_5_8': (5.725e9, 5.875e9),
        'wifi_6': (5.925e9, 7.125e9),
        'mmwave_28': (27.5e9, 28.35e9),
        'wideband': (1e9, 6e9)
    }


@pytest.fixture
def substrate_properties():
    """Various substrate material properties for testing."""
    return {
        'low_loss': {'dielectric_constant': 2.2, 'loss_tangent': 0.001, 'thickness_mm': 1.6},
        'standard': {'dielectric_constant': 4.4, 'loss_tangent': 0.02, 'thickness_mm': 1.6},
        'high_er': {'dielectric_constant': 10.2, 'loss_tangent': 0.003, 'thickness_mm': 1.27},
        'thin': {'dielectric_constant': 3.38, 'loss_tangent': 0.0027, 'thickness_mm': 0.508}
    }


@pytest.fixture
def optimization_configs():
    """Various optimization configurations for testing."""
    return {
        'fast': {'n_iterations': 5, 'learning_rate': 0.05, 'population_size': 20},
        'standard': {'n_iterations': 50, 'learning_rate': 0.01, 'population_size': 50},
        'thorough': {'n_iterations': 200, 'learning_rate': 0.005, 'population_size': 100},
        'multi_objective': {'n_iterations': 50, 'optimization_mode': 'multi_objective'},
    }


class GeometryValidator:
    """Utility class for validating antenna geometries in tests."""
    
    @staticmethod
    def is_valid_geometry(geometry):
        """Check if geometry is valid for antenna simulation."""
        if not isinstance(geometry, np.ndarray):
            return False, "Geometry must be numpy array"
        
        if geometry.ndim != 3:
            return False, f"Geometry must be 3D, got {geometry.ndim}D"
        
        if not (0 <= geometry.min() and geometry.max() <= 1):
            return False, f"Geometry values must be in [0,1], got [{geometry.min():.3f}, {geometry.max():.3f}]"
        
        if geometry.shape[0] < 4 or geometry.shape[1] < 4 or geometry.shape[2] < 2:
            return False, f"Geometry too small: {geometry.shape}"
        
        return True, "Valid geometry"
    
    @staticmethod
    def has_conductor(geometry, min_fraction=0.001):
        """Check if geometry has sufficient conductor material."""
        conductor_fraction = np.mean(geometry > 0.5)
        return conductor_fraction >= min_fraction
    
    @staticmethod
    def is_connected(geometry, layer=None):
        """Check if conductor regions are connected (simplified check)."""
        if layer is not None:
            geometry_2d = geometry[:, :, layer]
        else:
            geometry_2d = np.max(geometry, axis=2)  # Project to 2D
        
        # Simple connectivity check: conductor regions should not be isolated pixels
        conductor_mask = geometry_2d > 0.5
        if not np.any(conductor_mask):
            return False
        
        # Count isolated pixels (no adjacent conductor)
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(conductor_mask)
        isolated_pixels = np.sum(conductor_mask & ~dilated)
        total_conductor = np.sum(conductor_mask)
        
        # Less than 20% isolated pixels is considered "connected enough"
        return (isolated_pixels / total_conductor) < 0.2 if total_conductor > 0 else False


@pytest.fixture
def geometry_validator():
    """Geometry validation utility for tests."""
    return GeometryValidator()


class PerformanceChecker:
    """Utility class for checking antenna performance in tests."""
    
    @staticmethod
    def meets_gain_target(result, min_gain_dbi):
        """Check if result meets minimum gain requirement."""
        if hasattr(result, 'gain_dbi'):
            return result.gain_dbi >= min_gain_dbi
        return False
    
    @staticmethod
    def meets_vswr_target(result, max_vswr):
        """Check if result meets maximum VSWR requirement."""
        if hasattr(result, 'vswr'):
            return result.vswr <= max_vswr
        return False
    
    @staticmethod
    def meets_efficiency_target(result, min_efficiency):
        """Check if result meets minimum efficiency requirement."""
        if hasattr(result, 'efficiency'):
            return result.efficiency >= min_efficiency
        return False
    
    @staticmethod
    def check_all_targets(result, targets):
        """Check if result meets all performance targets."""
        checks = []
        
        if 'min_gain_dbi' in targets:
            checks.append(PerformanceChecker.meets_gain_target(result, targets['min_gain_dbi']))
        
        if 'max_vswr' in targets:
            checks.append(PerformanceChecker.meets_vswr_target(result, targets['max_vswr']))
        
        if 'min_efficiency' in targets:
            checks.append(PerformanceChecker.meets_efficiency_target(result, targets['min_efficiency']))
        
        return all(checks) if checks else True


@pytest.fixture
def performance_checker():
    """Performance checking utility for tests."""
    return PerformanceChecker()


# Test data generators
def generate_test_geometries(n_geometries=10, shape=(16, 16, 4), density=0.3):
    """Generate multiple test geometries with specified density."""
    geometries = []
    for i in range(n_geometries):
        np.random.seed(42 + i)  # Reproducible but different
        geometry = np.random.rand(*shape)
        geometry = (geometry < density).astype(float)  # Binary geometry
        geometries.append(geometry)
    return geometries


@pytest.fixture
def test_geometries_small():
    """Set of small test geometries."""
    return generate_test_geometries(5, (8, 8, 4), 0.2)


@pytest.fixture
def test_geometries_medium():
    """Set of medium test geometries."""
    return generate_test_geometries(3, (16, 16, 4), 0.3)


# Skip conditions for different test types
def pytest_runtest_setup(item):
    """Setup function to handle test skipping based on markers and conditions."""
    
    # Skip GPU tests if no GPU available
    if "gpu" in item.keywords:
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")
    
    # Skip slow tests if not explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--runslow", default=False):
        pytest.skip("Slow test skipped (use --runslow to run)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--rungpu", action="store_true", default=False,
        help="run GPU tests"
    )


# Custom assertion helpers
class Assertions:
    """Custom assertion helpers for antenna testing."""
    
    @staticmethod
    def assert_geometry_valid(geometry, name="geometry"):
        """Assert that geometry is valid for antenna simulation."""
        validator = GeometryValidator()
        is_valid, message = validator.is_valid_geometry(geometry)
        assert is_valid, f"{name} validation failed: {message}"
    
    @staticmethod
    def assert_performance_reasonable(result, name="result"):
        """Assert that performance metrics are reasonable."""
        if hasattr(result, 'gain_dbi'):
            assert -10 <= result.gain_dbi <= 30, f"{name} gain {result.gain_dbi:.1f} dBi outside reasonable range [-10, 30]"
        
        if hasattr(result, 'vswr'):
            assert 1.0 <= result.vswr <= 10.0, f"{name} VSWR {result.vswr:.2f} outside reasonable range [1.0, 10.0]"
        
        if hasattr(result, 'efficiency'):
            assert 0.0 <= result.efficiency <= 1.0, f"{name} efficiency {result.efficiency:.3f} outside range [0, 1]"
    
    @staticmethod
    def assert_optimization_progress(result, name="optimization"):
        """Assert that optimization made progress."""
        assert hasattr(result, 'objective_history'), f"{name} missing objective history"
        assert len(result.objective_history) > 0, f"{name} no optimization iterations"
        
        # Check for general improvement trend (allowing some noise)
        history = result.objective_history
        if len(history) >= 5:
            early_avg = np.mean(history[:len(history)//3])
            late_avg = np.mean(history[-len(history)//3:])
            improvement = (late_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0
            
            # Allow for either improvement or stability (some problems might start near optimum)
            assert improvement >= -0.5, f"{name} showed significant degradation: {improvement:.1%}"


@pytest.fixture
def assert_antenna():
    """Custom antenna-specific assertions."""
    return Assertions()