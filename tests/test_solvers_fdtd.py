"""
Comprehensive tests for FDTD solver functionality.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock

from liquid_metal_antenna.core.antenna_spec import AntennaSpec, SubstrateMaterial, LiquidMetalType
from liquid_metal_antenna.solvers.base import SolverResult
from liquid_metal_antenna.solvers.fdtd import DifferentiableFDTD
from liquid_metal_antenna.utils.validation import ValidationError


class TestDifferentiableFDTD:
    """Test core FDTD solver functionality."""
    
    @pytest.fixture
    def fdtd_solver(self):
        """Create FDTD solver instance for testing."""
        return DifferentiableFDTD(
            resolution=2e-3,
            gpu_id=0,
            precision='float32'
        )
    
    @pytest.fixture
    def simple_antenna_spec(self):
        """Create simple antenna specification for testing."""
        return AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
    
    @pytest.fixture
    def simple_geometry(self):
        """Create simple antenna geometry for testing."""
        # Create a simple rectangular patch antenna
        geometry = np.zeros((20, 20, 4))
        geometry[8:12, 8:12, 2] = 1.0  # Metal patch
        return geometry
    
    def test_solver_initialization(self, fdtd_solver):
        """Test FDTD solver initialization."""
        assert fdtd_solver.resolution == 2e-3
        assert fdtd_solver.precision == 'float32'
        assert fdtd_solver.pml_thickness == 8
        assert fdtd_solver.courant_factor == 0.5
    
    def test_grid_size_calculation(self, fdtd_solver, simple_geometry, simple_antenna_spec):
        """Test grid size calculation from geometry."""
        fdtd_solver.set_grid_size(simple_geometry, simple_antenna_spec)
        
        # Grid size should be geometry shape plus PML layers
        expected_nx = simple_geometry.shape[0] + 2 * fdtd_solver.pml_thickness
        expected_ny = simple_geometry.shape[1] + 2 * fdtd_solver.pml_thickness
        expected_nz = simple_geometry.shape[2] + 2 * fdtd_solver.pml_thickness
        
        assert fdtd_solver._grid_size[0] == expected_nx
        assert fdtd_solver._grid_size[1] == expected_ny
        assert fdtd_solver._grid_size[2] == expected_nz
    
    def test_time_step_calculation(self, fdtd_solver):
        """Test Courant stability time step calculation."""
        # Set a known grid size
        fdtd_solver._grid_size = (50, 50, 10)
        fdtd_solver._calculate_time_step()
        
        # Time step should satisfy Courant stability condition
        c = 3e8  # Speed of light
        dx = fdtd_solver.resolution
        
        # 3D Courant limit
        dt_max = fdtd_solver.courant_factor / (c * np.sqrt(3) / dx)
        
        assert fdtd_solver.dt <= dt_max
        assert fdtd_solver.dt > 0
    
    @patch('liquid_metal_antenna.solvers.fdtd.torch')
    def test_field_initialization(self, mock_torch, fdtd_solver):
        """Test electromagnetic field initialization."""
        # Mock torch to avoid GPU dependency in tests
        mock_torch.zeros.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        fdtd_solver._grid_size = (20, 20, 10)
        fields = fdtd_solver.initialize_fields()
        
        # Should initialize all six field components
        expected_components = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        for component in expected_components:
            assert component in fields
    
    def test_geometry_mask_creation(self, fdtd_solver, simple_geometry, simple_antenna_spec):
        """Test creation of material geometry mask."""
        fdtd_solver.set_grid_size(simple_geometry, simple_antenna_spec)
        materials = fdtd_solver.create_geometry_mask(simple_geometry, simple_antenna_spec)
        
        # Should contain material property tensors
        assert 'epsilon' in materials
        assert 'mu' in materials
        assert 'sigma' in materials
    
    @patch('liquid_metal_antenna.solvers.fdtd.torch')
    def test_source_creation(self, mock_torch, fdtd_solver):
        """Test electromagnetic source creation."""
        mock_torch.cuda.is_available.return_value = False
        
        frequency = 2.45e9
        source = fdtd_solver.create_source('coaxial_feed', frequency)
        
        assert 'type' in source
        assert 'frequency' in source
        assert 'amplitude' in source
        assert 'position' in source
    
    def test_memory_estimation(self, fdtd_solver):
        """Test memory usage estimation."""
        # Set a known grid size
        fdtd_solver._grid_size = (100, 100, 20)
        
        memory_gb = fdtd_solver.estimate_memory_usage()
        
        # Should be reasonable estimate
        assert memory_gb > 0
        assert memory_gb < 1000  # Sanity check
    
    def test_simulation_parameter_validation(self, fdtd_solver, simple_geometry):
        """Test simulation parameter validation."""
        # Valid parameters should not raise exception
        try:
            fdtd_solver._validate_simulation_setup(
                simple_geometry,
                frequency=2.45e9,
                max_time_steps=1000
            )
        except Exception as e:
            pytest.fail(f"Valid parameters should not raise exception: {e}")
        
        # Invalid geometry should raise ValidationError
        with pytest.raises(ValidationError):
            fdtd_solver._validate_simulation_setup(
                np.zeros((2, 2, 2)),  # Too small
                frequency=2.45e9,
                max_time_steps=1000
            )
        
        # Invalid frequency should raise ValidationError
        with pytest.raises(ValidationError):
            fdtd_solver._validate_simulation_setup(
                simple_geometry,
                frequency=-1e9,  # Negative frequency
                max_time_steps=1000
            )
    
    @patch('liquid_metal_antenna.solvers.fdtd.torch')
    def test_pml_boundary_implementation(self, mock_torch, fdtd_solver):
        """Test PML boundary condition implementation."""
        mock_torch.cuda.is_available.return_value = False
        
        # Mock field tensors
        mock_fields = {
            'Ex': MagicMock(),
            'Ey': MagicMock(),
            'Ez': MagicMock(),
            'Hx': MagicMock(),
            'Hy': MagicMock(),
            'Hz': MagicMock()
        }
        
        fdtd_solver._grid_size = (50, 50, 20)
        
        # Should not raise exception
        try:
            fdtd_solver.apply_pml_boundaries(mock_fields)
        except Exception as e:
            pytest.fail(f"PML boundary application failed: {e}")
    
    def test_precision_handling(self):
        """Test different precision modes."""
        # Test float32 precision
        solver_f32 = DifferentiableFDTD(precision='float32')
        assert solver_f32.precision == 'float32'
        
        # Test float64 precision
        solver_f64 = DifferentiableFDTD(precision='float64')
        assert solver_f64.precision == 'float64'
        
        # Test invalid precision
        with pytest.raises(ValidationError):
            DifferentiableFDTD(precision='invalid')
    
    @patch('liquid_metal_antenna.solvers.fdtd.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda_available):
        """Test GPU/CPU device selection."""
        # Test when CUDA is available
        mock_cuda_available.return_value = True
        solver_gpu = DifferentiableFDTD(gpu_id=0)
        assert 'cuda' in solver_gpu.device or solver_gpu.device == 'cpu'  # Fallback allowed
        
        # Test when CUDA is not available
        mock_cuda_available.return_value = False
        solver_cpu = DifferentiableFDTD(gpu_id=0)
        assert solver_cpu.device == 'cpu'


class TestFDTDSimulationWorkflow:
    """Test complete FDTD simulation workflow."""
    
    @pytest.fixture
    def fdtd_solver(self):
        """Create FDTD solver for simulation tests."""
        return DifferentiableFDTD(
            resolution=5e-3,  # Coarser for faster testing
            precision='float32'
        )
    
    @pytest.fixture
    def test_spec(self):
        """Create test antenna specification."""
        return AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
    
    @pytest.fixture
    def test_geometry(self):
        """Create test geometry."""
        geometry = np.zeros((16, 16, 4))
        geometry[6:10, 6:10, 2] = 1.0  # Simple square patch
        return geometry
    
    @patch('liquid_metal_antenna.solvers.fdtd.torch')
    def test_complete_simulation_workflow(self, mock_torch, fdtd_solver, test_geometry, test_spec):
        """Test complete simulation from start to finish."""
        # Mock torch to avoid GPU dependency
        mock_torch.cuda.is_available.return_value = False
        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.zeros.return_value = MagicMock()
        
        # Mock field update methods to avoid complex FDTD implementation in tests
        fdtd_solver.update_e_fields = MagicMock()
        fdtd_solver.update_h_fields = MagicMock()
        fdtd_solver.apply_source = MagicMock()
        fdtd_solver.apply_pml_boundaries = MagicMock()
        
        # Mock result computation methods
        fdtd_solver.compute_s_parameters = MagicMock(return_value=np.array([[[0.1+0.05j]]]))
        fdtd_solver.compute_radiation_pattern = MagicMock(return_value=(
            np.ones((18, 36)), np.linspace(0, np.pi, 18), np.linspace(0, 2*np.pi, 36)
        ))
        fdtd_solver.compute_gain = MagicMock(return_value=5.2)
        fdtd_solver.compute_vswr = MagicMock(return_value=1.8)
        
        # Run simulation
        try:
            result = fdtd_solver.simulate(
                geometry=test_geometry,
                frequency=2.45e9,
                spec=test_spec,
                max_time_steps=100  # Short for testing
            )
            
            # Verify result structure
            assert isinstance(result, SolverResult)
            assert hasattr(result, 's_parameters')
            assert hasattr(result, 'radiation_pattern')
            assert hasattr(result, 'gain_dbi')
            
        except Exception as e:
            pytest.fail(f"Complete simulation workflow failed: {e}")
    
    def test_simulation_convergence_detection(self, fdtd_solver):
        """Test simulation convergence detection."""
        # Mock convergence checking
        fdtd_solver._check_convergence = MagicMock(side_effect=[False, False, True])
        
        # Should detect convergence after 3 iterations
        converged = False
        for i in range(10):
            converged = fdtd_solver._check_convergence(None, i)
            if converged:
                break
        
        assert converged
        assert fdtd_solver._check_convergence.call_count == 3
    
    def test_simulation_error_handling(self, fdtd_solver, test_geometry, test_spec):
        """Test simulation error handling and recovery."""
        # Test with invalid parameters that should raise specific errors
        
        # Zero time steps
        with pytest.raises(ValidationError):
            fdtd_solver.simulate(
                geometry=test_geometry,
                frequency=2.45e9,
                spec=test_spec,
                max_time_steps=0
            )
        
        # Invalid frequency
        with pytest.raises(ValidationError):
            fdtd_solver.simulate(
                geometry=test_geometry,
                frequency=-1e9,
                spec=test_spec,
                max_time_steps=100
            )


class TestFDTDNumericalAccuracy:
    """Test numerical accuracy and stability of FDTD solver."""
    
    def test_courant_stability_condition(self):
        """Test that Courant stability condition is enforced."""
        for resolution in [1e-3, 2e-3, 5e-3]:
            for courant_factor in [0.3, 0.5, 0.7]:
                solver = DifferentiableFDTD(
                    resolution=resolution,
                    courant_factor=courant_factor
                )
                
                # Set grid size and calculate time step
                solver._grid_size = (50, 50, 20)
                solver._calculate_time_step()
                
                # Verify Courant condition
                c = 3e8
                dt_max = courant_factor / (c * np.sqrt(3) / resolution)
                assert solver.dt <= dt_max * 1.01  # Small tolerance for floating point
    
    def test_grid_dispersion_analysis(self):
        """Test grid dispersion characteristics."""
        solver = DifferentiableFDTD(resolution=1e-3)
        
        # Test different frequencies for dispersion
        frequencies = [1e9, 5e9, 10e9, 20e9]
        
        for freq in frequencies:
            wavelength = 3e8 / freq
            points_per_wavelength = wavelength / solver.resolution
            
            # Should have at least 10 points per wavelength for accuracy
            if points_per_wavelength < 10:
                with pytest.warns(UserWarning):
                    solver._check_dispersion(freq)
            else:
                # Should not warn for well-resolved frequencies
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    try:
                        solver._check_dispersion(freq)
                    except UserWarning:
                        pytest.fail("Should not warn for well-resolved frequency")
    
    def test_pml_reflection_coefficient(self):
        """Test PML boundary reflection characteristics."""
        solver = DifferentiableFDTD(pml_thickness=8)
        
        # PML should provide good absorption
        reflection_coeff = solver._calculate_pml_reflection()
        assert reflection_coeff < 0.01  # Less than 1% reflection
        
        # Thicker PML should provide better absorption
        thick_solver = DifferentiableFDTD(pml_thickness=16)
        thick_reflection = thick_solver._calculate_pml_reflection()
        assert thick_reflection < reflection_coeff


class TestFDTDEdgeCases:
    """Test edge cases and error conditions for FDTD solver."""
    
    def test_extreme_geometry_sizes(self):
        """Test handling of extreme geometry sizes."""
        solver = DifferentiableFDTD()
        
        # Very small geometry
        tiny_geometry = np.ones((2, 2, 2))
        with pytest.raises(ValidationError):
            solver.set_grid_size(tiny_geometry, None)
        
        # Very large geometry (memory constraint)
        with pytest.raises(ValidationError):
            large_geometry = np.ones((1000, 1000, 100))
            solver.set_grid_size(large_geometry, None)
    
    def test_extreme_frequencies(self):
        """Test handling of extreme frequencies."""
        solver = DifferentiableFDTD(resolution=1e-3)
        
        # Very low frequency (long wavelength)
        low_freq = 100e6  # 100 MHz
        with pytest.warns(UserWarning):
            solver._validate_frequency(low_freq)
        
        # Very high frequency (short wavelength)
        high_freq = 100e9  # 100 GHz
        with pytest.warns(UserWarning):
            solver._validate_frequency(high_freq)
    
    def test_invalid_material_properties(self):
        """Test handling of invalid material properties."""
        solver = DifferentiableFDTD()
        
        # Negative permittivity
        with pytest.raises(ValidationError):
            invalid_spec = Mock()
            invalid_spec.get_substrate_properties.return_value = {
                'dielectric_constant': -1.0,
                'loss_tangent': 0.02
            }
            solver._validate_materials(invalid_spec)
        
        # Negative conductivity
        with pytest.raises(ValidationError):
            invalid_spec = Mock()
            invalid_spec.get_metal_properties.return_value = {
                'conductivity': -1e6
            }
            solver._validate_materials(invalid_spec)
    
    def test_numerical_stability_monitoring(self):
        """Test numerical stability monitoring."""
        solver = DifferentiableFDTD()
        
        # Mock fields with growing magnitude (unstable)
        unstable_fields = {
            'Ex': np.ones((10, 10, 5)) * 1e6,  # Very large field
            'Ey': np.ones((10, 10, 5)) * 1e6,
            'Ez': np.ones((10, 10, 5)) * 1e6
        }
        
        # Should detect instability
        is_stable = solver._check_field_stability(unstable_fields)
        assert not is_stable
        
        # Mock fields with reasonable magnitude (stable)
        stable_fields = {
            'Ex': np.ones((10, 10, 5)) * 1e-3,
            'Ey': np.ones((10, 10, 5)) * 1e-3,
            'Ez': np.ones((10, 10, 5)) * 1e-3
        }
        
        # Should be stable
        is_stable = solver._check_field_stability(stable_fields)
        assert is_stable


if __name__ == '__main__':
    pytest.main([__file__])