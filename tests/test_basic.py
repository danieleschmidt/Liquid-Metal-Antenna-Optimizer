#!/usr/bin/env python3
"""
Basic tests for Liquid Metal Antenna Optimizer.
"""

import unittest
import numpy as np
import torch
import tempfile
import os

from liquid_metal_antenna import AntennaSpec, LMAOptimizer, DifferentiableFDTD
from liquid_metal_antenna.designs import ReconfigurablePatch, LiquidMetalMonopole
from liquid_metal_antenna.liquid_metal import GalinStanModel, FlowSimulator
from liquid_metal_antenna.core.antenna_spec import SubstrateMaterial, LiquidMetalType


class TestAntennaSpec(unittest.TestCase):
    """Test AntennaSpec functionality."""
    
    def test_basic_spec_creation(self):
        """Test basic antenna specification creation."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            size_constraint=(30, 30, 3)
        )
        
        self.assertEqual(spec.frequency_range.start, 2.4e9)
        self.assertEqual(spec.frequency_range.stop, 2.5e9)
        self.assertAlmostEqual(spec.frequency_range.center, 2.45e9)
        self.assertAlmostEqual(spec.frequency_range.bandwidth, 0.1e9)
        
        self.assertEqual(spec.metal, LiquidMetalType.GALINSTAN)
        self.assertAlmostEqual(spec.substrate.dielectric_constant, 3.38, places=2)
    
    def test_spec_validation(self):
        """Test specification validation."""
        # Test invalid frequency range
        with self.assertRaises(ValueError):
            AntennaSpec(frequency_range=(2.5e9, 2.4e9))  # start > stop
        
        # Test invalid VSWR
        with self.assertRaises(ValueError):
            AntennaSpec(
                frequency_range=(2.4e9, 2.5e9),
                max_vswr=0.5  # VSWR < 1.0
            )
    
    def test_wavelength_calculations(self):
        """Test wavelength calculations."""
        spec = AntennaSpec(frequency_range=(2.45e9, 2.45e9))
        
        # Free space wavelength at 2.45 GHz should be ~122 mm
        wavelength = spec.get_wavelength_at_center()
        self.assertAlmostEqual(wavelength, 122.4, delta=1.0)
        
        # Substrate wavelength should be shorter
        sub_wavelength = spec.get_substrate_wavelength_at_center()
        self.assertLess(sub_wavelength, wavelength)
    
    def test_spec_serialization(self):
        """Test specification serialization/deserialization."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Convert to dict and back
        spec_dict = spec.to_dict()
        spec_restored = AntennaSpec.from_dict(spec_dict)
        
        self.assertEqual(spec.frequency_range.start, spec_restored.frequency_range.start)
        self.assertEqual(spec.metal, spec_restored.metal)
        self.assertAlmostEqual(spec.substrate.dielectric_constant, 
                              spec_restored.substrate.dielectric_constant)


class TestReconfigurablePatch(unittest.TestCase):
    """Test reconfigurable patch antenna."""
    
    def setUp(self):
        """Set up test patch."""
        self.patch = ReconfigurablePatch(
            substrate_height=1.6,
            dielectric_constant=4.4,
            n_channels=4,
            channel_width=0.5
        )
    
    def test_patch_creation(self):
        """Test patch antenna creation."""
        self.assertEqual(self.patch.n_channels, 4)
        self.assertEqual(len(self.patch.channels), 4)
        self.assertFalse(any(self.patch.channel_states))  # All channels initially empty
    
    def test_channel_control(self):
        """Test channel state control."""
        # Set individual channel
        self.patch.set_channel_state(0, True)
        self.assertTrue(self.patch.channel_states[0])
        self.assertFalse(self.patch.channel_states[1])
        
        # Set all channels
        config = [True, False, True, False]
        self.patch.set_configuration(config)
        np.testing.assert_array_equal(self.patch.channel_states, config)
    
    def test_frequency_calculation(self):
        """Test resonant frequency calculation."""
        freq = self.patch.get_resonant_frequency()
        self.assertGreater(freq, 1e9)  # Should be > 1 GHz
        self.assertLess(freq, 10e9)    # Should be < 10 GHz
        
        # Different modes should give different frequencies
        freq_tm10 = self.patch.get_resonant_frequency('TM10')
        freq_tm01 = self.patch.get_resonant_frequency('TM01')
        self.assertNotAlmostEqual(freq_tm10, freq_tm01, delta=1e6)
    
    def test_geometry_tensor_creation(self):
        """Test geometry tensor generation."""
        geometry = self.patch.create_geometry_tensor()
        
        # Should be 3D tensor
        self.assertEqual(len(geometry.shape), 3)
        
        # Should contain some conductor material (1.0 values)
        self.assertGreater(torch.sum(geometry), 0)
        
        # Should be mostly air (0.0 values)
        self.assertGreater(torch.sum(geometry == 0), torch.sum(geometry == 1))
    
    def test_configuration_export(self):
        """Test configuration export/import."""
        self.patch.set_configuration([True, False, True, False])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            self.patch.export_config(filename)
            self.assertTrue(os.path.exists(filename))
            
            # Load configuration
            loaded_patch = ReconfigurablePatch.load_config(filename)
            np.testing.assert_array_equal(self.patch.channel_states, 
                                        loaded_patch.channel_states)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestLiquidMetalMonopole(unittest.TestCase):
    """Test liquid metal monopole antenna."""
    
    def setUp(self):
        """Set up test monopole."""
        self.monopole = LiquidMetalMonopole(
            max_height=30.0,  # mm
            n_segments=5
        )
    
    def test_monopole_creation(self):
        """Test monopole creation."""
        self.assertEqual(self.monopole.n_segments, 5)
        self.assertAlmostEqual(self.monopole.segment_height, 6.0)  # 30/5 = 6 mm
        self.assertTrue(self.monopole.segment_states[0])  # First segment active
    
    def test_segment_control(self):
        """Test segment control."""
        # Set active segments
        self.monopole.set_active_segments(3)
        expected = [True, True, True, False, False]
        np.testing.assert_array_equal(self.monopole.segment_states, expected)
        
        # Test custom configuration
        custom_config = [True, False, True, False, True]
        self.monopole.set_segment_configuration(custom_config)
        np.testing.assert_array_equal(self.monopole.segment_states, custom_config)
    
    def test_height_calculation(self):
        """Test active height calculation."""
        self.monopole.set_active_segments(3)
        height = self.monopole.get_active_height()
        self.assertAlmostEqual(height, 18.0)  # 3 * 6 mm
    
    def test_frequency_estimation(self):
        """Test frequency estimation."""
        self.monopole.set_active_segments(2)
        freq = self.monopole.get_resonant_frequency()
        
        # Should be reasonable frequency for 12mm monopole
        self.assertGreater(freq, 1e9)   # > 1 GHz
        self.assertLess(freq, 50e9)     # < 50 GHz
        
        # Longer antenna should have lower frequency
        self.monopole.set_active_segments(4)
        freq_long = self.monopole.get_resonant_frequency()
        self.assertLess(freq_long, freq)


class TestGalinStanModel(unittest.TestCase):
    """Test Galinstan material model."""
    
    def setUp(self):
        """Set up material model."""
        self.galinstan = GalinStanModel()
    
    def test_material_properties(self):
        """Test basic material properties."""
        # Properties at room temperature
        temp = 25.0
        sigma = self.galinstan.conductivity(temp)
        eta = self.galinstan.viscosity(temp)
        rho = self.galinstan.density(temp)
        
        # Check reasonable values
        self.assertGreater(sigma, 1e6)      # > 1 MS/m
        self.assertLess(sigma, 10e6)        # < 10 MS/m
        self.assertGreater(eta, 1e-3)       # > 1 mPa·s
        self.assertLess(eta, 10e-3)         # < 10 mPa·s
        self.assertGreater(rho, 6000)       # > 6000 kg/m³
        self.assertLess(rho, 7000)          # < 7000 kg/m³
    
    def test_temperature_dependence(self):
        """Test temperature dependence."""
        temp_low = 20.0
        temp_high = 60.0
        
        # Conductivity should decrease with temperature
        sigma_low = self.galinstan.conductivity(temp_low)
        sigma_high = self.galinstan.conductivity(temp_high)
        self.assertGreater(sigma_low, sigma_high)
        
        # Viscosity should decrease with temperature
        eta_low = self.galinstan.viscosity(temp_low)
        eta_high = self.galinstan.viscosity(temp_high)
        self.assertGreater(eta_low, eta_high)
    
    def test_skin_depth(self):
        """Test skin depth calculation."""
        freq = 2.45e9
        skin_depth = self.galinstan.skin_depth(freq, 25.0)
        
        # Should be in micrometer range for GHz frequencies
        self.assertGreater(skin_depth, 0.1e-6)  # > 0.1 μm
        self.assertLess(skin_depth, 10e-6)      # < 10 μm
    
    def test_array_input(self):
        """Test array input for properties."""
        temperatures = np.array([20, 30, 40, 50])
        
        conductivities = self.galinstan.conductivity(temperatures)
        self.assertEqual(len(conductivities), len(temperatures))
        
        viscosities = self.galinstan.viscosity(temperatures)
        self.assertEqual(len(viscosities), len(temperatures))


class TestDifferentiableFDTD(unittest.TestCase):
    """Test FDTD solver."""
    
    def setUp(self):
        """Set up FDTD solver."""
        self.solver = DifferentiableFDTD(
            resolution=2.0e-3,  # 2mm for fast testing
            precision='float32'
        )
    
    def test_solver_creation(self):
        """Test solver initialization."""
        self.assertEqual(self.solver.resolution, 2.0e-3)
        self.assertEqual(self.solver.precision, 'float32')
        self.assertIsNotNone(self.solver.device)
    
    def test_geometry_mask_creation(self):
        """Test geometry mask creation."""
        spec = AntennaSpec(frequency_range=(2.4e9, 2.5e9))
        geometry = torch.ones((16, 16, 8))  # Simple geometry
        
        materials = self.solver.create_geometry_mask(geometry, spec)
        
        # Should have correct shape [nx, ny, nz, 3] for [eps, sigma, mu]
        self.assertEqual(materials.shape[-1], 3)
        self.assertEqual(materials.shape[:3], (16, 16, 8))
    
    def test_field_initialization(self):
        """Test field initialization."""
        self.solver._grid_size = (16, 16, 8)
        fields = self.solver.initialize_fields()
        
        # Should have 6 field components
        expected_fields = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        for field_name in expected_fields:
            self.assertIn(field_name, fields)
            self.assertEqual(fields[field_name].shape, (16, 16, 8))
    
    def test_simple_simulation(self):
        """Test basic simulation run."""
        # Create simple geometry
        geometry = torch.zeros((16, 16, 8))
        geometry[6:10, 6:10, 6] = 1.0  # Small patch
        
        # Run simulation
        result = self.solver.simulate(
            geometry=geometry,
            frequency=2.45e9,
            compute_gradients=False,
            max_time_steps=50  # Very short for testing
        )
        
        # Check result structure
        self.assertIsNotNone(result.s_parameters)
        self.assertIsNotNone(result.frequencies)
        self.assertEqual(len(result.frequencies), 1)
        self.assertGreater(result.computation_time, 0)


class TestFlowSimulator(unittest.TestCase):
    """Test flow simulator."""
    
    def setUp(self):
        """Set up flow simulator."""
        self.flow_sim = FlowSimulator()
    
    def test_simulator_creation(self):
        """Test simulator initialization."""
        self.assertEqual(self.flow_sim.method, 'lattice_boltzmann')
        self.assertGreater(self.flow_sim.density, 6000)  # Galinstan density
    
    def test_channel_optimization(self):
        """Test channel design optimization."""
        actuation_points = [(10, 10), (20, 20), (30, 30)]  # mm coordinates
        
        design = self.flow_sim.optimize_channels(
            antenna_geometry="test_antenna.stl",
            actuation_points=actuation_points,
            max_pressure=5e3,  # 5 kPa
            response_time=1.0   # 1 second
        )
        
        self.assertIn('channels', design)
        self.assertEqual(len(design['channels']), len(actuation_points))
        self.assertIn('main_inlet', design)
        self.assertIn('design_pressure', design)
    
    def test_filling_simulation(self):
        """Test filling simulation."""
        # Create simple channel design
        channels = [{
            'id': 0,
            'start_point': (0, 0),
            'end_point': (0.01, 0.01),  # 1cm channel
            'width': 1e-3,              # 1mm
            'height': 0.5e-3,           # 0.5mm
            'length': 0.014             # ~1.4cm diagonal
        }]
        
        channel_design = {'channels': channels}
        
        filling_data = self.flow_sim.simulate_filling(
            channel_design=channel_design,
            inlet_pressure=2e3,  # 2 kPa
            temperature=25.0
        )
        
        self.assertIn('time_points', filling_data)
        self.assertIn('channel_fill_fraction', filling_data)
        self.assertGreater(len(filling_data['time_points']), 0)


class TestLMAOptimizer(unittest.TestCase):
    """Test LMA optimizer."""
    
    def setUp(self):
        """Set up optimizer."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            size_constraint=(20, 20, 2)  # Small for fast testing
        )
        self.optimizer = LMAOptimizer(spec=spec, device='cpu')
    
    def test_optimizer_creation(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.spec)
        self.assertIsNotNone(self.optimizer.solver)
        self.assertEqual(self.optimizer.device, 'cpu')
    
    def test_initial_geometry_creation(self):
        """Test initial geometry generation."""
        geometry = self.optimizer.create_initial_geometry(self.optimizer.spec)
        
        self.assertIsInstance(geometry, torch.Tensor)
        self.assertEqual(len(geometry.shape), 3)
        self.assertGreater(torch.sum(geometry), 0)  # Should have some conductor
    
    def test_quick_optimization(self):
        """Test very short optimization run."""
        result = self.optimizer.optimize(
            objective='max_gain',
            n_iterations=10,  # Very short for testing
            constraints={'vswr': '<2.0'}
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.geometry)
        self.assertGreater(result.iterations, 0)
        self.assertGreater(result.optimization_time, 0)
        self.assertEqual(len(result.objective_history), result.iterations)


def run_basic_tests():
    """Run all basic tests."""
    print("Running Liquid Metal Antenna Optimizer Tests...")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestAntennaSpec,
        TestReconfigurablePatch,
        TestLiquidMetalMonopole,
        TestGalinStanModel,
        TestDifferentiableFDTD,
        TestFlowSimulator,
        TestLMAOptimizer
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_basic_tests()