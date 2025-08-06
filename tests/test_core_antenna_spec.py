"""
Comprehensive tests for core antenna specification functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from liquid_metal_antenna.core.antenna_spec import (
    AntennaSpec, MaterialProperties, FrequencyRange, SizeConstraint,
    SubstrateMaterial, LiquidMetalType
)
from liquid_metal_antenna.utils.validation import ValidationError


class TestFrequencyRange:
    """Test FrequencyRange functionality."""
    
    def test_valid_frequency_range(self):
        """Test creating valid frequency range."""
        freq_range = FrequencyRange(2.4e9, 2.5e9)
        assert freq_range.start == 2.4e9
        assert freq_range.stop == 2.5e9
        assert freq_range.center == 2.45e9
        assert freq_range.bandwidth == 0.1e9
    
    def test_invalid_frequency_order(self):
        """Test invalid frequency order raises error."""
        with pytest.raises(ValidationError):
            FrequencyRange(2.5e9, 2.4e9)
    
    def test_negative_frequencies(self):
        """Test negative frequencies raise error."""
        with pytest.raises(ValidationError):
            FrequencyRange(-1e9, 2.4e9)
    
    def test_zero_bandwidth(self):
        """Test zero bandwidth raises error."""
        with pytest.raises(ValidationError):
            FrequencyRange(2.4e9, 2.4e9)
    
    def test_frequency_range_validation(self):
        """Test comprehensive frequency range validation."""
        # Too low frequency
        with pytest.raises(ValidationError):
            FrequencyRange(100, 1000)
        
        # Too high frequency
        with pytest.raises(ValidationError):
            FrequencyRange(150e9, 200e9)


class TestMaterialProperties:
    """Test MaterialProperties functionality."""
    
    def test_substrate_material_properties(self):
        """Test substrate material property retrieval."""
        props = MaterialProperties.get_substrate_properties(SubstrateMaterial.ROGERS_4003C)
        
        assert props['dielectric_constant'] == 3.38
        assert props['loss_tangent'] == 0.0027
        assert props['thickness_mm'] == 1.52
        assert 'thermal_conductivity' in props
        assert 'density' in props
    
    def test_liquid_metal_properties(self):
        """Test liquid metal property retrieval."""
        props = MaterialProperties.get_liquid_metal_properties(LiquidMetalType.GALINSTAN)
        
        assert props['conductivity'] > 1e6  # High conductivity
        assert props['density'] > 6000  # High density
        assert 'melting_point' in props
        assert 'surface_tension' in props
    
    def test_invalid_material_enum(self):
        """Test invalid material enum handling."""
        with pytest.raises(ValidationError):
            MaterialProperties.get_substrate_properties("invalid_material")
    
    def test_custom_material_validation(self):
        """Test custom material property validation."""
        # Valid custom properties
        valid_props = {
            'dielectric_constant': 4.5,
            'loss_tangent': 0.02,
            'thickness_mm': 1.0
        }
        validated = MaterialProperties.validate_custom_substrate(valid_props)
        assert validated['dielectric_constant'] == 4.5
        
        # Invalid dielectric constant
        invalid_props = {
            'dielectric_constant': 0.5,  # Less than 1
            'loss_tangent': 0.02,
            'thickness_mm': 1.0
        }
        with pytest.raises(ValidationError):
            MaterialProperties.validate_custom_substrate(invalid_props)


class TestSizeConstraint:
    """Test SizeConstraint functionality."""
    
    def test_valid_size_constraint(self):
        """Test creating valid size constraint."""
        constraint = SizeConstraint(50e-3, 50e-3, 1.6e-3)
        assert constraint.max_width == 50e-3
        assert constraint.max_height == 50e-3
        assert constraint.max_thickness == 1.6e-3
    
    def test_negative_dimensions(self):
        """Test negative dimensions raise error."""
        with pytest.raises(ValidationError):
            SizeConstraint(-10e-3, 50e-3, 1.6e-3)
    
    def test_zero_dimensions(self):
        """Test zero dimensions raise error."""
        with pytest.raises(ValidationError):
            SizeConstraint(0, 50e-3, 1.6e-3)
    
    def test_constraint_validation(self):
        """Test size constraint validation against geometry."""
        constraint = SizeConstraint(50e-3, 50e-3, 1.6e-3)
        
        # Valid geometry
        valid_geometry = np.ones((25, 25, 4))  # Within constraints
        assert constraint.validate_geometry(valid_geometry, resolution=2e-3)
        
        # Invalid geometry - too large
        invalid_geometry = np.ones((100, 50, 4))  # Exceeds width constraint
        with pytest.raises(ValidationError):
            constraint.validate_geometry(invalid_geometry, resolution=1e-3)


class TestAntennaSpec:
    """Test complete AntennaSpec functionality."""
    
    def test_basic_antenna_spec_creation(self):
        """Test creating basic antenna specification."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        assert spec.frequency_range.start == 2.4e9
        assert spec.frequency_range.stop == 2.5e9
        assert spec.substrate_material == SubstrateMaterial.ROGERS_4003C
        assert spec.liquid_metal == LiquidMetalType.GALINSTAN
        assert spec.size_constraint is not None
        assert spec.performance_targets is not None
    
    def test_custom_performance_targets(self):
        """Test antenna spec with custom performance targets."""
        targets = {
            'min_gain_dbi': 8.0,
            'max_vswr': 1.5,
            'min_efficiency': 0.9,
            'bandwidth_requirement': 'wide'
        }
        
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            performance_targets=targets
        )
        
        assert spec.performance_targets['min_gain_dbi'] == 8.0
        assert spec.performance_targets['max_vswr'] == 1.5
    
    def test_antenna_spec_validation(self):
        """Test comprehensive antenna specification validation."""
        # Test with all valid parameters
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            size_constraint=SizeConstraint(100e-3, 100e-3, 1.6e-3),
            performance_targets={
                'min_gain_dbi': 5.0,
                'max_vswr': 2.0,
                'min_efficiency': 0.8
            }
        )
        
        # Should not raise any exceptions
        spec.validate()
    
    def test_incompatible_size_and_frequency(self):
        """Test detection of incompatible size and frequency constraints."""
        # Very high frequency with very large size constraint
        with pytest.raises(ValidationError):
            AntennaSpec(
                frequency_range=(60e9, 61e9),  # 60 GHz
                substrate=SubstrateMaterial.ROGERS_4003C,
                metal=LiquidMetalType.GALINSTAN,
                size_constraint=SizeConstraint(1.0, 1.0, 1.6e-3)  # 1m x 1m - too large
            )
    
    def test_material_compatibility_check(self):
        """Test material compatibility checking."""
        spec = AntennaSpec(
            frequency_range=(60e9, 61e9),  # mmWave frequency
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Should warn about high-frequency limitations
        with pytest.warns(UserWarning):
            spec.validate()
    
    def test_export_import_specification(self):
        """Test specification export and import."""
        original_spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            performance_targets={'min_gain_dbi': 6.0}
        )
        
        # Export to dictionary
        spec_dict = original_spec.to_dict()
        
        # Import from dictionary
        imported_spec = AntennaSpec.from_dict(spec_dict)
        
        # Verify equivalence
        assert imported_spec.frequency_range.start == original_spec.frequency_range.start
        assert imported_spec.substrate_material == original_spec.substrate_material
        assert imported_spec.liquid_metal == original_spec.liquid_metal
        assert imported_spec.performance_targets['min_gain_dbi'] == 6.0
    
    def test_specification_optimization_hints(self):
        """Test specification optimization hints generation."""
        spec = AntennaSpec(
            frequency_range=(2.4e9, 2.5e9),
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN,
            performance_targets={'min_gain_dbi': 15.0}  # Very high gain requirement
        )
        
        hints = spec.get_optimization_hints()
        
        assert 'gain_enhancement' in hints
        assert 'array_configuration' in hints
        assert any('array' in hint.lower() for hint in hints.values())
    
    def test_frequency_dependent_properties(self):
        """Test frequency-dependent material properties."""
        spec = AntennaSpec(
            frequency_range=(1e9, 10e9),  # Wide frequency range
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Test frequency-dependent calculations
        props_1ghz = spec.get_effective_properties(1e9)
        props_10ghz = spec.get_effective_properties(10e9)
        
        # Properties should be different at different frequencies
        assert props_1ghz['wavelength'] > props_10ghz['wavelength']
        assert 'skin_depth' in props_1ghz
        assert 'skin_depth' in props_10ghz


class TestSpecificationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extreme_frequency_ranges(self):
        """Test extremely wide or narrow frequency ranges."""
        # Extremely narrow bandwidth
        with pytest.raises(ValidationError):
            AntennaSpec(
                frequency_range=(2.4e9, 2.4e9 + 1),  # 1 Hz bandwidth
                substrate=SubstrateMaterial.ROGERS_4003C,
                metal=LiquidMetalType.GALINSTAN
            )
        
        # Very wide bandwidth
        spec_wide = AntennaSpec(
            frequency_range=(1e9, 20e9),  # 19 GHz bandwidth
            substrate=SubstrateMaterial.ROGERS_4003C,
            metal=LiquidMetalType.GALINSTAN
        )
        
        # Should warn about ultra-wideband requirements
        with pytest.warns(UserWarning):
            spec_wide.validate()
    
    def test_custom_material_edge_cases(self):
        """Test edge cases with custom materials."""
        # Very high loss tangent
        high_loss_props = {
            'dielectric_constant': 10.0,
            'loss_tangent': 0.5,  # Very lossy
            'thickness_mm': 1.6
        }
        
        with pytest.warns(UserWarning):
            MaterialProperties.validate_custom_substrate(high_loss_props)
        
        # Very high dielectric constant
        high_er_props = {
            'dielectric_constant': 50.0,  # Very high
            'loss_tangent': 0.02,
            'thickness_mm': 1.6
        }
        
        with pytest.warns(UserWarning):
            MaterialProperties.validate_custom_substrate(high_er_props)
    
    def test_specification_serialization_edge_cases(self):
        """Test specification serialization with edge cases."""
        # Specification with all custom properties
        spec = AntennaSpec(
            frequency_range=(5.8e9, 6.2e9),
            substrate=SubstrateMaterial.CUSTOM,
            metal=LiquidMetalType.CUSTOM,
            custom_substrate_properties={
                'dielectric_constant': 2.2,
                'loss_tangent': 0.001,
                'thickness_mm': 0.8
            },
            custom_metal_properties={
                'conductivity': 2e7,
                'density': 8900,
                'viscosity': 1e-3
            }
        )
        
        # Should serialize and deserialize correctly
        spec_dict = spec.to_dict()
        restored_spec = AntennaSpec.from_dict(spec_dict)
        
        assert restored_spec.custom_substrate_properties['dielectric_constant'] == 2.2
        assert restored_spec.custom_metal_properties['conductivity'] == 2e7


if __name__ == '__main__':
    pytest.main([__file__])