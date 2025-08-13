"""
Antenna specification and configuration classes.
"""

from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
# Import numpy with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple numpy substitute for basic operations
    class SimpleNumPy:
        @staticmethod
        def sqrt(x):
            return x ** 0.5
    np = SimpleNumPy()


class SubstrateMaterial(Enum):
    """Standard substrate materials with their properties."""
    ROGERS_4003C = "rogers_4003c"
    FR4 = "fr4"
    DUROID_5880 = "duroid_5880"
    ALUMINA = "alumina"


class LiquidMetalType(Enum):
    """Supported liquid metal types."""
    GALINSTAN = "galinstan"
    MERCURY = "mercury"
    INDIUM = "indium"


@dataclass
class MaterialProperties:
    """Material properties for antenna design."""
    dielectric_constant: float
    loss_tangent: float
    thickness: float  # mm
    
    @classmethod
    def get_substrate_properties(cls, material: SubstrateMaterial) -> 'MaterialProperties':
        """Get predefined substrate properties."""
        properties = {
            SubstrateMaterial.ROGERS_4003C: cls(3.38, 0.0027, 1.6),
            SubstrateMaterial.FR4: cls(4.4, 0.02, 1.6),
            SubstrateMaterial.DUROID_5880: cls(2.2, 0.0009, 1.6),
            SubstrateMaterial.ALUMINA: cls(9.8, 0.0001, 0.635),
        }
        return properties[material]


@dataclass
class FrequencyRange:
    """Frequency range specification."""
    start: float  # Hz
    stop: float   # Hz
    
    def __post_init__(self):
        if self.start >= self.stop:
            raise ValueError("Start frequency must be less than stop frequency")
    
    @property
    def center(self) -> float:
        """Center frequency."""
        return (self.start + self.stop) / 2
    
    @property
    def bandwidth(self) -> float:
        """Bandwidth in Hz."""
        return self.stop - self.start
    
    @property
    def fractional_bandwidth(self) -> float:
        """Fractional bandwidth."""
        return self.bandwidth / self.center


@dataclass
class SizeConstraint:
    """Physical size constraints for the antenna."""
    length: float  # mm
    width: float   # mm
    height: float  # mm
    
    @property
    def volume(self) -> float:
        """Volume in cubic mm."""
        return self.length * self.width * self.height
    
    @property
    def area(self) -> float:
        """Area in square mm."""
        return self.length * self.width


class AntennaSpec:
    """
    Complete antenna specification including electrical and physical constraints.
    
    This class defines all the requirements and constraints for antenna design,
    including frequency range, substrate properties, liquid metal type, and
    physical size limitations.
    """
    
    def __init__(
        self,
        frequency_range: Union[Tuple[float, float], FrequencyRange],
        substrate: Union[str, SubstrateMaterial, MaterialProperties] = SubstrateMaterial.ROGERS_4003C,
        metal: Union[str, LiquidMetalType] = LiquidMetalType.GALINSTAN,
        size_constraint: Union[Tuple[float, float, float], SizeConstraint] = (50, 50, 3),
        polarization: str = "linear",
        **kwargs
    ):
        """
        Initialize antenna specification.
        
        Args:
            frequency_range: Operating frequency range in Hz, either as tuple or FrequencyRange
            substrate: Substrate material specification
            metal: Liquid metal type
            size_constraint: Maximum dimensions (length, width, height) in mm
            polarization: Antenna polarization type
            **kwargs: Additional specification parameters
        """
        # Handle frequency range
        if isinstance(frequency_range, tuple):
            self.frequency_range = FrequencyRange(frequency_range[0], frequency_range[1])
        else:
            self.frequency_range = frequency_range
        
        # Handle substrate material
        if isinstance(substrate, str):
            substrate = SubstrateMaterial(substrate)
        if isinstance(substrate, SubstrateMaterial):
            self.substrate = MaterialProperties.get_substrate_properties(substrate)
        else:
            self.substrate = substrate
        
        # Handle metal type
        if isinstance(metal, str):
            metal = LiquidMetalType(metal)
        self.metal = metal
        
        # Handle size constraint
        if isinstance(size_constraint, tuple):
            self.size_constraint = SizeConstraint(*size_constraint)
        else:
            self.size_constraint = size_constraint
        
        self.polarization = polarization
        
        # Additional parameters
        self.min_gain = kwargs.get('min_gain', 0.0)  # dBi
        self.max_vswr = kwargs.get('max_vswr', 2.0)
        self.min_efficiency = kwargs.get('min_efficiency', 0.8)
        self.min_bandwidth = kwargs.get('min_bandwidth', None)  # Hz
        
        # Liquid metal specific parameters
        self.max_channels = kwargs.get('max_channels', 8)
        self.min_channel_width = kwargs.get('min_channel_width', 0.5)  # mm
        self.actuation_time = kwargs.get('actuation_time', 1.0)  # seconds
        
        self._validate_spec()
    
    def _validate_spec(self) -> None:
        """Validate the antenna specification for consistency."""
        # Check frequency range is reasonable
        if self.frequency_range.start < 1e6:  # 1 MHz
            raise ValueError("Frequency range too low for antenna design")
        if self.frequency_range.stop > 100e9:  # 100 GHz
            raise ValueError("Frequency range too high for current implementation")
        
        # Check size constraints are positive
        if any(dim <= 0 for dim in [self.size_constraint.length, self.size_constraint.width, self.size_constraint.height]):
            raise ValueError("All size constraints must be positive")
        
        # Check VSWR constraint
        if self.max_vswr < 1.0:
            raise ValueError("VSWR must be >= 1.0")
        
        # Check efficiency constraint
        if not 0.0 <= self.min_efficiency <= 1.0:
            raise ValueError("Efficiency must be between 0.0 and 1.0")
    
    def get_wavelength_at_center(self) -> float:
        """Get free-space wavelength at center frequency in mm."""
        c = 299792458e3  # speed of light in mm/s
        return c / self.frequency_range.center
    
    def get_substrate_wavelength_at_center(self) -> float:
        """Get substrate wavelength at center frequency in mm."""
        return self.get_wavelength_at_center() / np.sqrt(self.substrate.dielectric_constant)
    
    def is_electrically_small(self) -> bool:
        """Check if antenna is electrically small (< λ/10)."""
        wavelength = self.get_wavelength_at_center()
        max_dimension = max(self.size_constraint.length, self.size_constraint.width)
        return max_dimension < wavelength / 10
    
    def get_liquid_metal_conductivity(self, temperature: float = 25.0) -> float:
        """
        Get liquid metal conductivity at given temperature.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Conductivity in S/m
        """
        # Simplified temperature-dependent conductivity model
        if self.metal == LiquidMetalType.GALINSTAN:
            # Galinstan conductivity at room temperature ~3.46e6 S/m
            base_conductivity = 3.46e6
            temp_coefficient = -0.001  # per degree C
            return base_conductivity * (1 + temp_coefficient * (temperature - 25))
        elif self.metal == LiquidMetalType.MERCURY:
            return 1.04e6
        elif self.metal == LiquidMetalType.INDIUM:
            return 1.16e7
        else:
            raise ValueError(f"Unknown liquid metal type: {self.metal}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary for serialization."""
        return {
            'frequency_range': {
                'start': self.frequency_range.start,
                'stop': self.frequency_range.stop
            },
            'substrate': {
                'dielectric_constant': self.substrate.dielectric_constant,
                'loss_tangent': self.substrate.loss_tangent,
                'thickness': self.substrate.thickness
            },
            'metal': self.metal.value,
            'size_constraint': {
                'length': self.size_constraint.length,
                'width': self.size_constraint.width,
                'height': self.size_constraint.height
            },
            'polarization': self.polarization,
            'min_gain': self.min_gain,
            'max_vswr': self.max_vswr,
            'min_efficiency': self.min_efficiency,
            'min_bandwidth': self.min_bandwidth,
            'max_channels': self.max_channels,
            'min_channel_width': self.min_channel_width,
            'actuation_time': self.actuation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AntennaSpec':
        """Create AntennaSpec from dictionary."""
        freq_range = FrequencyRange(
            data['frequency_range']['start'],
            data['frequency_range']['stop']
        )
        
        substrate = MaterialProperties(
            data['substrate']['dielectric_constant'],
            data['substrate']['loss_tangent'],
            data['substrate']['thickness']
        )
        
        size_constraint = SizeConstraint(
            data['size_constraint']['length'],
            data['size_constraint']['width'],
            data['size_constraint']['height']
        )
        
        return cls(
            frequency_range=freq_range,
            substrate=substrate,
            metal=LiquidMetalType(data['metal']),
            size_constraint=size_constraint,
            polarization=data['polarization'],
            min_gain=data.get('min_gain', 0.0),
            max_vswr=data.get('max_vswr', 2.0),
            min_efficiency=data.get('min_efficiency', 0.8),
            min_bandwidth=data.get('min_bandwidth'),
            max_channels=data.get('max_channels', 8),
            min_channel_width=data.get('min_channel_width', 0.5),
            actuation_time=data.get('actuation_time', 1.0)
        )
    
    def __repr__(self) -> str:
        return (
            f"AntennaSpec("
            f"freq={self.frequency_range.start/1e9:.2f}-{self.frequency_range.stop/1e9:.2f}GHz, "
            f"substrate=εᵣ={self.substrate.dielectric_constant:.1f}, "
            f"metal={self.metal.value}, "
            f"size={self.size_constraint.length}x{self.size_constraint.width}x{self.size_constraint.height}mm"
            f")"
        )