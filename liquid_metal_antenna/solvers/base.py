"""
Base solver interface and result classes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch


@dataclass
class SolverResult:
    """Result container for electromagnetic simulation."""
    
    # S-parameters
    s_parameters: np.ndarray  # Complex S-parameters [freq, port, port]
    frequencies: np.ndarray   # Frequency points in Hz
    
    # Field data (optional)
    electric_field: Optional[np.ndarray] = None  # E-field [x, y, z, component]
    magnetic_field: Optional[np.ndarray] = None  # H-field [x, y, z, component]
    
    # Radiation patterns (optional)
    radiation_pattern: Optional[np.ndarray] = None  # [theta, phi]
    theta_angles: Optional[np.ndarray] = None
    phi_angles: Optional[np.ndarray] = None
    
    # Computed metrics
    gain_dbi: Optional[float] = None
    max_gain_dbi: Optional[float] = None
    directivity_dbi: Optional[float] = None
    efficiency: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    vswr: Optional[np.ndarray] = None  # VSWR vs frequency
    
    # Convergence information
    converged: bool = False
    iterations: int = 0
    convergence_error: float = 1.0
    computation_time: float = 0.0
    
    def get_gain_at_frequency(self, frequency: float) -> float:
        """Get gain at specific frequency."""
        if self.gain_dbi is None:
            return 0.0
        if isinstance(self.gain_dbi, (int, float)):
            return float(self.gain_dbi)
        # If frequency-dependent gain is stored
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        return float(self.gain_dbi[freq_idx]) if hasattr(self.gain_dbi, '__getitem__') else float(self.gain_dbi)
    
    def get_vswr_at_frequency(self, frequency: float) -> float:
        """Get VSWR at specific frequency."""
        if self.vswr is None:
            return float('inf')
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        return float(self.vswr[freq_idx])
    
    def compute_bandwidth(self, vswr_threshold: float = 2.0) -> float:
        """Compute bandwidth where VSWR < threshold."""
        if self.vswr is None:
            return 0.0
        
        valid_freqs = self.frequencies[self.vswr < vswr_threshold]
        if len(valid_freqs) == 0:
            return 0.0
        
        return float(np.max(valid_freqs) - np.min(valid_freqs))


class BaseSolver(ABC):
    """
    Abstract base class for electromagnetic solvers.
    
    All solvers must implement the basic simulation interface
    and provide consistent result formats.
    """
    
    def __init__(
        self,
        resolution: float = 1.0e-3,  # Default 1mm resolution
        device: str = 'cpu',
        precision: str = 'float32'
    ):
        """
        Initialize solver with basic parameters.
        
        Args:
            resolution: Grid resolution in meters
            device: Computation device ('cpu', 'cuda', 'cuda:0', etc.)
            precision: Computation precision ('float32' or 'float64')
        """
        self.resolution = resolution
        self.device = device
        self.precision = precision
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Setup computation device."""
        if 'cuda' in self.device:
            if not torch.cuda.is_available():
                print(f"Warning: CUDA not available, falling back to CPU")
                self.device = 'cpu'
            else:
                torch.cuda.set_device(self.device)
    
    @abstractmethod
    def simulate(
        self,
        geometry: Union[np.ndarray, torch.Tensor],
        frequency: Union[float, np.ndarray],
        **kwargs
    ) -> SolverResult:
        """
        Run electromagnetic simulation.
        
        Args:
            geometry: Antenna geometry description
            frequency: Simulation frequency(ies) in Hz
            **kwargs: Additional solver-specific parameters
            
        Returns:
            SolverResult containing simulation results
        """
        pass
    
    @abstractmethod
    def compute_s_parameters(
        self,
        fields: Union[np.ndarray, torch.Tensor],
        frequency: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Compute S-parameters from field data."""
        pass
    
    @abstractmethod
    def compute_radiation_pattern(
        self,
        fields: Union[np.ndarray, torch.Tensor],
        frequency: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute radiation pattern from field data.
        
        Returns:
            pattern: Radiation pattern [theta, phi]
            theta: Theta angles in radians
            phi: Phi angles in radians
        """
        pass
    
    def compute_gain(self, pattern: np.ndarray) -> float:
        """Compute maximum gain from radiation pattern."""
        if pattern is None or pattern.size == 0:
            return 0.0
        
        # Convert power pattern to gain in dBi
        max_power = np.max(np.real(pattern))
        if max_power <= 0:
            return 0.0
        
        # Assuming isotropic reference, gain = 4π * max_power / total_power
        total_power = np.mean(np.real(pattern))
        if total_power <= 0:
            return 0.0
        
        gain_linear = max_power / total_power
        gain_dbi = 10 * np.log10(gain_linear) if gain_linear > 0 else -50.0
        
        return float(gain_dbi)
    
    def compute_efficiency(
        self,
        radiated_power: float,
        input_power: float
    ) -> float:
        """Compute radiation efficiency."""
        if input_power <= 0:
            return 0.0
        return min(radiated_power / input_power, 1.0)
    
    def compute_vswr(self, s11: np.ndarray) -> np.ndarray:
        """Compute VSWR from S11."""
        s11_mag = np.abs(s11)
        s11_mag = np.clip(s11_mag, 0, 0.999)  # Avoid division by zero
        vswr = (1 + s11_mag) / (1 - s11_mag)
        return vswr
    
    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Get current grid dimensions."""
        return getattr(self, '_grid_size', (64, 64, 32))
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in GB."""
        nx, ny, nz = self.grid_size
        total_cells = nx * ny * nz
        
        # Estimate based on FDTD field storage (6 field components)
        # plus auxiliary arrays for PML, sources, etc.
        fields_memory = total_cells * 6 * 4  # 4 bytes per float32
        aux_memory = total_cells * 2 * 4     # Additional arrays
        
        return (fields_memory + aux_memory) / 1e9  # Convert to GB
    
    def set_convergence_criteria(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-4
    ) -> None:
        """Set convergence criteria for iterative solvers."""
        self.max_iterations = max_iterations
        self.tolerance = tolerance