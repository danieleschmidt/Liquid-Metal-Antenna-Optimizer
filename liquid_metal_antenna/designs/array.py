"""
Liquid metal antenna array designs.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from ..core.antenna_spec import AntennaSpec
from .patch import ReconfigurablePatch


class LiquidMetalArray:
    """
    Phased array using liquid metal antennas.
    
    This class implements an array of liquid metal antennas with
    reconfigurable elements and beam steering capabilities.
    """
    
    def __init__(
        self,
        n_elements: Tuple[int, int] = (4, 4),
        element_spacing: float = 0.5,  # wavelengths
        element_type: str = 'patch',
        feed_network: str = 'corporate',
        phase_shifter_type: str = 'liquid_metal_delay_line'
    ):
        """
        Initialize liquid metal array.
        
        Args:
            n_elements: Array dimensions (nx, ny)
            element_spacing: Element spacing in wavelengths
            element_type: Type of array elements
            feed_network: Feed network topology
            phase_shifter_type: Phase shifter implementation
        """
        self.n_elements = n_elements
        self.element_spacing = element_spacing
        self.element_type = element_type
        self.feed_network = feed_network
        self.phase_shifter_type = phase_shifter_type
        
        # Create array elements
        self._create_elements()
        
        # Phase control
        self.element_phases = np.zeros(n_elements)
        
        # Beam steering parameters
        self.current_beam_direction = (0.0, 0.0)  # (theta, phi) in radians
    
    def _create_elements(self) -> None:
        """Create individual array elements."""
        nx, ny = self.n_elements
        self.elements = []
        
        for i in range(nx):
            row = []
            for j in range(ny):
                if self.element_type == 'patch':
                    element = ReconfigurablePatch(n_channels=4)  # Simplified elements
                else:
                    # Placeholder for other element types
                    element = ReconfigurablePatch(n_channels=4)
                
                row.append(element)
            self.elements.append(row)
    
    def set_beam_direction(self, theta: float, phi: float = 0.0, frequency: float = 2.45e9) -> None:
        """
        Set beam steering direction.
        
        Args:
            theta: Elevation angle in degrees
            phi: Azimuth angle in degrees  
            frequency: Operating frequency in Hz
        """
        # Convert to radians
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        
        self.current_beam_direction = (theta_rad, phi_rad)
        
        # Calculate required phase shifts
        c = 299792458
        wavelength = c / frequency
        k = 2 * np.pi / wavelength
        
        # Element spacing in meters
        dx = self.element_spacing * wavelength
        dy = self.element_spacing * wavelength
        
        nx, ny = self.n_elements
        
        # Calculate phase for each element
        for i in range(nx):
            for j in range(ny):
                # Element position relative to array center
                x_pos = (i - (nx - 1) / 2) * dx
                y_pos = (j - (ny - 1) / 2) * dy
                
                # Required phase shift for beam steering
                phase_shift = k * (x_pos * np.sin(theta_rad) * np.cos(phi_rad) +
                                  y_pos * np.sin(theta_rad) * np.sin(phi_rad))
                
                self.element_phases[i, j] = phase_shift
    
    def create_geometry_tensor(
        self,
        grid_resolution: float = 0.5e-3,
        frequency: float = 2.45e9,
        total_size: Optional[Tuple[float, float, float]] = None
    ) -> torch.Tensor:
        """
        Create geometry tensor for the entire array.
        
        Args:
            grid_resolution: Grid resolution in meters
            frequency: Design frequency in Hz
            total_size: Override total simulation size
            
        Returns:
            Geometry tensor for the array
        """
        # Calculate array dimensions
        c = 299792458
        wavelength = c / frequency
        spacing_m = self.element_spacing * wavelength
        
        nx, ny = self.n_elements
        array_width = (nx - 1) * spacing_m + 0.05  # 5cm extra margin
        array_length = (ny - 1) * spacing_m + 0.05
        array_height = 0.01  # 1cm height
        
        if total_size is None:
            total_size = (array_width * 1000, array_length * 1000, array_height * 1000)  # Convert to mm
        
        # Grid dimensions
        grid_nx = int(total_size[0] * 1e-3 / grid_resolution)
        grid_ny = int(total_size[1] * 1e-3 / grid_resolution)
        grid_nz = int(total_size[2] * 1e-3 / grid_resolution)
        
        # Initialize geometry
        geometry = torch.zeros((grid_nx, grid_ny, grid_nz), dtype=torch.float32)
        
        # Add ground plane
        ground_z = grid_nz // 4
        geometry[:, :, ground_z] = 1.0
        
        # Element dimensions in grid cells
        element_size_cells = int(0.03 / grid_resolution)  # 3cm elements
        spacing_cells_x = int(spacing_m / grid_resolution)
        spacing_cells_y = int(spacing_m / grid_resolution)
        
        # Array center in grid coordinates
        center_x = grid_nx // 2
        center_y = grid_ny // 2
        element_z = grid_nz - grid_nz // 4  # Top layer
        
        # Place elements
        for i in range(nx):
            for j in range(ny):
                # Element position
                elem_x = center_x + (i - (nx - 1) / 2) * spacing_cells_x
                elem_y = center_y + (j - (ny - 1) / 2) * spacing_cells_y
                
                # Element boundaries
                x_start = int(elem_x - element_size_cells // 2)
                x_end = int(elem_x + element_size_cells // 2)
                y_start = int(elem_y - element_size_cells // 2)
                y_end = int(elem_y + element_size_cells // 2)
                
                # Ensure within bounds
                x_start = max(0, min(x_start, grid_nx - 1))
                x_end = max(0, min(x_end, grid_nx))
                y_start = max(0, min(y_start, grid_ny - 1))
                y_end = max(0, min(y_end, grid_ny))
                
                # Add element
                geometry[x_start:x_end, y_start:y_end, element_z] = 1.0
        
        return geometry
    
    def compute_array_factor(
        self,
        theta_range: np.ndarray,
        phi_range: np.ndarray,
        frequency: float = 2.45e9
    ) -> np.ndarray:
        """
        Compute array factor for radiation pattern.
        
        Args:
            theta_range: Theta angles in radians
            phi_range: Phi angles in radians  
            frequency: Frequency in Hz
            
        Returns:
            Array factor [theta, phi]
        """
        c = 299792458
        wavelength = c / frequency
        k = 2 * np.pi / wavelength
        
        # Element spacing in meters
        dx = self.element_spacing * wavelength
        dy = self.element_spacing * wavelength
        
        nx, ny = self.n_elements
        
        # Create angle meshgrids
        theta_mesh, phi_mesh = np.meshgrid(theta_range, phi_range, indexing='ij')
        
        # Initialize array factor
        array_factor = np.zeros_like(theta_mesh, dtype=complex)
        
        # Sum contributions from all elements
        for i in range(nx):
            for j in range(ny):
                # Element position relative to array center
                x_pos = (i - (nx - 1) / 2) * dx
                y_pos = (j - (ny - 1) / 2) * dy
                
                # Phase contribution
                phase = k * (x_pos * np.sin(theta_mesh) * np.cos(phi_mesh) +
                           y_pos * np.sin(theta_mesh) * np.sin(phi_mesh))
                
                # Add element phase shift
                element_phase = self.element_phases[i, j]
                
                # Element contribution
                array_factor += np.exp(1j * (phase - element_phase))
        
        return np.abs(array_factor) ** 2
    
    def optimize_scan_pattern(
        self,
        scan_angles: np.ndarray,
        frequency: float = 2.45e9,
        side_lobe_level: float = -20.0,
        maintain_gain: bool = True
    ) -> Dict[float, Dict[str, Any]]:
        """Optimize scanning pattern (Generation 2 feature)."""
        print("Scan pattern optimization not implemented in Generation 1")
        return {}
    
    def animate_beam_scan(
        self,
        beam_states: Dict[float, Dict[str, Any]],
        output: str = 'beam_scan.gif',
        fps: int = 10
    ) -> None:
        """Animate beam scanning (Generation 2 feature)."""
        print("Beam scan animation not implemented in Generation 1")
    
    def calculate_mutual_coupling(self) -> np.ndarray:
        """Calculate mutual coupling between elements."""
        nx, ny = self.n_elements
        total_elements = nx * ny
        
        # Placeholder coupling matrix
        coupling_matrix = np.eye(total_elements, dtype=complex)
        
        # Add some coupling between adjacent elements
        for i in range(total_elements):
            for j in range(total_elements):
                if i != j:
                    # Distance-based coupling (simplified)
                    distance = abs(i - j)
                    coupling_matrix[i, j] = 0.1 / (1 + distance) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        
        return coupling_matrix
    
    def get_gain_pattern(
        self,
        frequency: float = 2.45e9,
        resolution: int = 180
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get antenna gain pattern.
        
        Args:
            frequency: Frequency in Hz
            resolution: Angular resolution in degrees
            
        Returns:
            gain_pattern: Gain in dB [theta, phi]
            theta: Theta angles in radians
            phi: Phi angles in radians
        """
        # Angular ranges
        theta = np.linspace(0, np.pi, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        
        # Compute array factor
        array_factor = self.compute_array_factor(theta, phi, frequency)
        
        # Convert to dB
        gain_pattern = 10 * np.log10(np.maximum(array_factor, 1e-10))
        gain_pattern = gain_pattern - np.max(gain_pattern)  # Normalize
        
        return gain_pattern, theta, phi
    
    def export_beam_config(self, filename: str) -> None:
        """Export current beam configuration."""
        import json
        
        theta_deg, phi_deg = np.degrees(self.current_beam_direction)
        
        config = {
            'antenna_type': 'LiquidMetalArray',
            'n_elements': self.n_elements,
            'element_spacing': self.element_spacing,
            'element_type': self.element_type,
            'feed_network': self.feed_network,
            'phase_shifter_type': self.phase_shifter_type,
            'beam_direction': {
                'theta_deg': theta_deg,
                'phi_deg': phi_deg
            },
            'element_phases': self.element_phases.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Beam configuration exported to {filename}")
    
    def __repr__(self) -> str:
        nx, ny = self.n_elements
        theta_deg, phi_deg = np.degrees(self.current_beam_direction)
        
        return (
            f"LiquidMetalArray("
            f"{nx}x{ny} elements, "
            f"spacing={self.element_spacing:.1f}λ, "
            f"beam=({theta_deg:.1f}°,{phi_deg:.1f}°)"
            f")"
        )