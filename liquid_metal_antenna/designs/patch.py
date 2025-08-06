"""
Reconfigurable patch antenna designs using liquid metal.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from ..core.antenna_spec import AntennaSpec


class ReconfigurablePatch:
    """
    Reconfigurable microstrip patch antenna using liquid metal channels.
    
    This class implements a patch antenna with embedded microfluidic channels
    that can be filled with liquid metal to create reconfigurable radiation
    patterns and frequency responses.
    """
    
    def __init__(
        self,
        substrate_height: float = 1.6,  # mm
        dielectric_constant: float = 4.4,
        n_channels: int = 8,
        channel_width: float = 0.5,  # mm
        patch_dimensions: Optional[Tuple[float, float]] = None  # (length, width) in mm
    ):
        """
        Initialize reconfigurable patch antenna.
        
        Args:
            substrate_height: Substrate thickness in mm
            dielectric_constant: Relative permittivity of substrate
            n_channels: Number of liquid metal channels
            channel_width: Width of each channel in mm
            patch_dimensions: Initial patch dimensions (auto-calculated if None)
        """
        self.substrate_height = substrate_height
        self.dielectric_constant = dielectric_constant
        self.n_channels = n_channels
        self.channel_width = channel_width
        self.patch_dimensions = patch_dimensions
        
        # Channel configuration
        self.channel_states = np.zeros(n_channels, dtype=bool)  # False = empty, True = filled
        
        # Default patch dimensions based on substrate properties
        if patch_dimensions is None:
            # Approximate patch dimensions for 2.45 GHz
            freq_design = 2.45e9
            c = 299792458
            lambda_0 = c / freq_design
            lambda_g = lambda_0 / np.sqrt(dielectric_constant)
            
            # Standard patch dimensions
            self.patch_length = lambda_g / 2 * 1000  # Convert to mm
            self.patch_width = lambda_g / 3 * 1000   # Convert to mm
        else:
            self.patch_length, self.patch_width = patch_dimensions
        
        self._create_channel_layout()
    
    def _create_channel_layout(self) -> None:
        """Create the layout of liquid metal channels."""
        # Simple channel layout - horizontal channels across the patch
        self.channels = []
        
        channel_spacing = self.patch_length / (self.n_channels + 1)
        
        for i in range(self.n_channels):
            y_pos = (i + 1) * channel_spacing - self.patch_length / 2
            
            # Channel definition: (start_x, start_y, end_x, end_y)
            channel = {
                'id': i,
                'start': (-self.patch_width / 2, y_pos),
                'end': (self.patch_width / 2, y_pos),
                'width': self.channel_width,
                'filled': self.channel_states[i]
            }
            
            self.channels.append(channel)
    
    def set_channel_state(self, channel_id: int, filled: bool) -> None:
        """Set the fill state of a specific channel."""
        if 0 <= channel_id < self.n_channels:
            self.channel_states[channel_id] = filled
            self.channels[channel_id]['filled'] = filled
        else:
            raise ValueError(f"Channel ID {channel_id} out of range (0-{self.n_channels-1})")
    
    def set_configuration(self, channel_fill: List[bool]) -> None:
        """Set the configuration of all channels."""
        if len(channel_fill) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channel states, got {len(channel_fill)}")
        
        self.channel_states = np.array(channel_fill, dtype=bool)
        for i, filled in enumerate(channel_fill):
            self.channels[i]['filled'] = filled
    
    def get_configuration(self) -> List[bool]:
        """Get current channel configuration."""
        return self.channel_states.tolist()
    
    def create_geometry_tensor(
        self,
        grid_resolution: float = 0.5e-3,  # 0.5mm
        total_size: Tuple[float, float, float] = (50, 50, 3)  # mm
    ) -> torch.Tensor:
        """
        Create geometry tensor for electromagnetic simulation.
        
        Args:
            grid_resolution: Grid resolution in meters
            total_size: Total simulation domain size (x, y, z) in mm
            
        Returns:
            Geometry tensor with 1 for conductor, 0 for air
        """
        # Convert dimensions to grid cells
        nx = int(total_size[0] * 1e-3 / grid_resolution)
        ny = int(total_size[1] * 1e-3 / grid_resolution)
        nz = int(total_size[2] * 1e-3 / grid_resolution)
        
        # Initialize geometry (air)
        geometry = torch.zeros((nx, ny, nz), dtype=torch.float32)
        
        # Add ground plane (bottom layer)
        ground_z = nz // 4
        geometry[:, :, ground_z] = 1.0
        
        # Add patch antenna (top layer)
        patch_z = nz - nz // 4
        
        # Patch boundaries in grid coordinates
        patch_x_cells = int(self.patch_length * 1e-3 / grid_resolution)
        patch_y_cells = int(self.patch_width * 1e-3 / grid_resolution)
        
        center_x, center_y = nx // 2, ny // 2
        
        x_start = center_x - patch_x_cells // 2
        x_end = center_x + patch_x_cells // 2
        y_start = center_y - patch_y_cells // 2
        y_end = center_y + patch_y_cells // 2
        
        # Create basic patch
        geometry[x_start:x_end, y_start:y_end, patch_z] = 1.0
        
        # Add liquid metal channels
        channel_width_cells = max(1, int(self.channel_width * 1e-3 / grid_resolution))
        
        for channel in self.channels:
            if channel['filled']:
                # Channel position in grid coordinates
                channel_y = center_y + int(channel['start'][1] * 1e-3 / grid_resolution)
                
                # Add horizontal channel
                for dy in range(-channel_width_cells//2, channel_width_cells//2 + 1):
                    y_pos = channel_y + dy
                    if 0 <= y_pos < ny:
                        geometry[x_start:x_end, y_pos, patch_z] = 1.0
        
        # Add feed line (simple microstrip feed)
        feed_width_cells = max(1, int(2e-3 / grid_resolution))  # 2mm feed width
        feed_length_cells = patch_x_cells // 3
        
        feed_x_start = center_x - feed_width_cells // 2
        feed_x_end = center_x + feed_width_cells // 2
        feed_y_start = y_start - feed_length_cells
        
        if feed_y_start >= 0:
            geometry[feed_x_start:feed_x_end, feed_y_start:y_start, patch_z] = 1.0
        
        return geometry
    
    def get_resonant_frequency(self, mode: str = 'TM10') -> float:
        """
        Calculate theoretical resonant frequency.
        
        Args:
            mode: Resonant mode ('TM10', 'TM01', 'TM11')
            
        Returns:
            Resonant frequency in Hz
        """
        c = 299792458  # Speed of light
        
        # Effective dielectric constant (approximation)
        eps_eff = (self.dielectric_constant + 1) / 2 + \
                  (self.dielectric_constant - 1) / 2 * \
                  (1 + 12 * self.substrate_height / (self.patch_width * 1e-3)) ** (-0.5)
        
        # Resonant frequencies for different modes
        if mode == 'TM10':
            # Length resonance
            f_res = c / (2 * self.patch_length * 1e-3 * np.sqrt(eps_eff))
        elif mode == 'TM01':
            # Width resonance
            f_res = c / (2 * self.patch_width * 1e-3 * np.sqrt(eps_eff))
        elif mode == 'TM11':
            # Combined mode
            f_res = c / (2 * np.sqrt((self.patch_length * 1e-3) ** 2 + (self.patch_width * 1e-3) ** 2) * np.sqrt(eps_eff))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return f_res
    
    def estimate_bandwidth(self) -> float:
        """Estimate antenna bandwidth."""
        # Simple bandwidth estimation based on patch dimensions
        f_res = self.get_resonant_frequency()
        
        # Fractional bandwidth approximation
        fractional_bw = self.substrate_height / (self.patch_length * np.sqrt(self.dielectric_constant))
        
        return f_res * fractional_bw
    
    def get_reconfiguration_states(self) -> Dict[str, Any]:
        """Get predefined reconfiguration states for different frequencies."""
        states = {}
        
        # State 1: Base configuration (all channels empty)
        states['base'] = {
            'channel_fill': [False] * self.n_channels,
            'description': 'Base patch antenna',
            'target_frequency': self.get_resonant_frequency()
        }
        
        # State 2: Extended patch (fill alternating channels)
        alt_fill = [i % 2 == 0 for i in range(self.n_channels)]
        states['extended'] = {
            'channel_fill': alt_fill,
            'description': 'Extended effective length',
            'target_frequency': self.get_resonant_frequency() * 0.9  # Lower frequency
        }
        
        # State 3: Maximum fill (all channels filled)
        states['maximum'] = {
            'channel_fill': [True] * self.n_channels,
            'description': 'Maximum conductor coverage',
            'target_frequency': self.get_resonant_frequency() * 0.8  # Even lower frequency
        }
        
        return states
    
    def export_config(self, filename: str) -> None:
        """Export current configuration to file."""
        import json
        
        config = {
            'antenna_type': 'ReconfigurablePatch',
            'substrate_height': self.substrate_height,
            'dielectric_constant': self.dielectric_constant,
            'patch_dimensions': [self.patch_length, self.patch_width],
            'n_channels': self.n_channels,
            'channel_width': self.channel_width,
            'channel_states': self.channel_states.tolist(),
            'resonant_frequency': self.get_resonant_frequency(),
            'estimated_bandwidth': self.estimate_bandwidth()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration exported to {filename}")
    
    @classmethod
    def load_config(cls, filename: str) -> 'ReconfigurablePatch':
        """Load configuration from file."""
        import json
        
        with open(filename, 'r') as f:
            config = json.load(f)
        
        antenna = cls(
            substrate_height=config['substrate_height'],
            dielectric_constant=config['dielectric_constant'],
            n_channels=config['n_channels'],
            channel_width=config['channel_width'],
            patch_dimensions=tuple(config['patch_dimensions'])
        )
        
        antenna.set_configuration(config['channel_states'])
        
        return antenna
    
    def __repr__(self) -> str:
        filled_channels = np.sum(self.channel_states)
        return (
            f"ReconfigurablePatch("
            f"size={self.patch_length:.1f}x{self.patch_width:.1f}mm, "
            f"channels={filled_channels}/{self.n_channels} filled, "
            f"f_res={self.get_resonant_frequency()/1e9:.2f}GHz"
            f")"
        )