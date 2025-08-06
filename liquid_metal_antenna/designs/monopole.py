"""
Liquid metal monopole antenna designs.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from ..core.antenna_spec import AntennaSpec


class LiquidMetalMonopole:
    """
    Reconfigurable monopole antenna using liquid metal channels.
    
    This class implements a monopole antenna with variable length achieved
    through liquid metal filling of microfluidic channels.
    """
    
    def __init__(
        self,
        max_height: float = 30.0,  # mm
        channel_width: float = 1.0,  # mm
        n_segments: int = 5,
        ground_plane_size: Tuple[float, float] = (40.0, 40.0)  # mm
    ):
        """
        Initialize liquid metal monopole.
        
        Args:
            max_height: Maximum antenna height in mm
            channel_width: Width of liquid metal channel in mm
            n_segments: Number of controllable segments
            ground_plane_size: Ground plane dimensions (width, length) in mm
        """
        self.max_height = max_height
        self.channel_width = channel_width
        self.n_segments = n_segments
        self.ground_plane_size = ground_plane_size
        
        # Segment configuration
        self.segment_height = max_height / n_segments
        self.segment_states = np.zeros(n_segments, dtype=bool)
        
        # Default: fill first segment (minimum antenna)
        self.segment_states[0] = True
    
    def set_active_segments(self, n_active: int) -> None:
        """Set number of active segments from bottom up."""
        if n_active < 1:
            raise ValueError("At least one segment must be active")
        if n_active > self.n_segments:
            raise ValueError(f"Cannot activate more than {self.n_segments} segments")
        
        # Fill segments from bottom up
        self.segment_states[:] = False
        self.segment_states[:n_active] = True
    
    def set_segment_configuration(self, states: List[bool]) -> None:
        """Set custom segment configuration."""
        if len(states) != self.n_segments:
            raise ValueError(f"Expected {self.n_segments} segment states")
        
        self.segment_states = np.array(states, dtype=bool)
    
    def get_active_height(self) -> float:
        """Get current active antenna height in mm."""
        if not np.any(self.segment_states):
            return 0.0
        
        # Find highest active segment
        highest_active = np.where(self.segment_states)[0][-1]
        return (highest_active + 1) * self.segment_height
    
    def get_resonant_frequency(self) -> float:
        """Calculate theoretical resonant frequency."""
        active_height = self.get_active_height()
        if active_height == 0:
            return 0.0
        
        # Quarter-wave monopole resonance
        c = 299792458  # Speed of light
        
        # Effective length accounting for end effects
        effective_length = active_height * 1e-3 + 0.025 * self.channel_width * 1e-3
        
        # Quarter wavelength resonance
        f_res = c / (4 * effective_length)
        
        return f_res
    
    def create_geometry_tensor(
        self,
        grid_resolution: float = 0.5e-3,  # 0.5mm
        total_size: Tuple[float, float, float] = (50, 50, 40)  # mm
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
        
        center_x, center_y = nx // 2, ny // 2
        
        # Add ground plane (bottom layer)
        ground_z = 2  # Small offset from bottom
        gp_width_cells = int(self.ground_plane_size[0] * 1e-3 / grid_resolution)
        gp_length_cells = int(self.ground_plane_size[1] * 1e-3 / grid_resolution)
        
        gp_x_start = center_x - gp_width_cells // 2
        gp_x_end = center_x + gp_width_cells // 2
        gp_y_start = center_y - gp_length_cells // 2
        gp_y_end = center_y + gp_length_cells // 2
        
        geometry[gp_x_start:gp_x_end, gp_y_start:gp_y_end, ground_z] = 1.0
        
        # Add monopole segments
        channel_width_cells = max(1, int(self.channel_width * 1e-3 / grid_resolution))
        segment_height_cells = int(self.segment_height * 1e-3 / grid_resolution)
        
        monopole_x_start = center_x - channel_width_cells // 2
        monopole_x_end = center_x + channel_width_cells // 2
        monopole_y_start = center_y - channel_width_cells // 2
        monopole_y_end = center_y + channel_width_cells // 2
        
        # Add active segments
        for i, active in enumerate(self.segment_states):
            if active:
                z_start = ground_z + 1 + i * segment_height_cells
                z_end = z_start + segment_height_cells
                
                # Ensure within bounds
                z_end = min(z_end, nz)
                
                if z_start < nz:
                    geometry[
                        monopole_x_start:monopole_x_end,
                        monopole_y_start:monopole_y_end,
                        z_start:z_end
                    ] = 1.0
        
        return geometry
    
    def get_frequency_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined configurations for different frequency bands."""
        configs = {}
        
        # Calculate frequency for each possible length
        for n_segments in range(1, self.n_segments + 1):
            height = n_segments * self.segment_height
            
            # Theoretical resonant frequency
            c = 299792458
            effective_length = height * 1e-3 + 0.025 * self.channel_width * 1e-3
            f_res = c / (4 * effective_length)
            
            # Create configuration
            states = [False] * self.n_segments
            states[:n_segments] = [True] * n_segments
            
            band_name = f"band_{n_segments}"
            
            configs[band_name] = {
                'segment_states': states,
                'active_height_mm': height,
                'resonant_frequency_ghz': f_res / 1e9,
                'description': f"{n_segments}-segment monopole ({height:.1f}mm)"
            }
        
        return configs
    
    def estimate_gain(self) -> float:
        """Estimate antenna gain."""
        active_height = self.get_active_height()
        
        if active_height == 0:
            return -50.0  # Very low gain for no antenna
        
        # Simple gain estimation for monopole
        # Typical monopole gain is around 2-5 dBi depending on height
        
        # Normalized height relative to quarter wavelength
        f_res = self.get_resonant_frequency()
        if f_res == 0:
            return -50.0
        
        c = 299792458
        quarter_wave = c / (4 * f_res)
        height_ratio = (active_height * 1e-3) / quarter_wave
        
        # Gain increases with height up to about 5/8 wavelength
        if height_ratio < 0.25:
            gain_dbi = 2.15 * height_ratio / 0.25  # Linear increase to quarter wave
        elif height_ratio <= 0.625:
            gain_dbi = 2.15 + 2.85 * (height_ratio - 0.25) / 0.375  # Up to 5 dBi at 5/8 wave
        else:
            gain_dbi = 5.0 - (height_ratio - 0.625)  # Decreases after 5/8 wave
        
        return max(gain_dbi, 0.0)
    
    def estimate_bandwidth(self) -> float:
        """Estimate antenna bandwidth."""
        f_res = self.get_resonant_frequency()
        
        if f_res == 0:
            return 0.0
        
        # Monopole bandwidth is typically 5-10% of resonant frequency
        # Thicker elements have wider bandwidth
        thickness_factor = self.channel_width / 1.0  # Relative to 1mm reference
        
        fractional_bw = 0.05 + 0.03 * min(thickness_factor, 2.0)  # 5-8% typical
        
        return f_res * fractional_bw
    
    def export_config(self, filename: str) -> None:
        """Export current configuration to file."""
        import json
        
        config = {
            'antenna_type': 'LiquidMetalMonopole',
            'max_height': self.max_height,
            'channel_width': self.channel_width,
            'n_segments': self.n_segments,
            'ground_plane_size': self.ground_plane_size,
            'segment_states': self.segment_states.tolist(),
            'active_height': self.get_active_height(),
            'resonant_frequency': self.get_resonant_frequency(),
            'estimated_gain': self.estimate_gain(),
            'estimated_bandwidth': self.estimate_bandwidth()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration exported to {filename}")
    
    @classmethod
    def load_config(cls, filename: str) -> 'LiquidMetalMonopole':
        """Load configuration from file."""
        import json
        
        with open(filename, 'r') as f:
            config = json.load(f)
        
        antenna = cls(
            max_height=config['max_height'],
            channel_width=config['channel_width'],
            n_segments=config['n_segments'],
            ground_plane_size=tuple(config['ground_plane_size'])
        )
        
        antenna.set_segment_configuration(config['segment_states'])
        
        return antenna
    
    def __repr__(self) -> str:
        active_segments = np.sum(self.segment_states)
        active_height = self.get_active_height()
        f_res = self.get_resonant_frequency()
        
        return (
            f"LiquidMetalMonopole("
            f"height={active_height:.1f}mm, "
            f"segments={active_segments}/{self.n_segments}, "
            f"f_res={f_res/1e9:.2f}GHz, "
            f"gain≈{self.estimate_gain():.1f}dBi"
            f")"
        )