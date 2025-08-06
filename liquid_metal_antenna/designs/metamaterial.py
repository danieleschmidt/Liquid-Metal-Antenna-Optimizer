"""
Metamaterial-inspired liquid metal antenna designs.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from ..core.antenna_spec import AntennaSpec


class MetasurfaceAntenna:
    """
    Liquid metal metasurface antenna with reconfigurable unit cells.
    
    This class implements a metasurface antenna using liquid metal
    unit cells that can be reconfigured for different radiation patterns.
    """
    
    def __init__(
        self,
        unit_cell: str = 'jerusalem_cross',
        periodicity: float = 6.0,  # mm
        n_cells: Tuple[int, int] = (10, 10),
        tuning_mechanism: str = 'liquid_metal_vias',
        substrate_height: float = 1.6  # mm
    ):
        """
        Initialize metasurface antenna.
        
        Args:
            unit_cell: Type of unit cell geometry
            periodicity: Unit cell period in mm
            n_cells: Number of unit cells (nx, ny)
            tuning_mechanism: Tuning method
            substrate_height: Substrate thickness in mm
        """
        self.unit_cell = unit_cell
        self.periodicity = periodicity
        self.n_cells = n_cells
        self.tuning_mechanism = tuning_mechanism
        self.substrate_height = substrate_height
        
        # Unit cell states (e.g., via fill states)
        nx, ny = n_cells
        self.cell_states = np.zeros((nx, ny), dtype=int)  # 0-255 for different states
        
        # Design parameters
        self._setup_unit_cell_parameters()
    
    def _setup_unit_cell_parameters(self) -> None:
        """Setup unit cell design parameters."""
        if self.unit_cell == 'jerusalem_cross':
            self.unit_cell_params = {
                'cross_length': self.periodicity * 0.8,
                'cross_width': self.periodicity * 0.1,
                'arm_length': self.periodicity * 0.3,
                'arm_width': self.periodicity * 0.05,
                'n_tuning_elements': 4  # Four corner vias
            }
        elif self.unit_cell == 'split_ring':
            self.unit_cell_params = {
                'outer_radius': self.periodicity * 0.4,
                'inner_radius': self.periodicity * 0.3,
                'gap_width': self.periodicity * 0.1,
                'metal_width': self.periodicity * 0.05,
                'n_tuning_elements': 2  # Gap tuning elements
            }
        else:
            # Default square patch
            self.unit_cell_params = {
                'patch_size': self.periodicity * 0.7,
                'metal_width': self.periodicity * 0.05,
                'n_tuning_elements': 1  # Central via
            }
    
    def set_cell_state(self, i: int, j: int, state: int) -> None:
        """Set the state of a specific unit cell."""
        nx, ny = self.n_cells
        if 0 <= i < nx and 0 <= j < ny:
            max_states = 2 ** self.unit_cell_params['n_tuning_elements']
            self.cell_states[i, j] = min(max(state, 0), max_states - 1)
        else:
            raise ValueError(f"Cell ({i}, {j}) out of bounds")
    
    def set_pattern_configuration(self, pattern: str = 'uniform') -> None:
        """Set predefined cell configuration patterns."""
        nx, ny = self.n_cells
        
        if pattern == 'uniform':
            # All cells in same state
            self.cell_states[:, :] = 1
        
        elif pattern == 'checkerboard':
            # Alternating pattern
            for i in range(nx):
                for j in range(ny):
                    self.cell_states[i, j] = (i + j) % 2
        
        elif pattern == 'gradient':
            # Linear phase gradient
            max_states = 2 ** self.unit_cell_params['n_tuning_elements']
            for i in range(nx):
                state = int((i / (nx - 1)) * (max_states - 1))
                self.cell_states[i, :] = state
        
        elif pattern == 'lens':
            # Lens-like focusing pattern
            center_x, center_y = nx // 2, ny // 2
            max_states = 2 ** self.unit_cell_params['n_tuning_elements']
            
            for i in range(nx):
                for j in range(ny):
                    distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
                    state = int((distance / max_distance) * (max_states - 1))
                    self.cell_states[i, j] = state
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    def create_unit_cell_geometry(
        self,
        state: int,
        grid_resolution: float = 0.1e-3  # 0.1mm
    ) -> torch.Tensor:
        """
        Create geometry for a single unit cell.
        
        Args:
            state: Unit cell state (tuning configuration)
            grid_resolution: Grid resolution in meters
            
        Returns:
            Unit cell geometry tensor
        """
        # Grid size for unit cell
        cell_size_m = self.periodicity * 1e-3
        n_grid = int(cell_size_m / grid_resolution)
        
        # Initialize with zeros (no conductor)
        cell_geometry = torch.zeros((n_grid, n_grid), dtype=torch.float32)
        
        center = n_grid // 2
        
        if self.unit_cell == 'jerusalem_cross':
            # Main cross
            cross_len = int(self.unit_cell_params['cross_length'] * 1e-3 / grid_resolution)
            cross_wid = max(1, int(self.unit_cell_params['cross_width'] * 1e-3 / grid_resolution))
            
            # Horizontal bar
            h_start = center - cross_len // 2
            h_end = center + cross_len // 2
            v_start = center - cross_wid // 2
            v_end = center + cross_wid // 2
            
            if 0 <= h_start < n_grid and 0 <= h_end <= n_grid:
                cell_geometry[h_start:h_end, v_start:v_end] = 1.0
            
            # Vertical bar
            if 0 <= v_start < n_grid and 0 <= v_end <= n_grid:
                cell_geometry[v_start:v_end, h_start:h_end] = 1.0
            
            # Arms based on state
            arm_len = int(self.unit_cell_params['arm_length'] * 1e-3 / grid_resolution)
            arm_wid = max(1, int(self.unit_cell_params['arm_width'] * 1e-3 / grid_resolution))
            
            # Four arms controlled by 4-bit state
            if state & 0x1:  # Right arm
                arm_start = center + cross_len // 2
                arm_end = arm_start + arm_len
                if arm_end <= n_grid:
                    cell_geometry[arm_start:arm_end, center-arm_wid//2:center+arm_wid//2] = 1.0
            
            if state & 0x2:  # Left arm
                arm_end = center - cross_len // 2
                arm_start = arm_end - arm_len
                if arm_start >= 0:
                    cell_geometry[arm_start:arm_end, center-arm_wid//2:center+arm_wid//2] = 1.0
            
            if state & 0x4:  # Top arm
                arm_start = center + cross_len // 2
                arm_end = arm_start + arm_len
                if arm_end <= n_grid:
                    cell_geometry[center-arm_wid//2:center+arm_wid//2, arm_start:arm_end] = 1.0
            
            if state & 0x8:  # Bottom arm
                arm_end = center - cross_len // 2
                arm_start = arm_end - arm_len
                if arm_start >= 0:
                    cell_geometry[center-arm_wid//2:center+arm_wid//2, arm_start:arm_end] = 1.0
        
        elif self.unit_cell == 'split_ring':
            # Create split ring resonator
            outer_r = int(self.unit_cell_params['outer_radius'] * 1e-3 / grid_resolution)
            inner_r = int(self.unit_cell_params['inner_radius'] * 1e-3 / grid_resolution)
            gap_w = max(1, int(self.unit_cell_params['gap_width'] * 1e-3 / grid_resolution))
            
            # Create ring (simplified rectangular approximation)
            ring_thickness = outer_r - inner_r
            
            # Outer square
            outer_start = center - outer_r
            outer_end = center + outer_r
            
            if 0 <= outer_start and outer_end <= n_grid:
                cell_geometry[outer_start:outer_end, outer_start:outer_start+ring_thickness] = 1.0  # Left
                cell_geometry[outer_start:outer_end, outer_end-ring_thickness:outer_end] = 1.0       # Right
                cell_geometry[outer_start:outer_start+ring_thickness, outer_start:outer_end] = 1.0  # Top
                cell_geometry[outer_end-ring_thickness:outer_end, outer_start:outer_end] = 1.0      # Bottom
                
                # Remove gap based on state
                gap_start = center - gap_w // 2
                gap_end = center + gap_w // 2
                
                if not (state & 0x1):  # Gap in right side
                    cell_geometry[gap_start:gap_end, outer_end-ring_thickness:outer_end] = 0.0
        
        else:
            # Default square patch
            patch_size = int(self.unit_cell_params['patch_size'] * 1e-3 / grid_resolution)
            
            if state > 0:
                patch_start = center - patch_size // 2
                patch_end = center + patch_size // 2
                
                if 0 <= patch_start and patch_end <= n_grid:
                    cell_geometry[patch_start:patch_end, patch_start:patch_end] = 1.0
        
        return cell_geometry
    
    def create_geometry_tensor(
        self,
        grid_resolution: float = 0.1e-3,  # 0.1mm
        include_substrate: bool = True
    ) -> torch.Tensor:
        """
        Create complete metasurface geometry tensor.
        
        Args:
            grid_resolution: Grid resolution in meters
            include_substrate: Include substrate layer
            
        Returns:
            Complete metasurface geometry tensor
        """
        # Total metasurface dimensions
        nx, ny = self.n_cells
        total_width = nx * self.periodicity * 1e-3  # Convert to meters
        total_length = ny * self.periodicity * 1e-3
        total_height = self.substrate_height * 1e-3
        
        # Grid dimensions
        grid_nx = int(total_width / grid_resolution)
        grid_ny = int(total_length / grid_resolution)
        grid_nz = max(8, int(total_height / grid_resolution))
        
        # Initialize geometry
        geometry = torch.zeros((grid_nx, grid_ny, grid_nz), dtype=torch.float32)
        
        # Add ground plane if requested
        if include_substrate:
            geometry[:, :, 0] = 1.0  # Ground plane at bottom
        
        # Unit cell size in grid cells
        cell_grid_size = int(self.periodicity * 1e-3 / grid_resolution)
        
        # Metasurface layer
        metasurface_z = grid_nz - 1
        
        # Place unit cells
        for i in range(nx):
            for j in range(ny):
                # Get unit cell geometry
                cell_state = self.cell_states[i, j]
                unit_cell = self.create_unit_cell_geometry(cell_state, grid_resolution)
                
                # Position in global grid
                start_x = i * cell_grid_size
                end_x = start_x + unit_cell.shape[0]
                start_y = j * cell_grid_size
                end_y = start_y + unit_cell.shape[1]
                
                # Ensure within bounds
                end_x = min(end_x, grid_nx)
                end_y = min(end_y, grid_ny)
                
                # Place unit cell
                actual_size_x = end_x - start_x
                actual_size_y = end_y - start_y
                
                if actual_size_x > 0 and actual_size_y > 0:
                    # Resize unit cell if needed
                    if unit_cell.shape[0] != actual_size_x or unit_cell.shape[1] != actual_size_y:
                        import torch.nn.functional as F
                        unit_cell = F.interpolate(
                            unit_cell.unsqueeze(0).unsqueeze(0),
                            size=(actual_size_x, actual_size_y),
                            mode='nearest'
                        ).squeeze()
                    
                    geometry[start_x:end_x, start_y:end_y, metasurface_z] = unit_cell
        
        return geometry
    
    def estimate_resonant_frequency(self) -> float:
        """Estimate resonant frequency based on unit cell design."""
        c = 299792458  # Speed of light
        
        if self.unit_cell == 'jerusalem_cross':
            # Resonance related to cross length
            effective_length = self.unit_cell_params['cross_length'] * 1e-3
            f_res = c / (2 * effective_length)  # Half-wave resonance
        
        elif self.unit_cell == 'split_ring':
            # Resonance related to ring circumference
            avg_radius = (self.unit_cell_params['outer_radius'] + 
                         self.unit_cell_params['inner_radius']) / 2
            circumference = 2 * np.pi * avg_radius * 1e-3
            f_res = c / circumference  # Full-wave resonance
        
        else:
            # Square patch resonance
            patch_size = self.unit_cell_params['patch_size'] * 1e-3
            f_res = c / (2 * patch_size)
        
        return f_res
    
    def get_phase_distribution(self) -> np.ndarray:
        """Get current phase distribution across the metasurface."""
        nx, ny = self.n_cells
        max_states = 2 ** self.unit_cell_params['n_tuning_elements']
        
        # Convert cell states to phase values
        phase_distribution = np.zeros((nx, ny))
        
        for i in range(nx):
            for j in range(ny):
                state = self.cell_states[i, j]
                # Linear mapping from state to phase
                phase_distribution[i, j] = (state / (max_states - 1)) * 2 * np.pi
        
        return phase_distribution
    
    def export_config(self, filename: str) -> None:
        """Export metasurface configuration."""
        import json
        
        config = {
            'antenna_type': 'MetasurfaceAntenna',
            'unit_cell': self.unit_cell,
            'periodicity': self.periodicity,
            'n_cells': self.n_cells,
            'tuning_mechanism': self.tuning_mechanism,
            'substrate_height': self.substrate_height,
            'unit_cell_params': self.unit_cell_params,
            'cell_states': self.cell_states.tolist(),
            'estimated_frequency': self.estimate_resonant_frequency()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Metasurface configuration exported to {filename}")
    
    @classmethod
    def load_config(cls, filename: str) -> 'MetasurfaceAntenna':
        """Load metasurface configuration."""
        import json
        
        with open(filename, 'r') as f:
            config = json.load(f)
        
        antenna = cls(
            unit_cell=config['unit_cell'],
            periodicity=config['periodicity'],
            n_cells=tuple(config['n_cells']),
            tuning_mechanism=config['tuning_mechanism'],
            substrate_height=config['substrate_height']
        )
        
        antenna.cell_states = np.array(config['cell_states'])
        
        return antenna
    
    def __repr__(self) -> str:
        nx, ny = self.n_cells
        f_res = self.estimate_resonant_frequency()
        active_cells = np.sum(self.cell_states > 0)
        
        return (
            f"MetasurfaceAntenna("
            f"{nx}x{ny} {self.unit_cell} cells, "
            f"period={self.periodicity:.1f}mm, "
            f"f_res≈{f_res/1e9:.2f}GHz, "
            f"active={active_cells}/{nx*ny}"
            f")"
        )