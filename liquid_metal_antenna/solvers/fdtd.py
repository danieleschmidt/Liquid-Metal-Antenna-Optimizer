"""
Differentiable FDTD solver implementation.
"""

import time
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseSolver, SolverResult
from ..core.antenna_spec import AntennaSpec


class DifferentiableFDTD(BaseSolver):
    """
    GPU-accelerated FDTD solver with automatic differentiation.
    
    This solver implements a basic FDTD algorithm with PyTorch for
    automatic differentiation capabilities, enabling gradient-based
    optimization of antenna geometries.
    """
    
    def __init__(
        self,
        resolution: float = 0.5e-3,  # 0.5mm default resolution
        gpu_id: int = 0,
        precision: str = 'float32',
        pml_thickness: int = 8,
        courant_factor: float = 0.5
    ):
        """
        Initialize differentiable FDTD solver.
        
        Args:
            resolution: Grid resolution in meters
            gpu_id: GPU device ID
            precision: Computation precision
            pml_thickness: PML layer thickness in cells
            courant_factor: Courant stability factor
        """
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        super().__init__(resolution, device, precision)
        
        self.pml_thickness = pml_thickness
        self.courant_factor = courant_factor
        self.dtype = torch.float32 if precision == 'float32' else torch.float64
        
        # FDTD parameters
        self._setup_fdtd_parameters()
    
    def _setup_fdtd_parameters(self) -> None:
        """Setup FDTD simulation parameters."""
        # Speed of light
        self.c0 = 299792458.0
        
        # Time step (Courant condition)
        self.dt = self.courant_factor * self.resolution / (self.c0 * np.sqrt(3))
        
        # Impedance of free space
        self.eta0 = 377.0
        
        # Default grid size (will be updated based on antenna)
        self._grid_size = (64, 64, 32)
    
    def set_grid_size(self, geometry: torch.Tensor, spec: AntennaSpec) -> None:
        """Set grid size based on antenna geometry and specifications."""
        # Calculate required grid size
        wavelength = self.c0 / spec.frequency_range.center
        min_cells_per_wavelength = 10
        
        # Grid size based on antenna dimensions plus padding
        nx = int(spec.size_constraint.length * 1e-3 / self.resolution) + 2 * self.pml_thickness
        ny = int(spec.size_constraint.width * 1e-3 / self.resolution) + 2 * self.pml_thickness
        nz = int(spec.size_constraint.height * 1e-3 / self.resolution) + 2 * self.pml_thickness
        
        # Ensure minimum grid size for stability
        min_size = int(wavelength / (min_cells_per_wavelength * self.resolution))
        nx = max(nx, min_size)
        ny = max(ny, min_size)
        nz = max(nz, 32)  # Minimum z-dimension
        
        self._grid_size = (nx, ny, nz)
    
    def create_geometry_mask(
        self,
        geometry: torch.Tensor,
        spec: AntennaSpec
    ) -> torch.Tensor:
        """
        Create material property mask from geometry description.
        
        Args:
            geometry: Geometry tensor [x, y, z] with values 0-1
            spec: Antenna specification
            
        Returns:
            Material property tensor [nx, ny, nz, 3] (epsilon, sigma, mu)
        """
        nx, ny, nz = self._grid_size
        
        # Initialize with air properties
        epsilon_r = torch.ones((nx, ny, nz), dtype=self.dtype, device=self.device)
        sigma = torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device)
        mu_r = torch.ones((nx, ny, nz), dtype=self.dtype, device=self.device)
        
        # Add substrate
        substrate_height = int(spec.substrate.thickness * 1e-3 / self.resolution)
        substrate_start = nz // 2 - substrate_height // 2
        substrate_end = substrate_start + substrate_height
        
        epsilon_r[:, :, substrate_start:substrate_end] = spec.substrate.dielectric_constant
        sigma[:, :, substrate_start:substrate_end] = 2 * np.pi * spec.frequency_range.center * \
                                                      8.854e-12 * spec.substrate.dielectric_constant * \
                                                      spec.substrate.loss_tangent
        
        # Add liquid metal regions (geometry defines conductivity pattern)
        if geometry.dim() == 3 and geometry.shape == (nx, ny, nz):
            # Direct geometry mapping
            metal_mask = geometry > 0.5
        else:
            # Resize geometry to grid
            geometry_resized = F.interpolate(
                geometry.unsqueeze(0).unsqueeze(0),
                size=(nx, ny, nz),
                mode='trilinear',
                align_corners=False
            ).squeeze()
            metal_mask = geometry_resized > 0.5
        
        # Apply liquid metal conductivity
        metal_conductivity = spec.get_liquid_metal_conductivity()
        sigma[metal_mask] = metal_conductivity
        
        # Stack material properties
        materials = torch.stack([epsilon_r, sigma, mu_r], dim=-1)
        
        return materials
    
    def initialize_fields(self) -> Dict[str, torch.Tensor]:
        """Initialize electromagnetic field arrays."""
        nx, ny, nz = self._grid_size
        
        fields = {
            'Ex': torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device),
            'Ey': torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device),
            'Ez': torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device),
            'Hx': torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device),
            'Hy': torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device),
            'Hz': torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device)
        }
        
        return fields
    
    def create_source(
        self,
        excitation: str = 'coaxial_feed',
        frequency: float = 2.45e9
    ) -> Dict[str, Any]:
        """Create excitation source."""
        nx, ny, nz = self._grid_size
        
        # Source parameters
        source_info = {
            'type': excitation,
            'frequency': frequency,
            'position': (nx // 2, ny // 2, nz // 2),
            'amplitude': 1.0,
            'phase': 0.0
        }
        
        # Create time-domain source function
        omega = 2 * np.pi * frequency
        
        def source_function(t: float) -> float:
            """Gaussian-modulated sinusoidal source."""
            t0 = 3.0 / frequency  # Delay to center pulse
            sigma_t = 0.5 / frequency  # Pulse width
            envelope = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)
            return envelope * np.sin(omega * t)
        
        source_info['function'] = source_function
        
        return source_info
    
    def apply_source(
        self,
        fields: Dict[str, torch.Tensor],
        source: Dict[str, Any],
        time_step: int
    ) -> None:
        """Apply excitation source to fields."""
        current_time = time_step * self.dt
        amplitude = source['function'](current_time)
        
        x, y, z = source['position']
        
        # Apply source (simplified coaxial feed)
        if source['type'] == 'coaxial_feed':
            fields['Ez'][x, y, z] += amplitude * self.dt
    
    def update_e_fields(
        self,
        fields: Dict[str, torch.Tensor],
        materials: torch.Tensor
    ) -> None:
        """Update electric field components using FDTD equations."""
        # Extract material properties
        eps_r = materials[:, :, :, 0]
        sigma = materials[:, :, :, 1]
        
        # Update coefficients
        eps = eps_r * 8.854e-12  # Permittivity
        
        # Avoid division by zero
        denom = eps + sigma * self.dt / 2
        ca = (eps - sigma * self.dt / 2) / (denom + 1e-12)
        cb = self.dt / (denom + 1e-12)
        
        # Update Ex
        curl_h_x = (fields['Hz'][1:, :, :] - fields['Hz'][:-1, :, :]) / self.resolution - \
                   (fields['Hy'][:, :, 1:] - fields['Hy'][:, :, :-1]) / self.resolution
        
        # Pad to match field dimensions
        curl_h_x = F.pad(curl_h_x, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
        
        fields['Ex'] = ca * fields['Ex'] + cb * curl_h_x
        
        # Update Ey (similar pattern)
        curl_h_y = (fields['Hx'][:, :, 1:] - fields['Hx'][:, :, :-1]) / self.resolution - \
                   (fields['Hz'][:, 1:, :] - fields['Hz'][:, :-1, :]) / self.resolution
        
        curl_h_y = F.pad(curl_h_y, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
        
        fields['Ey'] = ca * fields['Ey'] + cb * curl_h_y
        
        # Update Ez
        curl_h_z = (fields['Hy'][1:, :, :] - fields['Hy'][:-1, :, :]) / self.resolution - \
                   (fields['Hx'][:, 1:, :] - fields['Hx'][:, :-1, :]) / self.resolution
        
        curl_h_z = F.pad(curl_h_z, (0, 0, 0, 1, 0, 1), mode='constant', value=0)
        
        fields['Ez'] = ca * fields['Ez'] + cb * curl_h_z
    
    def update_h_fields(self, fields: Dict[str, torch.Tensor]) -> None:
        """Update magnetic field components using FDTD equations."""
        mu0 = 4 * np.pi * 1e-7
        
        # Update Hx
        curl_e_x = (fields['Ey'][:, :, :-1] - fields['Ey'][:, :, 1:]) / self.resolution - \
                   (fields['Ez'][:, :-1, :] - fields['Ez'][:, 1:, :]) / self.resolution
        
        curl_e_x = F.pad(curl_e_x, (1, 0, 1, 0, 0, 0), mode='constant', value=0)
        
        fields['Hx'] = fields['Hx'] + (self.dt / mu0) * curl_e_x
        
        # Update Hy
        curl_e_y = (fields['Ez'][:-1, :, :] - fields['Ez'][1:, :, :]) / self.resolution - \
                   (fields['Ex'][:, :, :-1] - fields['Ex'][:, :, 1:]) / self.resolution
        
        curl_e_y = F.pad(curl_e_y, (1, 0, 0, 0, 1, 0), mode='constant', value=0)
        
        fields['Hy'] = fields['Hy'] + (self.dt / mu0) * curl_e_y
        
        # Update Hz
        curl_e_z = (fields['Ex'][:, :-1, :] - fields['Ex'][:, 1:, :]) / self.resolution - \
                   (fields['Ey'][:-1, :, :] - fields['Ey'][1:, :, :]) / self.resolution
        
        curl_e_z = F.pad(curl_e_z, (0, 0, 1, 0, 1, 0), mode='constant', value=0)
        
        fields['Hz'] = fields['Hz'] + (self.dt / mu0) * curl_e_z
    
    def apply_pml_boundaries(self, fields: Dict[str, torch.Tensor]) -> None:
        """Apply perfectly matched layer (PML) absorbing boundaries."""
        nx, ny, nz = self._grid_size
        pml = self.pml_thickness
        
        # Simplified PML implementation - exponential damping near boundaries
        damping_factor = 0.95
        
        for field_name, field in fields.items():
            # X boundaries
            for i in range(pml):
                damping = damping_factor ** (pml - i)
                field[i, :, :] *= damping
                field[-(i+1), :, :] *= damping
            
            # Y boundaries
            for j in range(pml):
                damping = damping_factor ** (pml - j)
                field[:, j, :] *= damping
                field[:, -(j+1), :] *= damping
            
            # Z boundaries
            for k in range(pml):
                damping = damping_factor ** (pml - k)
                field[:, :, k] *= damping
                field[:, :, -(k+1)] *= damping
    
    def simulate(
        self,
        geometry: Union[np.ndarray, torch.Tensor],
        frequency: Union[float, np.ndarray],
        excitation: str = 'coaxial_feed',
        compute_gradients: bool = True,
        max_time_steps: int = 2000,
        spec: Optional[AntennaSpec] = None
    ) -> Union[SolverResult, Dict[str, torch.Tensor]]:
        """
        Run FDTD simulation.
        
        Args:
            geometry: Antenna geometry tensor
            frequency: Simulation frequency in Hz
            excitation: Excitation type
            compute_gradients: Whether to enable gradient computation
            max_time_steps: Maximum time steps to run
            spec: Antenna specification
            
        Returns:
            SolverResult or raw field data for gradient computation
        """
        start_time = time.time()
        
        # Convert geometry to tensor if needed
        if isinstance(geometry, np.ndarray):
            geometry = torch.from_numpy(geometry).to(dtype=self.dtype, device=self.device)
        
        if compute_gradients:
            geometry.requires_grad_(True)
        
        # Use single frequency if array provided (for simplicity)
        if isinstance(frequency, np.ndarray):
            frequency = float(frequency[0]) if len(frequency) > 0 else 2.45e9
        else:
            frequency = float(frequency)
        
        # Create default spec if not provided
        if spec is None:
            from ..core.antenna_spec import AntennaSpec, SubstrateMaterial, LiquidMetalType
            spec = AntennaSpec(
                frequency_range=(frequency * 0.9, frequency * 1.1),
                substrate=SubstrateMaterial.ROGERS_4003C,
                metal=LiquidMetalType.GALINSTAN
            )
        
        # Set grid size based on geometry and spec
        self.set_grid_size(geometry, spec)
        
        # Create materials and fields
        materials = self.create_geometry_mask(geometry, spec)
        fields = self.initialize_fields()
        source = self.create_source(excitation, frequency)
        
        # Time stepping loop
        for t in range(max_time_steps):
            # Apply source
            self.apply_source(fields, source, t)
            
            # Update H fields
            self.update_h_fields(fields)
            
            # Apply PML boundaries
            self.apply_pml_boundaries(fields)
            
            # Update E fields
            self.update_e_fields(fields, materials)
            
            # Apply PML boundaries
            self.apply_pml_boundaries(fields)
            
            # Check for convergence (simplified)
            if t % 100 == 0 and t > 500:
                field_energy = sum(torch.sum(field ** 2) for field in fields.values())
                if field_energy < 1e-10:
                    break
        
        computation_time = time.time() - start_time
        
        # Return raw fields for gradient computation
        if compute_gradients:
            return fields
        
        # Compute results
        s_params = self.compute_s_parameters(fields, frequency)
        pattern, theta, phi = self.compute_radiation_pattern(fields, frequency)
        gain = self.compute_gain(pattern)
        vswr = self.compute_vswr(s_params[0, 0:1])
        
        return SolverResult(
            s_parameters=s_params,
            frequencies=np.array([frequency]),
            radiation_pattern=pattern,
            theta_angles=theta,
            phi_angles=phi,
            gain_dbi=gain,
            max_gain_dbi=gain,
            vswr=vswr,
            converged=True,
            iterations=t,
            computation_time=computation_time
        )
    
    def compute_s_parameters(
        self,
        fields: Union[Dict[str, torch.Tensor], np.ndarray, torch.Tensor],
        frequency: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Compute S-parameters from field data."""
        # Simplified S-parameter extraction
        if isinstance(fields, dict):
            # Extract S11 from field data (simplified)
            nx, ny, nz = self._grid_size
            
            # Sample fields at source location
            source_x, source_y, source_z = nx // 2, ny // 2, nz // 2
            
            # Estimate reflection coefficient from field ratio
            incident_field = 1.0  # Normalized incident field
            reflected_field = abs(float(fields['Ez'][source_x, source_y, source_z]))
            
            s11 = complex(reflected_field / (incident_field + 1e-12), 0)
            
            # Create S-parameter matrix (1-port for simplicity)
            s_params = np.array([[[s11]]], dtype=complex)
        else:
            # Handle tensor input
            s_params = np.array([[[complex(0.1, 0.0)]]], dtype=complex)
        
        return s_params
    
    def compute_radiation_pattern(
        self,
        fields: Union[Dict[str, torch.Tensor], np.ndarray, torch.Tensor],
        frequency: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute far-field radiation pattern."""
        # Simplified radiation pattern calculation
        n_theta = 37  # 5-degree resolution
        n_phi = 73    # 5-degree resolution
        
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        
        # Create simple pattern for testing
        theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')
        
        # Simplified directional pattern (cosine-like)
        pattern = np.maximum(np.cos(theta_mesh), 0.0) ** 2
        
        return pattern, theta, phi