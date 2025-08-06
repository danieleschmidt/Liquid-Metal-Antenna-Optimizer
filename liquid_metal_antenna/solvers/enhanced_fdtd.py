"""
Enhanced FDTD solver with improved robustness and convergence.
"""

import time
import warnings
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseSolver, SolverResult
from .fdtd import DifferentiableFDTD
from ..core.antenna_spec import AntennaSpec
from ..utils.validation import ValidationError, validate_geometry
from ..utils.logging_config import get_logger, LoggingContextManager
from ..utils.diagnostics import SystemDiagnostics


class ConvergenceMonitor:
    """Monitor and analyze FDTD convergence."""
    
    def __init__(self, window_size: int = 50):
        """
        Initialize convergence monitor.
        
        Args:
            window_size: Window size for moving average analysis
        """
        self.window_size = window_size
        self.energy_history = []
        self.field_max_history = []
        self.iteration_history = []
        self.converged = False
        self.convergence_iteration = None
        
    def update(self, iteration: int, fields: Dict[str, torch.Tensor]) -> bool:
        """
        Update convergence monitor with current field state.
        
        Args:
            iteration: Current iteration number
            fields: Current field values
            
        Returns:
            True if converged
        """
        # Calculate total field energy
        total_energy = sum(torch.sum(field ** 2).item() for field in fields.values())
        
        # Calculate maximum field magnitude
        field_max = max(torch.max(torch.abs(field)).item() for field in fields.values())
        
        # Store history
        self.energy_history.append(total_energy)
        self.field_max_history.append(field_max)
        self.iteration_history.append(iteration)
        
        # Check convergence
        if len(self.energy_history) >= self.window_size:
            self.converged = self._check_convergence()
            if self.converged and self.convergence_iteration is None:
                self.convergence_iteration = iteration
        
        return self.converged
    
    def _check_convergence(self) -> bool:
        """Check if fields have converged."""
        if len(self.energy_history) < self.window_size:
            return False
        
        # Get recent energy history
        recent_energies = self.energy_history[-self.window_size:]
        
        # Check for energy stability
        energy_std = np.std(recent_energies)
        energy_mean = np.mean(recent_energies)
        
        if energy_mean > 0:
            relative_std = energy_std / energy_mean
            
            # Converged if relative standard deviation is small
            if relative_std < 1e-6:
                return True
        
        # Check for field decay
        if len(self.field_max_history) >= self.window_size:
            recent_max_fields = self.field_max_history[-self.window_size:]
            
            # Check if fields are decaying to low levels
            if all(field < 1e-8 for field in recent_max_fields[-10:]):
                return True
        
        return False
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get convergence analysis metrics."""
        if len(self.energy_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Energy analysis
        energy_trend = np.polyfit(
            range(len(self.energy_history)), 
            self.energy_history, 
            1
        )[0] if len(self.energy_history) > 1 else 0
        
        # Field stability
        recent_fields = self.field_max_history[-min(50, len(self.field_max_history)):]
        field_stability = 1.0 / (1.0 + np.std(recent_fields)) if recent_fields else 0
        
        return {
            'status': 'converged' if self.converged else 'running',
            'convergence_iteration': self.convergence_iteration,
            'total_iterations': len(self.iteration_history),
            'final_energy': self.energy_history[-1] if self.energy_history else 0,
            'energy_trend': energy_trend,
            'field_stability': field_stability,
            'max_field_magnitude': self.field_max_history[-1] if self.field_max_history else 0
        }


class EnhancedFDTD(DifferentiableFDTD):
    """
    Enhanced FDTD solver with improved robustness and convergence.
    
    Features:
    - Adaptive time stepping
    - Convergence monitoring
    - Numerical stability checks
    - Memory optimization
    - Error recovery
    """
    
    def __init__(
        self,
        resolution: float = 0.5e-3,
        gpu_id: int = 0,
        precision: str = 'float32',
        pml_thickness: int = 8,
        courant_factor: float = 0.5,
        stability_check: bool = True,
        adaptive_stepping: bool = True,
        convergence_tolerance: float = 1e-6,
        max_memory_gb: float = 8.0
    ):
        """
        Initialize enhanced FDTD solver.
        
        Args:
            resolution: Grid resolution in meters
            gpu_id: GPU device ID
            precision: Computation precision
            pml_thickness: PML layer thickness
            courant_factor: Courant stability factor
            stability_check: Enable numerical stability monitoring
            adaptive_stepping: Enable adaptive time stepping
            convergence_tolerance: Convergence tolerance
            max_memory_gb: Maximum memory usage limit
        """
        super().__init__(resolution, gpu_id, precision, pml_thickness, courant_factor)
        
        self.stability_check = stability_check
        self.adaptive_stepping = adaptive_stepping
        self.convergence_tolerance = convergence_tolerance
        self.max_memory_gb = max_memory_gb
        
        # Enhanced monitoring
        self.convergence_monitor = ConvergenceMonitor()
        self.stability_monitor = []
        
        # Diagnostics
        self.diagnostics = SystemDiagnostics()
        
        # Logger
        self.logger = get_logger('enhanced_fdtd')
        
        # Error recovery state
        self.last_stable_state = None
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        self.logger.info("Enhanced FDTD solver initialized with robustness features")
    
    def simulate(
        self,
        geometry: Union[np.ndarray, torch.Tensor],
        frequency: Union[float, np.ndarray],
        excitation: str = 'coaxial_feed',
        compute_gradients: bool = True,
        max_time_steps: int = 5000,
        spec: Optional[AntennaSpec] = None,
        convergence_check_interval: int = 100,
        stability_check_interval: int = 50
    ) -> Union[SolverResult, Dict[str, torch.Tensor]]:
        """
        Run enhanced FDTD simulation with robustness features.
        
        Args:
            geometry: Antenna geometry tensor
            frequency: Simulation frequency in Hz
            excitation: Excitation type
            compute_gradients: Whether to enable gradient computation
            max_time_steps: Maximum time steps
            spec: Antenna specification
            convergence_check_interval: Iterations between convergence checks
            stability_check_interval: Iterations between stability checks
            
        Returns:
            SolverResult or raw fields
        """
        with LoggingContextManager("FDTD Simulation", self.logger):
            return self._run_simulation_with_monitoring(
                geometry, frequency, excitation, compute_gradients,
                max_time_steps, spec, convergence_check_interval, stability_check_interval
            )
    
    def _run_simulation_with_monitoring(
        self,
        geometry: Union[np.ndarray, torch.Tensor],
        frequency: Union[float, np.ndarray],
        excitation: str,
        compute_gradients: bool,
        max_time_steps: int,
        spec: Optional[AntennaSpec],
        convergence_check_interval: int,
        stability_check_interval: int
    ) -> Union[SolverResult, Dict[str, torch.Tensor]]:
        """Run simulation with comprehensive monitoring."""
        
        # Pre-simulation validation and setup
        self._validate_simulation_parameters(geometry, frequency, max_time_steps)
        
        start_time = time.time()
        
        # Convert geometry and setup
        if isinstance(geometry, np.ndarray):
            geometry = torch.from_numpy(geometry).to(dtype=self.dtype, device=self.device)
        
        if compute_gradients:
            geometry.requires_grad_(True)
        
        # Handle frequency
        if isinstance(frequency, np.ndarray):
            frequency = float(frequency[0]) if len(frequency) > 0 else 2.45e9
        else:
            frequency = float(frequency)
        
        # Create spec if not provided
        if spec is None:
            from ..core.antenna_spec import AntennaSpec, SubstrateMaterial, LiquidMetalType
            spec = AntennaSpec(
                frequency_range=(frequency * 0.9, frequency * 1.1),
                substrate=SubstrateMaterial.ROGERS_4003C,
                metal=LiquidMetalType.GALINSTAN
            )
        
        # Setup simulation
        self.set_grid_size(geometry, spec)
        self._check_memory_requirements()
        
        materials = self.create_geometry_mask(geometry, spec)
        fields = self.initialize_fields()
        source = self.create_source(excitation, frequency)
        
        # Reset monitoring
        self.convergence_monitor = ConvergenceMonitor()
        self.stability_monitor = []
        
        # Adaptive time stepping parameters
        current_dt = self.dt
        dt_scale_factor = 1.0
        
        self.logger.info(f"Starting simulation: {max_time_steps} max steps, "
                        f"grid={self._grid_size}, freq={frequency/1e9:.2f}GHz")
        
        # Main time stepping loop
        for t in range(max_time_steps):
            try:
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
                
                # Stability monitoring
                if self.stability_check and t % stability_check_interval == 0:
                    stability_ok = self._check_numerical_stability(fields, t)
                    if not stability_ok:
                        self.logger.warning(f"Numerical instability detected at step {t}")
                        
                        if self._attempt_recovery(fields, t):
                            self.logger.info(f"Recovered from instability at step {t}")
                            continue
                        else:
                            self.logger.error(f"Failed to recover from instability")
                            break
                
                # Convergence monitoring
                if t % convergence_check_interval == 0:
                    converged = self.convergence_monitor.update(t, fields)
                    
                    if converged:
                        self.logger.info(f"Converged at iteration {t}")
                        break
                
                # Adaptive time stepping
                if self.adaptive_stepping and t % 200 == 0:
                    dt_scale_factor = self._adapt_time_step(fields, t)
                    if dt_scale_factor != 1.0:
                        current_dt = self.dt * dt_scale_factor
                        self.logger.debug(f"Adapted time step by factor {dt_scale_factor:.3f}")
                
                # Memory management
                if t % 500 == 0:
                    self._manage_memory()
                
                # Progress logging
                if t % 1000 == 0 and t > 0:
                    convergence_metrics = self.convergence_monitor.get_convergence_metrics()
                    self.logger.info(f"Step {t}: energy={convergence_metrics.get('final_energy', 0):.2e}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"GPU memory error at step {t}: {str(e)}")
                    if self._handle_memory_error():
                        continue
                    else:
                        break
                else:
                    self.logger.error(f"Runtime error at step {t}: {str(e)}")
                    break
            
            except Exception as e:
                self.logger.error(f"Unexpected error at step {t}: {str(e)}")
                break
        
        computation_time = time.time() - start_time
        
        # Get convergence metrics
        convergence_metrics = self.convergence_monitor.get_convergence_metrics()
        
        self.logger.info(f"Simulation completed: {t+1} steps, "
                        f"{computation_time:.2f}s, "
                        f"status={convergence_metrics.get('status', 'unknown')}")
        
        # Return raw fields for gradient computation
        if compute_gradients:
            return fields
        
        # Compute results with error handling
        try:
            s_params = self.compute_s_parameters(fields, frequency)
            pattern, theta, phi = self.compute_radiation_pattern(fields, frequency)
            gain = self.compute_gain(pattern)
            vswr = self.compute_vswr(s_params[0, 0:1])
            
            # Enhanced result with convergence information
            result = SolverResult(
                s_parameters=s_params,
                frequencies=np.array([frequency]),
                radiation_pattern=pattern,
                theta_angles=theta,
                phi_angles=phi,
                gain_dbi=gain,
                max_gain_dbi=gain,
                vswr=vswr,
                converged=convergence_metrics.get('status') == 'converged',
                iterations=t + 1,
                convergence_error=1.0 - convergence_metrics.get('field_stability', 0),
                computation_time=computation_time
            )
            
            # Add enhanced metrics
            result.convergence_metrics = convergence_metrics
            result.stability_score = self._calculate_stability_score()
            result.memory_usage_gb = self._get_current_memory_usage()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error computing simulation results: {str(e)}")
            
            # Return minimal result
            return SolverResult(
                s_parameters=np.array([[[complex(0.1, 0.0)]]], dtype=complex),
                frequencies=np.array([frequency]),
                converged=False,
                iterations=t + 1,
                computation_time=computation_time
            )
    
    def _validate_simulation_parameters(
        self,
        geometry: Union[np.ndarray, torch.Tensor],
        frequency: Union[float, np.ndarray],
        max_time_steps: int
    ) -> None:
        """Validate simulation parameters before starting."""
        try:
            # Geometry validation
            validate_geometry(geometry, max_memory_gb=self.max_memory_gb)
            
            # Frequency validation
            if isinstance(frequency, (list, np.ndarray)):
                for f in frequency:
                    if not (1e6 <= f <= 100e9):
                        raise ValidationError(f"Frequency {f/1e9:.2f} GHz outside valid range")
            else:
                if not (1e6 <= frequency <= 100e9):
                    raise ValidationError(f"Frequency {frequency/1e9:.2f} GHz outside valid range")
            
            # Time step validation
            if max_time_steps <= 0:
                raise ValidationError("max_time_steps must be positive")
            if max_time_steps > 100000:
                self.logger.warning(f"Very large time step count: {max_time_steps}")
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {str(e)}")
            raise
    
    def _check_memory_requirements(self) -> None:
        """Check if simulation will fit in available memory."""
        estimated_memory = self.estimate_memory_usage()
        
        # Get system memory info
        try:
            system_metrics = self.diagnostics.get_system_metrics()
            available_memory = system_metrics.memory_available_gb
            
            if self.device.startswith('cuda'):
                gpu_memory = system_metrics.gpu_memory_total_gb - system_metrics.gpu_memory_used_gb
                available_memory = min(available_memory, gpu_memory)
            
            if estimated_memory > available_memory * 0.9:  # Leave 10% headroom
                raise ValidationError(
                    f"Insufficient memory: need {estimated_memory:.1f}GB, "
                    f"available {available_memory:.1f}GB"
                )
            
        except Exception as e:
            self.logger.warning(f"Memory check failed: {str(e)}")
    
    def _check_numerical_stability(self, fields: Dict[str, torch.Tensor], iteration: int) -> bool:
        """Check for numerical instabilities."""
        stability_metrics = {}
        
        # Check for NaN or infinity
        for field_name, field in fields.items():
            if not torch.isfinite(field).all():
                stability_metrics[f'{field_name}_finite'] = False
                self.logger.warning(f"Non-finite values in {field_name} at iteration {iteration}")
                return False
            else:
                stability_metrics[f'{field_name}_finite'] = True
        
        # Check field magnitudes
        max_field_magnitudes = {}
        for field_name, field in fields.items():
            max_magnitude = torch.max(torch.abs(field)).item()
            max_field_magnitudes[field_name] = max_magnitude
            
            # Check for excessive growth
            if max_magnitude > 1e6:
                stability_metrics[f'{field_name}_magnitude'] = False
                self.logger.warning(f"Excessive field magnitude in {field_name}: {max_magnitude:.2e}")
                return False
            else:
                stability_metrics[f'{field_name}_magnitude'] = True
        
        # Check total energy growth
        total_energy = sum(torch.sum(field ** 2).item() for field in fields.values())
        if len(self.stability_monitor) > 0:
            energy_ratio = total_energy / (self.stability_monitor[-1].get('total_energy', 1e-10))
            if energy_ratio > 100:  # 100x energy growth indicates instability
                stability_metrics['energy_growth'] = False
                self.logger.warning(f"Excessive energy growth: {energy_ratio:.2e}x")
                return False
            else:
                stability_metrics['energy_growth'] = True
        
        # Store stability metrics
        stability_metrics['iteration'] = iteration
        stability_metrics['total_energy'] = total_energy
        stability_metrics['max_field_magnitudes'] = max_field_magnitudes
        self.stability_monitor.append(stability_metrics)
        
        # Keep only recent history
        if len(self.stability_monitor) > 100:
            self.stability_monitor.pop(0)
        
        return True
    
    def _attempt_recovery(self, fields: Dict[str, torch.Tensor], iteration: int) -> bool:
        """Attempt to recover from numerical instability."""
        if self.recovery_attempts >= self.max_recovery_attempts:
            return False
        
        self.recovery_attempts += 1
        self.logger.info(f"Attempting recovery #{self.recovery_attempts}")
        
        # Strategy 1: Scale down fields
        scale_factor = 0.1
        for field_name, field in fields.items():
            fields[field_name] = field * scale_factor
        
        # Strategy 2: Reduce time step
        self.dt *= 0.5
        
        # Strategy 3: Clear high-frequency components (simple low-pass filter)
        try:
            for field_name, field in fields.items():
                if field.dim() >= 3:
                    # Apply simple smoothing
                    kernel = torch.ones(3, 3, 3, device=field.device) / 27
                    kernel = kernel.unsqueeze(0).unsqueeze(0)
                    
                    # Pad and convolve
                    field_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='reflect')
                    smoothed = F.conv3d(field_padded, kernel, padding=0)
                    fields[field_name] = smoothed.squeeze(0).squeeze(0)
        
        except Exception as e:
            self.logger.warning(f"Field smoothing failed: {str(e)}")
        
        return True
    
    def _adapt_time_step(self, fields: Dict[str, torch.Tensor], iteration: int) -> float:
        """Adapt time step based on field behavior."""
        if not self.adaptive_stepping:
            return 1.0
        
        # Calculate field change rate
        total_energy = sum(torch.sum(field ** 2).item() for field in fields.values())
        
        if len(self.stability_monitor) < 2:
            return 1.0
        
        # Get energy from previous check
        prev_energy = self.stability_monitor[-2].get('total_energy', total_energy)
        energy_change_rate = abs(total_energy - prev_energy) / (prev_energy + 1e-10)
        
        # Adapt time step
        if energy_change_rate > 0.1:  # Rapid changes - reduce time step
            return 0.8
        elif energy_change_rate < 0.001:  # Slow changes - increase time step
            return 1.2
        else:
            return 1.0
    
    def _manage_memory(self) -> None:
        """Manage GPU memory usage."""
        if 'cuda' in self.device:
            try:
                import torch
                torch.cuda.empty_cache()
                
                # Get memory usage
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9
                
                if memory_reserved > self.max_memory_gb * 0.9:
                    self.logger.warning(f"High GPU memory usage: {memory_reserved:.1f}GB")
                
            except Exception:
                pass
    
    def _handle_memory_error(self) -> bool:
        """Handle GPU memory errors."""
        self.logger.info("Attempting to recover from memory error")
        
        try:
            import torch
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Reduce precision if possible
            if self.precision == 'float32':
                self.precision = 'float16'
                self.dtype = torch.float16
                self.logger.info("Reduced precision to float16")
                return True
            
            # Reduce grid size
            nx, ny, nz = self._grid_size
            self._grid_size = (nx//2, ny//2, nz//2)
            self.logger.info(f"Reduced grid size to {self._grid_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {str(e)}")
            return False
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall stability score from 0-1."""
        if not self.stability_monitor:
            return 0.5
        
        # Count stable iterations
        stable_iterations = sum(
            1 for metrics in self.stability_monitor
            if all(metrics.get(key, False) for key in ['energy_growth'] 
                  if key in metrics)
        )
        
        total_iterations = len(self.stability_monitor)
        stability_score = stable_iterations / total_iterations if total_iterations > 0 else 0.5
        
        return stability_score
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            if 'cuda' in self.device:
                import torch
                return torch.cuda.memory_allocated(self.device) / 1e9
            else:
                # For CPU, use process memory
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1e9
        except Exception:
            return 0.0
    
    def get_solver_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive solver diagnostics."""
        convergence_metrics = self.convergence_monitor.get_convergence_metrics()
        stability_score = self._calculate_stability_score()
        memory_usage = self._get_current_memory_usage()
        
        return {
            'solver_type': 'EnhancedFDTD',
            'configuration': {
                'resolution': self.resolution,
                'precision': self.precision,
                'grid_size': self._grid_size,
                'pml_thickness': self.pml_thickness,
                'courant_factor': self.courant_factor,
                'stability_check': self.stability_check,
                'adaptive_stepping': self.adaptive_stepping
            },
            'convergence_metrics': convergence_metrics,
            'stability_score': stability_score,
            'memory_usage_gb': memory_usage,
            'recovery_attempts': self.recovery_attempts,
            'stability_checks_performed': len(self.stability_monitor)
        }
    
    def export_diagnostics_report(self, filename: str) -> None:
        """Export comprehensive diagnostics report."""
        diagnostics = self.get_solver_diagnostics()
        
        # Add system metrics
        system_metrics = self.diagnostics.get_system_metrics()
        diagnostics['system_metrics'] = system_metrics.to_dict()
        
        # Add stability history
        diagnostics['stability_history'] = self.stability_monitor[-50:]  # Last 50 entries
        
        import json
        with open(filename, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        self.logger.info(f"Diagnostics report exported to {filename}")