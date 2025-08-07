"""
Advanced beam steering and phased array optimization for liquid metal antennas.

Implements beam forming, null steering, and multi-beam capabilities
with liquid metal reconfigurable elements.
"""

import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks

from ..utils.logging_config import get_logger, LoggingContextManager
from ..utils.validation import ValidationError
from ..solvers.base import SolverResult, BaseSolver
from ..core.antenna_spec import AntennaSpec
from ..core.optimizer import OptimizationResult


@dataclass
class BeamPattern:
    """Radiation pattern representation for beam analysis."""
    theta: np.ndarray  # Elevation angles (radians)
    phi: np.ndarray    # Azimuth angles (radians)
    gain: np.ndarray   # Gain pattern (linear scale)
    gain_db: np.ndarray  # Gain pattern (dB scale)
    max_gain: float    # Maximum gain (dB)
    main_lobe_direction: Tuple[float, float]  # (theta, phi) in radians
    side_lobe_level: float  # Maximum side lobe level (dB)
    beamwidth_3db: Tuple[float, float]  # 3dB beamwidths (theta, phi)
    nulls: List[Tuple[float, float]]  # Null directions


@dataclass
class BeamSteeringResult:
    """Result from beam steering optimization."""
    steering_angle: Tuple[float, float]  # Target steering angle
    achieved_angle: Tuple[float, float]  # Achieved steering angle
    beam_pattern: BeamPattern
    liquid_metal_states: np.ndarray  # Channel fill states
    phase_distribution: np.ndarray  # Phase distribution across array
    amplitude_distribution: np.ndarray  # Amplitude distribution
    steering_error: float  # Angular error in steering
    side_lobe_level: float  # Side lobe level (dB)
    efficiency: float  # Array efficiency
    gain_loss: float  # Gain loss due to steering (dB)


class LiquidMetalPhaseShifter:
    """Phase shifter implementation using liquid metal delays."""
    
    def __init__(
        self,
        max_delay: float = 1e-9,  # Maximum delay (seconds)
        channel_length: float = 20e-3,  # Channel length (meters)
        frequency: float = 2.45e9,  # Operating frequency
        liquid_metal_type: str = 'galinstan'
    ):
        """
        Initialize liquid metal phase shifter.
        
        Args:
            max_delay: Maximum achievable delay
            channel_length: Physical channel length
            frequency: Operating frequency
            liquid_metal_type: Type of liquid metal
        """
        self.max_delay = max_delay
        self.channel_length = channel_length
        self.frequency = frequency
        self.liquid_metal_type = liquid_metal_type
        
        self.logger = get_logger('phase_shifter')
        
        # Calculate material properties
        self.c = 299792458  # Speed of light
        self.wavelength = self.c / frequency
        
        # Liquid metal properties (simplified)
        if liquid_metal_type == 'galinstan':
            self.liquid_permittivity = 1.0  # Relative permittivity (approximate)
            self.liquid_conductivity = 3.46e6  # Conductivity (S/m)
        else:
            self.liquid_permittivity = 1.0
            self.liquid_conductivity = 1e6
        
        # Calculate phase shift resolution
        self.max_phase_shift = 2 * np.pi * self.max_delay * frequency
        self.phase_resolution = self.max_phase_shift / 256  # 8-bit resolution
        
    def calculate_phase_shift(self, fill_ratio: float) -> float:
        """
        Calculate phase shift for given channel fill ratio.
        
        Args:
            fill_ratio: Channel fill ratio (0-1)
            
        Returns:
            Phase shift in radians
        """
        # Effective delay based on fill ratio
        effective_delay = fill_ratio * self.max_delay
        
        # Phase shift calculation
        phase_shift = 2 * np.pi * self.frequency * effective_delay
        
        # Include dispersion effects (simplified)
        dispersion_factor = 1 + 0.1 * fill_ratio  # Empirical correction
        phase_shift *= dispersion_factor
        
        return phase_shift % (2 * np.pi)
    
    def calculate_required_fill_ratio(self, target_phase: float) -> float:
        """
        Calculate required fill ratio for target phase shift.
        
        Args:
            target_phase: Target phase shift (radians)
            
        Returns:
            Required fill ratio (0-1)
        """
        # Normalize phase to [0, 2π]
        target_phase = target_phase % (2 * np.pi)
        
        # Inverse calculation with dispersion correction
        if target_phase == 0:
            return 0.0
        
        # Iterative solution for dispersion compensation
        fill_ratio = target_phase / self.max_phase_shift
        
        for _ in range(5):  # Few iterations should suffice
            current_phase = self.calculate_phase_shift(fill_ratio)
            error = target_phase - current_phase
            
            if abs(error) < 0.01:  # Converged
                break
                
            # Update estimate
            gradient = self.max_phase_shift * (1 + 0.2 * fill_ratio)
            fill_ratio += error / gradient
            fill_ratio = np.clip(fill_ratio, 0, 1)
        
        return fill_ratio
    
    def get_bandwidth_limitation(self) -> float:
        """Calculate bandwidth limitation due to dispersion."""
        # Simplified dispersion analysis
        group_delay_variation = 0.1 * self.max_delay  # Empirical
        bandwidth_limit = 1 / (2 * np.pi * group_delay_variation)
        
        return bandwidth_limit


class BeamformingArray:
    """Liquid metal phased array with beam forming capabilities."""
    
    def __init__(
        self,
        n_elements: Tuple[int, int] = (8, 8),
        element_spacing: Tuple[float, float] = (0.5, 0.5),  # In wavelengths
        frequency: float = 2.45e9,
        array_geometry: str = 'rectangular',
        element_type: str = 'liquid_metal_patch',
        feed_network: str = 'series'
    ):
        """
        Initialize beamforming array.
        
        Args:
            n_elements: Number of elements (x, y)
            element_spacing: Element spacing in wavelengths
            frequency: Operating frequency
            array_geometry: Array geometry type
            element_type: Type of array elements
            feed_network: Feed network configuration
        """
        self.n_elements = n_elements
        self.element_spacing = element_spacing
        self.frequency = frequency
        self.array_geometry = array_geometry
        self.element_type = element_type
        self.feed_network = feed_network
        
        self.logger = get_logger('beamforming_array')
        
        # Calculate array properties
        self.wavelength = 299792458 / frequency
        self.total_elements = n_elements[0] * n_elements[1]
        
        # Generate element positions
        self.element_positions = self._generate_element_positions()
        
        # Initialize phase shifters
        self.phase_shifters = [
            LiquidMetalPhaseShifter(frequency=frequency)
            for _ in range(self.total_elements)
        ]
        
        # Array state
        self.current_weights = np.ones(self.total_elements, dtype=complex)
        self.current_phases = np.zeros(self.total_elements)
        self.liquid_metal_states = np.zeros(self.total_elements)
        
    def _generate_element_positions(self) -> np.ndarray:
        """Generate element positions in the array."""
        n_x, n_y = self.n_elements
        dx, dy = self.element_spacing
        
        # Convert spacing to meters
        dx_m = dx * self.wavelength
        dy_m = dy * self.wavelength
        
        positions = []
        
        if self.array_geometry == 'rectangular':
            # Rectangular grid
            for i in range(n_x):
                for j in range(n_y):
                    x = (i - (n_x - 1) / 2) * dx_m
                    y = (j - (n_y - 1) / 2) * dy_m
                    positions.append([x, y, 0])
        
        elif self.array_geometry == 'circular':
            # Circular array arrangement
            radius = min(n_x, n_y) * max(dx_m, dy_m) / 2
            n_total = self.total_elements
            
            for i in range(n_total):
                angle = 2 * np.pi * i / n_total
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append([x, y, 0])
        
        elif self.array_geometry == 'triangular':
            # Triangular lattice
            row = 0
            col = 0
            
            for i in range(self.total_elements):
                if row % 2 == 0:
                    x = col * dx_m
                else:
                    x = (col + 0.5) * dx_m
                
                y = row * dy_m * np.sqrt(3) / 2
                positions.append([x, y, 0])
                
                col += 1
                if col >= n_x - (row % 2):
                    col = 0
                    row += 1
        
        return np.array(positions)
    
    def calculate_array_factor(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate array factor for given observation directions.
        
        Args:
            theta: Elevation angles (radians)
            phi: Azimuth angles (radians)
            weights: Complex element weights
            
        Returns:
            Array factor
        """
        if weights is None:
            weights = self.current_weights
        
        k = 2 * np.pi / self.wavelength  # Wave number
        
        # Create meshgrids for vectorized computation
        theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')
        
        # Direction vectors
        sin_theta = np.sin(theta_mesh)
        cos_theta = np.cos(theta_mesh)
        sin_phi = np.sin(phi_mesh)
        cos_phi = np.cos(phi_mesh)
        
        # Initialize array factor
        array_factor = np.zeros_like(theta_mesh, dtype=complex)
        
        # Sum contributions from all elements
        for i, pos in enumerate(self.element_positions):
            x, y, z = pos
            
            # Phase contribution from element position
            phase_contribution = k * (
                x * sin_theta * cos_phi +
                y * sin_theta * sin_phi +
                z * cos_theta
            )
            
            # Add weighted contribution
            array_factor += weights[i] * np.exp(1j * phase_contribution)
        
        return array_factor
    
    def steer_beam(
        self,
        target_theta: float,
        target_phi: float,
        method: str = 'phase_only'
    ) -> BeamSteeringResult:
        """
        Steer beam to target direction.
        
        Args:
            target_theta: Target elevation angle (radians)
            target_phi: Target azimuth angle (radians)
            method: Steering method ('phase_only', 'amplitude_phase')
            
        Returns:
            Beam steering result
        """
        self.logger.info(f"Steering beam to θ={np.degrees(target_theta):.1f}°, "
                        f"φ={np.degrees(target_phi):.1f}°")
        
        k = 2 * np.pi / self.wavelength
        
        # Calculate required phase shifts for beam steering
        required_phases = np.zeros(self.total_elements)
        
        for i, pos in enumerate(self.element_positions):
            x, y, z = pos
            
            # Phase shift needed to steer to target direction
            phase_shift = -k * (
                x * np.sin(target_theta) * np.cos(target_phi) +
                y * np.sin(target_theta) * np.sin(target_phi) +
                z * np.cos(target_theta)
            )
            
            required_phases[i] = phase_shift
        
        # Normalize phases to reference element (center element)
        ref_element = len(required_phases) // 2
        required_phases = required_phases - required_phases[ref_element]
        required_phases = (required_phases + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate liquid metal states for required phases
        liquid_metal_states = np.zeros(self.total_elements)
        achieved_phases = np.zeros(self.total_elements)
        
        for i in range(self.total_elements):
            fill_ratio = self.phase_shifters[i].calculate_required_fill_ratio(required_phases[i])
            liquid_metal_states[i] = fill_ratio
            achieved_phases[i] = self.phase_shifters[i].calculate_phase_shift(fill_ratio)
        
        # Update array state
        if method == 'phase_only':
            self.current_weights = np.exp(1j * achieved_phases)
        elif method == 'amplitude_phase':
            # Optimize both amplitude and phase (simplified)
            amplitudes = np.ones(self.total_elements)  # Equal amplitudes for now
            self.current_weights = amplitudes * np.exp(1j * achieved_phases)
        
        self.current_phases = achieved_phases
        self.liquid_metal_states = liquid_metal_states
        
        # Calculate beam pattern
        beam_pattern = self._calculate_beam_pattern()
        
        # Analyze steering performance
        steering_result = self._analyze_steering_performance(
            target_theta, target_phi, beam_pattern
        )
        
        return steering_result
    
    def optimize_null_steering(
        self,
        null_directions: List[Tuple[float, float]],
        main_beam_direction: Optional[Tuple[float, float]] = None,
        side_lobe_level: float = -20.0
    ) -> BeamSteeringResult:
        """
        Optimize array for null steering in specified directions.
        
        Args:
            null_directions: List of (theta, phi) null directions
            main_beam_direction: Desired main beam direction
            side_lobe_level: Maximum allowed side lobe level (dB)
            
        Returns:
            Optimized beam steering result
        """
        self.logger.info(f"Optimizing null steering for {len(null_directions)} null directions")
        
        def objective_function(weights_flat):
            # Reshape weights
            weights_real = weights_flat[:self.total_elements]
            weights_imag = weights_flat[self.total_elements:]
            weights = weights_real + 1j * weights_imag
            
            objective = 0.0
            
            # Null constraints
            for theta_null, phi_null in null_directions:
                af_null = self.calculate_array_factor(
                    np.array([theta_null]), np.array([phi_null]), weights
                )[0, 0]
                objective += abs(af_null) ** 2  # Minimize power at nulls
            
            # Main beam constraint
            if main_beam_direction:
                theta_main, phi_main = main_beam_direction
                af_main = self.calculate_array_factor(
                    np.array([theta_main]), np.array([phi_main]), weights
                )[0, 0]
                objective -= 10 * abs(af_main) ** 2  # Maximize power at main beam
            
            # Side lobe constraint (simplified)
            # This would need a more sophisticated implementation for real use
            total_power = np.sum(abs(weights) ** 2)
            objective += 0.1 * total_power  # Regularization
            
            return objective
        
        # Initial guess (uniform weights)
        initial_weights = np.ones(self.total_elements, dtype=complex)
        initial_real = np.real(initial_weights)
        initial_imag = np.imag(initial_weights)
        initial_guess = np.concatenate([initial_real, initial_imag])
        
        # Optimize
        result = minimize(
            objective_function,
            initial_guess,
            method='BFGS',
            options={'maxiter': 1000}
        )
        
        if result.success:
            # Extract optimized weights
            weights_real = result.x[:self.total_elements]
            weights_imag = result.x[self.total_elements:]
            optimized_weights = weights_real + 1j * weights_imag
            
            # Convert to phase and amplitude
            amplitudes = np.abs(optimized_weights)
            phases = np.angle(optimized_weights)
            
            # Calculate liquid metal states
            liquid_metal_states = np.zeros(self.total_elements)
            for i in range(self.total_elements):
                liquid_metal_states[i] = self.phase_shifters[i].calculate_required_fill_ratio(phases[i])
            
            # Update array state
            self.current_weights = optimized_weights
            self.current_phases = phases
            self.liquid_metal_states = liquid_metal_states
            
            # Calculate beam pattern
            beam_pattern = self._calculate_beam_pattern()
            
            # Create result
            main_direction = main_beam_direction or (0, 0)
            steering_result = self._analyze_steering_performance(
                main_direction[0], main_direction[1], beam_pattern
            )
            
            return steering_result
        
        else:
            self.logger.error("Null steering optimization failed")
            raise OptimizationError("Null steering optimization failed")
    
    def multi_beam_optimization(
        self,
        beam_directions: List[Tuple[float, float]],
        beam_weights: Optional[List[float]] = None
    ) -> List[BeamSteeringResult]:
        """
        Optimize array for multiple simultaneous beams.
        
        Args:
            beam_directions: List of desired beam directions
            beam_weights: Relative weights for each beam
            
        Returns:
            List of beam steering results
        """
        if beam_weights is None:
            beam_weights = [1.0] * len(beam_directions)
        
        self.logger.info(f"Optimizing for {len(beam_directions)} simultaneous beams")
        
        # Multi-beam optimization using superposition principle
        combined_weights = np.zeros(self.total_elements, dtype=complex)
        
        for i, (theta, phi) in enumerate(beam_directions):
            # Calculate weights for individual beam
            beam_result = self.steer_beam(theta, phi)
            
            # Add to combined weights with specified weight
            combined_weights += beam_weights[i] * self.current_weights
        
        # Normalize combined weights
        combined_weights = combined_weights / np.max(np.abs(combined_weights))
        
        # Update array state
        self.current_weights = combined_weights
        self.current_phases = np.angle(combined_weights)
        
        # Calculate liquid metal states
        liquid_metal_states = np.zeros(self.total_elements)
        for i in range(self.total_elements):
            liquid_metal_states[i] = self.phase_shifters[i].calculate_required_fill_ratio(
                self.current_phases[i]
            )
        
        self.liquid_metal_states = liquid_metal_states
        
        # Calculate beam pattern
        beam_pattern = self._calculate_beam_pattern()
        
        # Analyze each beam
        results = []
        for theta, phi in beam_directions:
            result = self._analyze_steering_performance(theta, phi, beam_pattern)
            results.append(result)
        
        return results
    
    def _calculate_beam_pattern(self) -> BeamPattern:
        """Calculate current beam pattern."""
        # Angular sampling
        theta = np.linspace(0, np.pi, 181)  # 1-degree resolution
        phi = np.linspace(0, 2*np.pi, 361)  # 1-degree resolution
        
        # Calculate array factor
        array_factor = self.calculate_array_factor(theta, phi, self.current_weights)
        
        # Convert to gain pattern
        gain_linear = np.abs(array_factor) ** 2
        gain_db = 10 * np.log10(gain_linear + 1e-10)  # Avoid log(0)
        
        # Find maximum gain and its direction
        max_idx = np.unravel_index(np.argmax(gain_db), gain_db.shape)
        max_gain = gain_db[max_idx]
        main_lobe_direction = (theta[max_idx[0]], phi[max_idx[1]])
        
        # Find side lobe level
        # Simple approach: find maximum excluding main lobe region
        main_lobe_mask = np.zeros_like(gain_db, dtype=bool)
        theta_idx, phi_idx = max_idx
        
        # Define main lobe region (±10 degrees)
        theta_margin = 10  # degrees
        phi_margin = 10   # degrees
        
        theta_range = slice(
            max(0, theta_idx - theta_margin),
            min(len(theta), theta_idx + theta_margin + 1)
        )
        phi_range = slice(
            max(0, phi_idx - phi_margin),
            min(len(phi), phi_idx + phi_margin + 1)
        )
        
        main_lobe_mask[theta_range, phi_range] = True
        side_lobe_gain = np.ma.masked_array(gain_db, mask=main_lobe_mask)
        side_lobe_level = np.max(side_lobe_gain) if side_lobe_gain.count() > 0 else -50.0
        
        # Calculate beamwidths (simplified - find 3dB points)
        # This is a simplified calculation; real implementation would be more sophisticated
        beamwidth_theta = self._calculate_beamwidth(gain_db[:, phi_idx], theta)
        beamwidth_phi = self._calculate_beamwidth(gain_db[theta_idx, :], phi)
        
        # Find nulls (local minima below threshold)
        null_threshold = max_gain - 40  # dB
        nulls = self._find_nulls(gain_db, theta, phi, null_threshold)
        
        return BeamPattern(
            theta=theta,
            phi=phi,
            gain=gain_linear,
            gain_db=gain_db,
            max_gain=max_gain,
            main_lobe_direction=main_lobe_direction,
            side_lobe_level=side_lobe_level,
            beamwidth_3db=(beamwidth_theta, beamwidth_phi),
            nulls=nulls
        )
    
    def _calculate_beamwidth(self, pattern_slice: np.ndarray, angles: np.ndarray) -> float:
        """Calculate 3dB beamwidth from pattern slice."""
        max_gain = np.max(pattern_slice)
        half_power_level = max_gain - 3  # 3dB down
        
        # Find indices where pattern crosses half-power level
        indices = np.where(pattern_slice >= half_power_level)[0]
        
        if len(indices) == 0:
            return 0.0
        
        # Calculate beamwidth
        angle_range = angles[indices[-1]] - angles[indices[0]]
        
        return angle_range
    
    def _find_nulls(
        self,
        gain_db: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        threshold: float
    ) -> List[Tuple[float, float]]:
        """Find null directions in the pattern."""
        nulls = []
        
        # Simple null detection - find local minima below threshold
        # This is a simplified implementation
        
        for i in range(1, len(theta) - 1):
            for j in range(1, len(phi) - 1):
                if gain_db[i, j] < threshold:
                    # Check if it's a local minimum
                    neighbors = [
                        gain_db[i-1, j], gain_db[i+1, j],
                        gain_db[i, j-1], gain_db[i, j+1]
                    ]
                    
                    if gain_db[i, j] < min(neighbors):
                        nulls.append((theta[i], phi[j]))
        
        # Limit number of nulls reported
        return nulls[:10]
    
    def _analyze_steering_performance(
        self,
        target_theta: float,
        target_phi: float,
        beam_pattern: BeamPattern
    ) -> BeamSteeringResult:
        """Analyze beam steering performance."""
        # Calculate steering error
        achieved_theta, achieved_phi = beam_pattern.main_lobe_direction
        
        steering_error = np.sqrt(
            (target_theta - achieved_theta) ** 2 +
            (target_phi - achieved_phi) ** 2
        )
        
        # Calculate gain loss (compared to broadside)
        # This would require a reference pattern for accurate calculation
        gain_loss = 0.0  # Placeholder
        
        # Calculate array efficiency (simplified)
        total_power = np.sum(np.abs(self.current_weights) ** 2)
        uniform_power = self.total_elements
        efficiency = total_power / uniform_power
        
        return BeamSteeringResult(
            steering_angle=(target_theta, target_phi),
            achieved_angle=(achieved_theta, achieved_phi),
            beam_pattern=beam_pattern,
            liquid_metal_states=self.liquid_metal_states.copy(),
            phase_distribution=self.current_phases.copy(),
            amplitude_distribution=np.abs(self.current_weights),
            steering_error=steering_error,
            side_lobe_level=beam_pattern.side_lobe_level,
            efficiency=efficiency,
            gain_loss=gain_loss
        )
    
    def get_array_status(self) -> Dict[str, Any]:
        """Get current array status."""
        return {
            'total_elements': self.total_elements,
            'frequency': self.frequency,
            'array_geometry': self.array_geometry,
            'element_spacing': self.element_spacing,
            'current_phases': self.current_phases.tolist(),
            'liquid_metal_states': self.liquid_metal_states.tolist(),
            'active_elements': np.sum(np.abs(self.current_weights) > 0.1),
            'max_phase_shift': np.max(self.current_phases) - np.min(self.current_phases),
            'average_fill_ratio': np.mean(self.liquid_metal_states)
        }
    
    def export_beam_pattern(self, filepath: str) -> None:
        """Export current beam pattern to file."""
        beam_pattern = self._calculate_beam_pattern()
        
        export_data = {
            'frequency': self.frequency,
            'array_geometry': self.array_geometry,
            'n_elements': self.n_elements,
            'element_spacing': self.element_spacing,
            'theta_degrees': np.degrees(beam_pattern.theta).tolist(),
            'phi_degrees': np.degrees(beam_pattern.phi).tolist(),
            'gain_db': beam_pattern.gain_db.tolist(),
            'max_gain': beam_pattern.max_gain,
            'main_lobe_direction_degrees': [
                np.degrees(beam_pattern.main_lobe_direction[0]),
                np.degrees(beam_pattern.main_lobe_direction[1])
            ],
            'side_lobe_level': beam_pattern.side_lobe_level,
            'beamwidth_3db_degrees': [
                np.degrees(beam_pattern.beamwidth_3db[0]),
                np.degrees(beam_pattern.beamwidth_3db[1])
            ],
            'liquid_metal_states': self.liquid_metal_states.tolist()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Beam pattern exported to {filepath}")


class AdaptiveBeamformer:
    """Adaptive beamforming with interference nulling."""
    
    def __init__(self, array: BeamformingArray):
        """
        Initialize adaptive beamformer.
        
        Args:
            array: Beamforming array instance
        """
        self.array = array
        self.logger = get_logger('adaptive_beamformer')
        
        # Adaptive parameters
        self.adaptation_rate = 0.01
        self.reference_signal = None
        self.interference_directions = []
        
    def train_adaptive_weights(
        self,
        signal_samples: np.ndarray,
        reference_signal: np.ndarray,
        algorithm: str = 'lms'
    ) -> np.ndarray:
        """
        Train adaptive weights using received signal samples.
        
        Args:
            signal_samples: Received signal samples (n_samples, n_elements)
            reference_signal: Reference signal samples
            algorithm: Adaptation algorithm ('lms', 'rls')
            
        Returns:
            Optimized weights
        """
        if algorithm == 'lms':
            return self._lms_adaptation(signal_samples, reference_signal)
        elif algorithm == 'rls':
            return self._rls_adaptation(signal_samples, reference_signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _lms_adaptation(
        self,
        signal_samples: np.ndarray,
        reference_signal: np.ndarray
    ) -> np.ndarray:
        """Least Mean Squares adaptive beamforming."""
        n_samples, n_elements = signal_samples.shape
        weights = np.ones(n_elements, dtype=complex)
        
        for i in range(n_samples):
            # Array output
            array_output = np.dot(weights.conj(), signal_samples[i, :])
            
            # Error signal
            error = reference_signal[i] - array_output
            
            # Weight update
            weights += self.adaptation_rate * error * signal_samples[i, :].conj()
        
        return weights
    
    def _rls_adaptation(
        self,
        signal_samples: np.ndarray,
        reference_signal: np.ndarray
    ) -> np.ndarray:
        """Recursive Least Squares adaptive beamforming."""
        n_samples, n_elements = signal_samples.shape
        weights = np.ones(n_elements, dtype=complex)
        
        # Initialize RLS parameters
        P = np.eye(n_elements) * 1000  # Inverse correlation matrix
        forgetting_factor = 0.99
        
        for i in range(n_samples):
            x = signal_samples[i, :].reshape(-1, 1)
            d = reference_signal[i]
            
            # Kalman gain
            k = P @ x / (forgetting_factor + x.conj().T @ P @ x)
            
            # A priori error
            error = d - weights.conj() @ x.flatten()
            
            # Weight update
            weights += (k * error.conj()).flatten()
            
            # Update inverse correlation matrix
            P = (P - k @ x.conj().T @ P) / forgetting_factor
        
        return weights


class OptimizationError(Exception):
    """Exception raised for optimization failures."""
    pass


# Utility functions for beam analysis
def calculate_array_gain(
    element_positions: np.ndarray,
    weights: np.ndarray,
    frequency: float,
    theta: float = 0,
    phi: float = 0
) -> float:
    """Calculate array gain in specified direction."""
    wavelength = 299792458 / frequency
    k = 2 * np.pi / wavelength
    
    # Direction vector
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Calculate array factor
    array_factor = 0
    for i, pos in enumerate(element_positions):
        phase = k * np.dot(pos, direction)
        array_factor += weights[i] * np.exp(1j * phase)
    
    # Array gain (simplified - assumes isotropic elements)
    gain = abs(array_factor) ** 2 / len(weights)
    
    return 10 * np.log10(gain)


def analyze_grating_lobes(
    element_spacing: Tuple[float, float],
    frequency: float,
    max_scan_angle: float = np.pi/3
) -> Dict[str, Any]:
    """Analyze grating lobe behavior."""
    wavelength = 299792458 / frequency
    dx, dy = element_spacing
    
    # Convert spacing to wavelengths
    dx_wl = dx * wavelength
    dy_wl = dy * wavelength
    
    # Grating lobe criteria
    max_spacing_x = wavelength / (1 + np.sin(max_scan_angle))
    max_spacing_y = wavelength / (1 + np.sin(max_scan_angle))
    
    grating_lobe_free_x = dx_wl <= max_spacing_x
    grating_lobe_free_y = dy_wl <= max_spacing_y
    
    return {
        'grating_lobe_free': grating_lobe_free_x and grating_lobe_free_y,
        'max_spacing_x_wavelengths': max_spacing_x / wavelength,
        'max_spacing_y_wavelengths': max_spacing_y / wavelength,
        'current_spacing_x_wavelengths': dx_wl / wavelength,
        'current_spacing_y_wavelengths': dy_wl / wavelength,
        'first_grating_lobe_angle_x': np.degrees(
            np.arcsin(wavelength / dx_wl - 1) if dx_wl > wavelength else np.nan
        ),
        'first_grating_lobe_angle_y': np.degrees(
            np.arcsin(wavelength / dy_wl - 1) if dy_wl > wavelength else np.nan
        )
    }