"""
Advanced antenna designs with multi-band and beam-forming capabilities.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
from dataclasses import dataclass

from ..core.antenna_spec import AntennaSpec
from ..utils.validation import ValidationError, validate_frequency_range
from ..utils.logging_config import get_logger
from .patch import ReconfigurablePatch


@dataclass
class BandConfiguration:
    """Configuration for a single frequency band."""
    
    frequency: float  # Center frequency in Hz
    bandwidth: float  # Bandwidth in Hz
    channel_states: List[bool]  # Channel fill states for this band
    gain_target: float = 0.0  # Target gain in dBi
    vswr_max: float = 2.0  # Maximum VSWR
    efficiency_min: float = 0.8  # Minimum efficiency
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        if not (1.0 <= self.vswr_max <= 10.0):
            raise ValueError("VSWR must be between 1.0 and 10.0")
        if not (0.0 <= self.efficiency_min <= 1.0):
            raise ValueError("Efficiency must be between 0.0 and 1.0")


class MultiBandPatchAntenna(ReconfigurablePatch):
    """Multi-band reconfigurable patch antenna with advanced control."""
    
    def __init__(
        self,
        substrate_height: float = 1.6,
        dielectric_constant: float = 4.4,
        n_channels: int = 12,  # More channels for better control
        channel_width: float = 0.4,
        patch_dimensions: Optional[Tuple[float, float]] = None,
        isolation_requirement: float = 20.0  # dB isolation between bands
    ):
        """
        Initialize multi-band patch antenna.
        
        Args:
            substrate_height: Substrate thickness in mm
            dielectric_constant: Relative permittivity
            n_channels: Number of liquid metal channels
            channel_width: Channel width in mm
            patch_dimensions: Patch dimensions (length, width) in mm
            isolation_requirement: Required isolation between bands in dB
        """
        super().__init__(
            substrate_height=substrate_height,
            dielectric_constant=dielectric_constant,
            n_channels=n_channels,
            channel_width=channel_width,
            patch_dimensions=patch_dimensions
        )
        
        self.isolation_requirement = isolation_requirement
        self.band_configurations = {}
        self.coupling_matrix = np.eye(n_channels)  # Initialize with no coupling
        
        self.logger = get_logger('multiband_patch')
        
        # Enhanced channel layout for multi-band operation
        self._create_advanced_channel_layout()
    
    def _create_advanced_channel_layout(self) -> None:
        """Create advanced channel layout for multi-band operation."""
        # Override parent method with more sophisticated layout
        self.channels = []
        
        # Create both horizontal and vertical channels for better control
        n_horizontal = self.n_channels // 2
        n_vertical = self.n_channels - n_horizontal
        
        # Horizontal channels for length tuning
        for i in range(n_horizontal):
            y_pos = (i + 1) * self.patch_length / (n_horizontal + 1) - self.patch_length / 2
            
            channel = {
                'id': i,
                'type': 'horizontal',
                'start': (-self.patch_width / 2, y_pos),
                'end': (self.patch_width / 2, y_pos),
                'width': self.channel_width,
                'filled': self.channel_states[i] if i < len(self.channel_states) else False
            }
            self.channels.append(channel)
        
        # Vertical channels for width tuning
        for i in range(n_vertical):
            channel_id = n_horizontal + i
            x_pos = (i + 1) * self.patch_width / (n_vertical + 1) - self.patch_width / 2
            
            channel = {
                'id': channel_id,
                'type': 'vertical',
                'start': (x_pos, -self.patch_length / 2),
                'end': (x_pos, self.patch_length / 2),
                'width': self.channel_width,
                'filled': self.channel_states[channel_id] if channel_id < len(self.channel_states) else False
            }
            self.channels.append(channel)
    
    def add_band_configuration(
        self,
        band_name: str,
        frequency: float,
        bandwidth: float,
        channel_states: List[bool],
        **kwargs
    ) -> None:
        """
        Add configuration for a frequency band.
        
        Args:
            band_name: Unique name for the band
            frequency: Center frequency in Hz
            bandwidth: Bandwidth in Hz
            channel_states: Channel fill states for this band
            **kwargs: Additional configuration parameters
        """
        try:
            validate_frequency_range((frequency - bandwidth/2, frequency + bandwidth/2))
            
            if len(channel_states) != self.n_channels:
                raise ValidationError(f"Expected {self.n_channels} channel states, got {len(channel_states)}")
            
            config = BandConfiguration(
                frequency=frequency,
                bandwidth=bandwidth,
                channel_states=channel_states,
                **kwargs
            )
            
            self.band_configurations[band_name] = config
            self.logger.info(f"Added band configuration: {band_name} @ {frequency/1e9:.2f} GHz")
            
        except Exception as e:
            self.logger.error(f"Failed to add band configuration: {str(e)}")
            raise
    
    def switch_to_band(self, band_name: str) -> None:
        """
        Switch antenna to specified frequency band.
        
        Args:
            band_name: Name of band configuration to activate
        """
        if band_name not in self.band_configurations:
            raise ValueError(f"Band configuration '{band_name}' not found")
        
        config = self.band_configurations[band_name]
        self.set_configuration(config.channel_states)
        
        self.logger.info(f"Switched to band: {band_name} ({config.frequency/1e9:.2f} GHz)")
    
    def optimize_multiband_configuration(
        self,
        target_frequencies: List[float],
        bandwidth_requirements: List[float],
        isolation_target: float = 20.0
    ) -> Dict[str, BandConfiguration]:
        """
        Optimize channel configurations for multi-band operation.
        
        Args:
            target_frequencies: Target center frequencies in Hz
            bandwidth_requirements: Required bandwidths in Hz
            isolation_target: Target isolation between bands in dB
            
        Returns:
            Dictionary of optimized band configurations
        """
        if len(target_frequencies) != len(bandwidth_requirements):
            raise ValueError("Frequency and bandwidth lists must have same length")
        
        self.logger.info(f"Optimizing for {len(target_frequencies)} bands")
        
        optimized_configs = {}
        
        for i, (freq, bw) in enumerate(zip(target_frequencies, bandwidth_requirements)):
            band_name = f"band_{i+1}"
            
            # Generate initial configuration based on frequency
            channel_config = self._generate_band_configuration(freq, bw)
            
            # Optimize for isolation if multiple bands
            if len(target_frequencies) > 1:
                channel_config = self._optimize_isolation(
                    channel_config, freq, target_frequencies, isolation_target
                )
            
            config = BandConfiguration(
                frequency=freq,
                bandwidth=bw,
                channel_states=channel_config,
                gain_target=3.0,  # Default target gain
                vswr_max=2.0,
                efficiency_min=0.8
            )
            
            optimized_configs[band_name] = config
            self.add_band_configuration(band_name, freq, bw, channel_config)
        
        self.logger.info("Multi-band optimization complete")
        return optimized_configs
    
    def _generate_band_configuration(self, frequency: float, bandwidth: float) -> List[bool]:
        """Generate channel configuration for specific frequency band."""
        # Simple heuristic: use frequency to determine which channels to fill
        base_freq = self.get_resonant_frequency()
        freq_ratio = frequency / base_freq
        
        # Determine number of channels to activate based on frequency ratio
        if freq_ratio < 0.7:
            # Lower frequency - need more channels (larger effective size)
            n_active = int(self.n_channels * 0.8)
        elif freq_ratio > 1.3:
            # Higher frequency - need fewer channels (smaller effective size)
            n_active = int(self.n_channels * 0.3)
        else:
            # Near base frequency
            n_active = int(self.n_channels * 0.5)
        
        # Create configuration with distributed channel activation
        configuration = [False] * self.n_channels
        
        if n_active > 0:
            # Distribute active channels evenly
            step = self.n_channels / n_active
            for i in range(n_active):
                idx = int(i * step)
                if idx < self.n_channels:
                    configuration[idx] = True
        
        return configuration
    
    def _optimize_isolation(
        self,
        base_config: List[bool],
        target_freq: float,
        all_frequencies: List[float],
        isolation_target: float
    ) -> List[bool]:
        """Optimize channel configuration for isolation between bands."""
        # Simplified isolation optimization
        # In practice, this would use electromagnetic simulation
        
        optimized_config = base_config.copy()
        
        # Avoid configurations that might cause coupling
        other_frequencies = [f for f in all_frequencies if f != target_freq]
        
        for other_freq in other_frequencies:
            freq_ratio = target_freq / other_freq
            
            # If frequencies are harmonically related, modify configuration
            if abs(freq_ratio - 2.0) < 0.1 or abs(freq_ratio - 0.5) < 0.1:
                # Modify configuration to reduce harmonic coupling
                # Toggle some channels to break harmonic relationship
                for i in range(2, self.n_channels, 3):  # Every third channel
                    if i < len(optimized_config):
                        optimized_config[i] = not optimized_config[i]
        
        return optimized_config
    
    def estimate_isolation(
        self,
        config1: List[bool],
        config2: List[bool],
        freq1: float,
        freq2: float
    ) -> float:
        """
        Estimate isolation between two band configurations.
        
        Args:
            config1: Channel states for first band
            config2: Channel states for second band
            freq1: First band frequency
            freq2: Second band frequency
            
        Returns:
            Estimated isolation in dB
        """
        # Simplified isolation model
        # Real implementation would use electromagnetic simulation
        
        # Calculate overlap between configurations
        overlap = sum(c1 and c2 for c1, c2 in zip(config1, config2))
        total_active = sum(config1) + sum(config2)
        
        if total_active == 0:
            return float('inf')  # Perfect isolation
        
        overlap_ratio = overlap / total_active
        
        # Frequency separation effect
        freq_ratio = abs(freq1 - freq2) / min(freq1, freq2)
        freq_isolation = 20 * np.log10(1 + freq_ratio)
        
        # Spatial separation effect
        spatial_isolation = 30 * (1 - overlap_ratio)
        
        total_isolation = freq_isolation + spatial_isolation
        
        return max(total_isolation, 10.0)  # Minimum 10 dB isolation
    
    def analyze_multiband_performance(self) -> Dict[str, Any]:
        """
        Analyze performance across all configured bands.
        
        Returns:
            Performance analysis results
        """
        if not self.band_configurations:
            return {'error': 'No band configurations defined'}
        
        analysis = {
            'bands': {},
            'isolation_matrix': {},
            'overall_performance': {}
        }
        
        band_names = list(self.band_configurations.keys())
        
        # Analyze each band
        for band_name, config in self.band_configurations.items():
            self.switch_to_band(band_name)
            
            # Estimate performance metrics
            estimated_gain = self._estimate_band_gain(config)
            estimated_vswr = self._estimate_band_vswr(config)
            estimated_efficiency = self._estimate_band_efficiency(config)
            
            analysis['bands'][band_name] = {
                'frequency_ghz': config.frequency / 1e9,
                'bandwidth_mhz': config.bandwidth / 1e6,
                'estimated_gain_dbi': estimated_gain,
                'estimated_vswr': estimated_vswr,
                'estimated_efficiency': estimated_efficiency,
                'meets_gain_target': estimated_gain >= config.gain_target,
                'meets_vswr_requirement': estimated_vswr <= config.vswr_max,
                'meets_efficiency_requirement': estimated_efficiency >= config.efficiency_min
            }
        
        # Calculate isolation matrix
        for i, band1 in enumerate(band_names):
            for j, band2 in enumerate(band_names):
                if i != j:
                    config1 = self.band_configurations[band1]
                    config2 = self.band_configurations[band2]
                    
                    isolation = self.estimate_isolation(
                        config1.channel_states,
                        config2.channel_states,
                        config1.frequency,
                        config2.frequency
                    )
                    
                    analysis['isolation_matrix'][f'{band1}_{band2}'] = {
                        'isolation_db': isolation,
                        'meets_requirement': isolation >= self.isolation_requirement
                    }
        
        # Overall performance summary
        all_bands_data = list(analysis['bands'].values())
        min_isolation = min(
            item['isolation_db'] for item in analysis['isolation_matrix'].values()
        ) if analysis['isolation_matrix'] else float('inf')
        
        analysis['overall_performance'] = {
            'total_bands': len(band_names),
            'avg_gain_dbi': np.mean([b['estimated_gain_dbi'] for b in all_bands_data]),
            'avg_vswr': np.mean([b['estimated_vswr'] for b in all_bands_data]),
            'avg_efficiency': np.mean([b['estimated_efficiency'] for b in all_bands_data]),
            'min_isolation_db': min_isolation,
            'all_requirements_met': all(
                b['meets_gain_target'] and b['meets_vswr_requirement'] and b['meets_efficiency_requirement']
                for b in all_bands_data
            ) and min_isolation >= self.isolation_requirement
        }
        
        return analysis
    
    def _estimate_band_gain(self, config: BandConfiguration) -> float:
        """Estimate antenna gain for band configuration."""
        # Simple gain estimation model
        base_gain = 6.0  # dBi for basic patch
        
        # Channel effects
        active_channels = sum(config.channel_states)
        channel_effect = 0.5 * (active_channels / self.n_channels - 0.5)  # -0.25 to +0.25 dB
        
        # Frequency effects
        optimal_freq = self.get_resonant_frequency()
        freq_ratio = config.frequency / optimal_freq
        freq_effect = -2.0 * abs(1 - freq_ratio)  # Loss when away from optimal frequency
        
        estimated_gain = base_gain + channel_effect + freq_effect
        return max(estimated_gain, 0.0)
    
    def _estimate_band_vswr(self, config: BandConfiguration) -> float:
        """Estimate VSWR for band configuration."""
        # Simple VSWR estimation model
        base_vswr = 1.5
        
        # Channel configuration effects
        active_channels = sum(config.channel_states)
        if active_channels == 0:
            return 10.0  # Very poor match with no channels
        
        # Frequency effects
        optimal_freq = self.get_resonant_frequency()
        freq_ratio = config.frequency / optimal_freq
        freq_mismatch = abs(1 - freq_ratio)
        
        vswr_degradation = 1.0 + 3.0 * freq_mismatch
        estimated_vswr = base_vswr * vswr_degradation
        
        return min(estimated_vswr, 10.0)
    
    def _estimate_band_efficiency(self, config: BandConfiguration) -> float:
        """Estimate radiation efficiency for band configuration."""
        # Simple efficiency estimation model
        base_efficiency = 0.85
        
        # Loss due to liquid metal resistance
        active_channels = sum(config.channel_states)
        conductor_loss = 0.05 * (active_channels / self.n_channels)
        
        # Frequency-dependent losses
        freq_loss = 0.01 * (config.frequency / 1e9) ** 0.5
        
        estimated_efficiency = base_efficiency - conductor_loss - freq_loss
        return max(estimated_efficiency, 0.3)  # Minimum 30% efficiency
    
    def export_multiband_config(self, filename: str) -> None:
        """Export multi-band configuration to file."""
        import json
        
        config_data = {
            'antenna_type': 'MultiBandPatchAntenna',
            'substrate_height': self.substrate_height,
            'dielectric_constant': self.dielectric_constant,
            'patch_dimensions': [self.patch_length, self.patch_width],
            'n_channels': self.n_channels,
            'channel_width': self.channel_width,
            'isolation_requirement': self.isolation_requirement,
            'band_configurations': {},
            'performance_analysis': self.analyze_multiband_performance()
        }
        
        # Export band configurations
        for band_name, config in self.band_configurations.items():
            config_data['band_configurations'][band_name] = {
                'frequency': config.frequency,
                'bandwidth': config.bandwidth,
                'channel_states': config.channel_states,
                'gain_target': config.gain_target,
                'vswr_max': config.vswr_max,
                'efficiency_min': config.efficiency_min
            }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Multi-band configuration exported to {filename}")
    
    def __repr__(self) -> str:
        n_bands = len(self.band_configurations)
        active_channels = sum(self.channel_states)
        
        return (
            f"MultiBandPatchAntenna("
            f"{n_bands} bands, "
            f"{active_channels}/{self.n_channels} channels active, "
            f"isolation≥{self.isolation_requirement:.0f}dB"
            f")"
        )


class BeamSteeringArray:
    """Advanced beam steering array with liquid metal phase shifters."""
    
    def __init__(
        self,
        n_elements: Tuple[int, int] = (8, 8),
        element_spacing: float = 0.5,
        element_type: str = 'patch',
        phase_resolution: int = 6,  # bits of phase resolution
        frequency_range: Tuple[float, float] = (2.0e9, 6.0e9)
    ):
        """
        Initialize beam steering array.
        
        Args:
            n_elements: Array dimensions (nx, ny)
            element_spacing: Element spacing in wavelengths
            element_type: Type of array elements
            phase_resolution: Phase shifter resolution in bits
            frequency_range: Operating frequency range
        """
        self.n_elements = n_elements
        self.element_spacing = element_spacing
        self.element_type = element_type
        self.phase_resolution = phase_resolution
        self.frequency_range = frequency_range
        
        self.logger = get_logger('beam_steering')
        
        # Phase quantization levels
        self.n_phase_levels = 2 ** phase_resolution
        self.phase_step = 2 * np.pi / self.n_phase_levels
        
        # Initialize phase shifter states
        nx, ny = n_elements
        self.phase_shifter_states = np.zeros((nx, ny), dtype=int)
        
        # Current beam parameters
        self.current_theta = 0.0  # Elevation in radians
        self.current_phi = 0.0    # Azimuth in radians
        self.current_frequency = sum(frequency_range) / 2  # Default center frequency
        
        # Array calibration data
        self.calibration_data = {}
        
        self.logger.info(f"Initialized beam steering array: {nx}x{ny} elements, "
                        f"{phase_resolution}-bit phase control")
    
    def set_beam_direction(
        self,
        theta: float,
        phi: float = 0.0,
        frequency: Optional[float] = None
    ) -> None:
        """
        Set beam steering direction with quantized phase shifters.
        
        Args:
            theta: Elevation angle in degrees
            phi: Azimuth angle in degrees
            frequency: Operating frequency in Hz (None for current)
        """
        if frequency is None:
            frequency = self.current_frequency
        
        # Validate inputs
        if not -90 <= theta <= 90:
            raise ValueError("Theta must be between -90 and 90 degrees")
        if not 0 <= phi < 360:
            raise ValueError("Phi must be between 0 and 360 degrees")
        
        # Convert to radians
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        
        self.current_theta = theta_rad
        self.current_phi = phi_rad
        self.current_frequency = frequency
        
        # Calculate required phase shifts
        c = 299792458
        wavelength = c / frequency
        k = 2 * np.pi / wavelength
        
        # Element spacing in meters
        dx = self.element_spacing * wavelength
        dy = self.element_spacing * wavelength
        
        nx, ny = self.n_elements
        
        # Calculate and quantize phase for each element
        for i in range(nx):
            for j in range(ny):
                # Element position relative to array center
                x_pos = (i - (nx - 1) / 2) * dx
                y_pos = (j - (ny - 1) / 2) * dy
                
                # Required phase shift for beam steering
                required_phase = k * (
                    x_pos * np.sin(theta_rad) * np.cos(phi_rad) +
                    y_pos * np.sin(theta_rad) * np.sin(phi_rad)
                )
                
                # Wrap phase to [0, 2π)
                wrapped_phase = required_phase % (2 * np.pi)
                
                # Quantize to available phase levels
                quantized_level = int(wrapped_phase / self.phase_step)
                quantized_level = min(quantized_level, self.n_phase_levels - 1)
                
                self.phase_shifter_states[i, j] = quantized_level
        
        self.logger.info(f"Beam steered to ({theta:.1f}°, {phi:.1f}°) @ {frequency/1e9:.2f}GHz")
    
    def get_actual_phase_distribution(self) -> np.ndarray:
        """Get actual quantized phase distribution."""
        return self.phase_shifter_states * self.phase_step
    
    def calculate_quantization_error(self) -> Dict[str, float]:
        """Calculate phase quantization error statistics."""
        # Calculate ideal phases
        ideal_phases = self._calculate_ideal_phases(
            self.current_theta, self.current_phi, self.current_frequency
        )
        
        # Calculate actual phases
        actual_phases = self.get_actual_phase_distribution()
        
        # Calculate errors
        phase_errors = np.abs(actual_phases - ideal_phases)
        phase_errors = np.minimum(phase_errors, 2*np.pi - phase_errors)  # Wrap errors
        
        return {
            'rms_error_degrees': np.sqrt(np.mean(phase_errors**2)) * 180/np.pi,
            'max_error_degrees': np.max(phase_errors) * 180/np.pi,
            'mean_error_degrees': np.mean(phase_errors) * 180/np.pi,
            'phase_resolution_degrees': self.phase_step * 180/np.pi
        }
    
    def _calculate_ideal_phases(
        self,
        theta: float,
        phi: float, 
        frequency: float
    ) -> np.ndarray:
        """Calculate ideal (unquantized) phase distribution."""
        c = 299792458
        wavelength = c / frequency
        k = 2 * np.pi / wavelength
        
        dx = self.element_spacing * wavelength
        dy = self.element_spacing * wavelength
        
        nx, ny = self.n_elements
        phases = np.zeros((nx, ny))
        
        for i in range(nx):
            for j in range(ny):
                x_pos = (i - (nx - 1) / 2) * dx
                y_pos = (j - (ny - 1) / 2) * dy
                
                phase = k * (
                    x_pos * np.sin(theta) * np.cos(phi) +
                    y_pos * np.sin(theta) * np.sin(phi)
                )
                
                phases[i, j] = phase % (2 * np.pi)
        
        return phases
    
    def scan_beam_continuously(
        self,
        theta_range: Tuple[float, float],
        phi_range: Tuple[float, float],
        n_steps: int = 36,
        dwell_time: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform continuous beam scanning over specified angular range.
        
        Args:
            theta_range: Elevation range (min_deg, max_deg)
            phi_range: Azimuth range (min_deg, max_deg)
            n_steps: Number of beam positions
            dwell_time: Dwell time per position in seconds
            
        Returns:
            List of scan results
        """
        import time
        
        theta_min, theta_max = theta_range
        phi_min, phi_max = phi_range
        
        # Generate scan pattern (spiral or raster)
        scan_positions = self._generate_scan_pattern(
            theta_range, phi_range, n_steps
        )
        
        scan_results = []
        
        for i, (theta, phi) in enumerate(scan_positions):
            start_time = time.time()
            
            # Steer beam to position
            self.set_beam_direction(theta, phi)
            
            # Calculate performance metrics
            quantization_error = self.calculate_quantization_error()
            directivity = self._estimate_directivity(theta, phi)
            
            # Simulate dwell time
            time.sleep(max(0, dwell_time - (time.time() - start_time)))
            
            result = {
                'step': i + 1,
                'theta_deg': theta,
                'phi_deg': phi,
                'quantization_error': quantization_error,
                'estimated_directivity_dbi': directivity,
                'scan_time': time.time() - start_time
            }
            
            scan_results.append(result)
            
            if i % 10 == 0:  # Log progress
                self.logger.info(f"Scan progress: {i+1}/{len(scan_positions)} positions")
        
        self.logger.info(f"Beam scan complete: {len(scan_results)} positions")
        return scan_results
    
    def _generate_scan_pattern(
        self,
        theta_range: Tuple[float, float],
        phi_range: Tuple[float, float],
        n_steps: int
    ) -> List[Tuple[float, float]]:
        """Generate scan pattern positions."""
        theta_min, theta_max = theta_range
        phi_min, phi_max = phi_range
        
        # Create spiral scan pattern
        positions = []
        
        for i in range(n_steps):
            # Spiral parameters
            r_norm = i / (n_steps - 1)  # Normalized radius
            angle = 2 * np.pi * i / 6  # 6 points per revolution
            
            # Map to angular ranges
            theta = theta_min + (theta_max - theta_min) * r_norm
            phi = phi_min + (phi_max - phi_min) * (0.5 + 0.5 * np.cos(angle))
            
            positions.append((theta, phi))
        
        return positions
    
    def _estimate_directivity(self, theta: float, phi: float) -> float:
        """Estimate array directivity at steering direction."""
        nx, ny = self.n_elements
        total_elements = nx * ny
        
        # Basic directivity estimation for uniform array
        # D = 4π * A_eff / λ² where A_eff is effective aperture area
        
        c = 299792458
        wavelength = c / self.current_frequency
        
        # Physical aperture area
        aperture_length = (nx - 1) * self.element_spacing * wavelength
        aperture_width = (ny - 1) * self.element_spacing * wavelength
        physical_area = aperture_length * aperture_width
        
        # Effective aperture (reduced by tapering and quantization errors)
        quantization_error = self.calculate_quantization_error()
        efficiency_factor = np.cos(np.radians(quantization_error['rms_error_degrees'] / 2))
        
        effective_area = physical_area * efficiency_factor
        
        # Directivity calculation
        directivity_linear = 4 * np.pi * effective_area / (wavelength ** 2)
        directivity_dbi = 10 * np.log10(directivity_linear)
        
        return directivity_dbi
    
    def calibrate_array(
        self,
        calibration_frequencies: List[float],
        reference_direction: Tuple[float, float] = (0.0, 0.0)
    ) -> Dict[str, Any]:
        """
        Perform array calibration to characterize phase shifter performance.
        
        Args:
            calibration_frequencies: List of calibration frequencies
            reference_direction: Reference beam direction (theta, phi) in degrees
            
        Returns:
            Calibration results
        """
        self.logger.info("Starting array calibration...")
        
        calibration_results = {
            'reference_direction': reference_direction,
            'frequencies_ghz': [f/1e9 for f in calibration_frequencies],
            'phase_accuracy': {},
            'frequency_response': {},
            'element_variations': {}
        }
        
        ref_theta, ref_phi = reference_direction
        
        for freq in calibration_frequencies:
            freq_key = f"{freq/1e9:.2f}GHz"
            
            # Set beam to reference direction
            self.set_beam_direction(ref_theta, ref_phi, freq)
            
            # Analyze phase accuracy
            quantization_error = self.calculate_quantization_error()
            calibration_results['phase_accuracy'][freq_key] = quantization_error
            
            # Estimate frequency response
            directivity = self._estimate_directivity(ref_theta, ref_phi)
            calibration_results['frequency_response'][freq_key] = {
                'directivity_dbi': directivity,
                'estimated_gain_dbi': directivity - 1.0  # Account for losses
            }
            
            # Check for element-to-element variations (simplified)
            phase_dist = self.get_actual_phase_distribution()
            phase_std = np.std(phase_dist)
            calibration_results['element_variations'][freq_key] = {
                'phase_std_degrees': phase_std * 180/np.pi,
                'uniformity_score': max(0, 100 - phase_std * 180/np.pi)
            }
        
        # Store calibration data
        self.calibration_data = calibration_results
        
        self.logger.info("Array calibration complete")
        return calibration_results
    
    def export_calibration_data(self, filename: str) -> None:
        """Export calibration data to file."""
        import json
        
        if not self.calibration_data:
            raise ValueError("No calibration data available. Run calibrate_array() first.")
        
        export_data = {
            'array_configuration': {
                'n_elements': self.n_elements,
                'element_spacing': self.element_spacing,
                'phase_resolution': self.phase_resolution,
                'frequency_range_ghz': [f/1e9 for f in self.frequency_range]
            },
            'calibration_timestamp': self.calibration_data.get('timestamp', 'unknown'),
            'calibration_results': self.calibration_data
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Calibration data exported to {filename}")
    
    def __repr__(self) -> str:
        nx, ny = self.n_elements
        current_theta_deg = np.degrees(self.current_theta)
        current_phi_deg = np.degrees(self.current_phi)
        
        return (
            f"BeamSteeringArray("
            f"{nx}x{ny} elements, "
            f"{self.phase_resolution}-bit phase, "
            f"beam@({current_theta_deg:.1f}°,{current_phi_deg:.1f}°)"
            f")"
        )