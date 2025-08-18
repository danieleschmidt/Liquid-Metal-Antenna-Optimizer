"""
Comprehensive validation utilities for liquid metal antenna optimizer.
"""

import re
import os
from typing import Union, Tuple, Any, Dict, List, Optional
import warnings


class ValidationError(Exception):
    """Custom validation error with detailed context."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, suggestions: Optional[List[str]] = None):
        self.context = context or {}
        self.suggestions = suggestions or []
        
        # Build detailed message
        detailed_message = message
        if self.context:
            detailed_message += f"\nContext: {self.context}"
        if self.suggestions:
            detailed_message += f"\nSuggestions: {'; '.join(self.suggestions)}"
        
        super().__init__(detailed_message)


def validate_frequency_range(
    frequency_range: Union[Tuple[float, float], object],
    min_freq: float = 1e6,      # 1 MHz minimum
    max_freq: float = 100e9,    # 100 GHz maximum
    min_bandwidth: float = 1e3  # 1 kHz minimum bandwidth
) -> None:
    """
    Validate frequency range with comprehensive checks.
    
    Args:
        frequency_range: Frequency range to validate
        min_freq: Minimum allowed frequency in Hz
        max_freq: Maximum allowed frequency in Hz
        min_bandwidth: Minimum required bandwidth in Hz
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Extract start and stop frequencies
        if hasattr(frequency_range, 'start') and hasattr(frequency_range, 'stop'):
            start_freq = frequency_range.start
            stop_freq = frequency_range.stop
        elif isinstance(frequency_range, (tuple, list)) and len(frequency_range) == 2:
            start_freq, stop_freq = frequency_range
        else:
            raise ValidationError(
                "Invalid frequency range format",
                context={'provided': str(type(frequency_range))},
                suggestions=["Use tuple (start_hz, stop_hz) or FrequencyRange object"]
            )
        
        # Type validation
        if not isinstance(start_freq, (int, float)) or not isinstance(stop_freq, (int, float)):
            raise ValidationError(
                "Frequency values must be numeric",
                context={'start_type': type(start_freq), 'stop_type': type(stop_freq)},
                suggestions=["Ensure frequencies are int or float values in Hz"]
            )
        
        # Range validation
        if start_freq <= 0 or stop_freq <= 0:
            raise ValidationError(
                "Frequencies must be positive",
                context={'start_freq': start_freq, 'stop_freq': stop_freq},
                suggestions=["Use positive frequency values in Hz"]
            )
        
        if start_freq >= stop_freq:
            raise ValidationError(
                "Start frequency must be less than stop frequency",
                context={'start_freq': start_freq, 'stop_freq': stop_freq},
                suggestions=["Ensure start_freq < stop_freq"]
            )
        
        # Bandwidth validation
        bandwidth = stop_freq - start_freq
        if bandwidth < min_bandwidth:
            raise ValidationError(
                f"Bandwidth too small: {bandwidth/1e3:.1f} kHz < {min_bandwidth/1e3:.1f} kHz minimum",
                context={'bandwidth': bandwidth, 'min_bandwidth': min_bandwidth},
                suggestions=["Increase frequency range or reduce minimum bandwidth requirement"]
            )
        
        # Physical limits
        if start_freq < min_freq:
            raise ValidationError(
                f"Start frequency too low: {start_freq/1e6:.1f} MHz < {min_freq/1e6:.1f} MHz minimum",
                context={'start_freq': start_freq, 'min_freq': min_freq},
                suggestions=["Use frequencies above 1 MHz for antenna applications"]
            )
        
        if stop_freq > max_freq:
            raise ValidationError(
                f"Stop frequency too high: {stop_freq/1e9:.1f} GHz > {max_freq/1e9:.1f} GHz maximum",
                context={'stop_freq': stop_freq, 'max_freq': max_freq},
                suggestions=["Use frequencies below 100 GHz for current implementation"]
            )
        
        # Warn about unusual frequency ranges
        if start_freq < 100e6:  # Below 100 MHz
            warnings.warn(f"Low frequency {start_freq/1e6:.1f} MHz may require large antenna structures")
        
        if stop_freq > 30e9:  # Above 30 GHz
            warnings.warn(f"High frequency {stop_freq/1e9:.1f} GHz may require very fine simulation resolution")
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Unexpected error during frequency validation: {str(e)}",
            context={'original_error': str(e)},
            suggestions=["Check input format and values"]
        )


def validate_geometry(
    geometry: Any,
    min_size: Tuple[int, int, int] = (8, 8, 4),
    max_size: Tuple[int, int, int] = (512, 512, 256),
    max_memory_gb: float = 16.0
) -> None:
    """
    Validate antenna geometry with memory and size checks.
    
    Args:
        geometry: Geometry tensor or array to validate
        min_size: Minimum allowed dimensions
        max_size: Maximum allowed dimensions
        max_memory_gb: Maximum memory usage limit
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Import here to avoid circular dependencies
        try:
            import torch
            import numpy as np
            has_torch = True
        except ImportError:
            import numpy as np
            has_torch = False
        
        # Convert to numpy for consistent handling
        if has_torch and isinstance(geometry, torch.Tensor):
            geometry_array = geometry.detach().cpu().numpy()
        elif isinstance(geometry, np.ndarray):
            geometry_array = geometry
        else:
            raise ValidationError(
                "Geometry must be numpy array or torch tensor",
                context={'provided_type': type(geometry)},
                suggestions=["Convert geometry to numpy.ndarray or torch.Tensor"]
            )
        
        # Dimension validation
        if geometry_array.ndim != 3:
            raise ValidationError(
                f"Geometry must be 3-dimensional, got {geometry_array.ndim}D",
                context={'shape': geometry_array.shape},
                suggestions=["Ensure geometry has shape (nx, ny, nz)"]
            )
        
        # Size validation
        shape = geometry_array.shape
        for i, (current, minimum, maximum) in enumerate(zip(shape, min_size, max_size)):
            axis_name = ['x', 'y', 'z'][i]
            
            if current < minimum:
                raise ValidationError(
                    f"Geometry too small in {axis_name}-dimension: {current} < {minimum}",
                    context={'shape': shape, 'min_size': min_size},
                    suggestions=[f"Increase {axis_name}-dimension to at least {minimum}"]
                )
            
            if current > maximum:
                raise ValidationError(
                    f"Geometry too large in {axis_name}-dimension: {current} > {maximum}",
                    context={'shape': shape, 'max_size': max_size},
                    suggestions=[f"Reduce {axis_name}-dimension to at most {maximum}"]
                )
        
        # Memory usage estimation
        element_size = 4  # Assume float32
        total_elements = np.prod(shape)
        field_components = 6  # Ex, Ey, Ez, Hx, Hy, Hz
        estimated_memory_gb = (total_elements * field_components * element_size) / 1e9
        
        if estimated_memory_gb > max_memory_gb:
            raise ValidationError(
                f"Estimated memory usage {estimated_memory_gb:.1f} GB exceeds limit {max_memory_gb:.1f} GB",
                context={'geometry_shape': shape, 'estimated_memory_gb': estimated_memory_gb},
                suggestions=[
                    "Reduce geometry dimensions",
                    "Increase memory limit",
                    "Use lower precision (float16)"
                ]
            )
        
        # Value validation
        min_val = np.min(geometry_array)
        max_val = np.max(geometry_array)
        
        if min_val < 0:
            warnings.warn(f"Geometry contains negative values (min: {min_val:.3f})")
        
        if max_val > 1:
            warnings.warn(f"Geometry contains values > 1 (max: {max_val:.3f})")
        
        # Check for valid conductor distribution
        conductor_fraction = np.mean(geometry_array > 0.5)
        if conductor_fraction < 0.001:  # Less than 0.1% conductor
            warnings.warn(f"Very sparse conductor distribution: {conductor_fraction:.1%}")
        elif conductor_fraction > 0.5:  # More than 50% conductor
            warnings.warn(f"Very dense conductor distribution: {conductor_fraction:.1%}")
        
        # Check for NaN or infinity
        if not np.isfinite(geometry_array).all():
            raise ValidationError(
                "Geometry contains NaN or infinity values",
                context={'finite_count': np.isfinite(geometry_array).sum(), 'total_count': geometry_array.size},
                suggestions=["Remove NaN/infinity values from geometry"]
            )
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Unexpected error during geometry validation: {str(e)}",
            context={'original_error': str(e)},
            suggestions=["Check geometry format and values"]
        )


def validate_material_properties(
    dielectric_constant: float,
    loss_tangent: float,
    thickness: float
) -> None:
    """
    Validate material property values.
    
    Args:
        dielectric_constant: Relative permittivity
        loss_tangent: Dielectric loss tangent
        thickness: Substrate thickness in mm
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Dielectric constant validation
        if not isinstance(dielectric_constant, (int, float)):
            raise ValidationError(
                "Dielectric constant must be numeric",
                context={'type': type(dielectric_constant)},
                suggestions=["Use float or int value"]
            )
        
        if dielectric_constant < 1.0:
            raise ValidationError(
                f"Dielectric constant {dielectric_constant:.2f} < 1.0 (unphysical)",
                context={'dielectric_constant': dielectric_constant},
                suggestions=["Use εᵣ ≥ 1.0 for passive materials"]
            )
        
        if dielectric_constant > 100:
            raise ValidationError(
                f"Dielectric constant {dielectric_constant:.2f} > 100 (unusual for antenna substrates)",
                context={'dielectric_constant': dielectric_constant},
                suggestions=["Common antenna substrates have εᵣ = 2-12"]
            )
        
        # Loss tangent validation
        if not isinstance(loss_tangent, (int, float)):
            raise ValidationError(
                "Loss tangent must be numeric",
                context={'type': type(loss_tangent)},
                suggestions=["Use float or int value"]
            )
        
        if loss_tangent < 0:
            raise ValidationError(
                f"Loss tangent {loss_tangent:.4f} < 0 (unphysical)",
                context={'loss_tangent': loss_tangent},
                suggestions=["Use tan δ ≥ 0 for passive materials"]
            )
        
        if loss_tangent > 1.0:
            raise ValidationError(
                f"Loss tangent {loss_tangent:.4f} > 1.0 (extremely lossy)",
                context={'loss_tangent': loss_tangent},
                suggestions=["Common antenna substrates have tan δ < 0.1"]
            )
        
        # Thickness validation
        if not isinstance(thickness, (int, float)):
            raise ValidationError(
                "Thickness must be numeric",
                context={'type': type(thickness)},
                suggestions=["Use float or int value in mm"]
            )
        
        if thickness <= 0:
            raise ValidationError(
                f"Thickness {thickness:.3f} mm ≤ 0 (unphysical)",
                context={'thickness': thickness},
                suggestions=["Use positive thickness value"]
            )
        
        if thickness < 0.1:
            warnings.warn(f"Very thin substrate: {thickness:.3f} mm")
        elif thickness > 10.0:
            warnings.warn(f"Very thick substrate: {thickness:.3f} mm")
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Unexpected error during material validation: {str(e)}",
            context={'original_error': str(e)},
            suggestions=["Check material property values"]
        )


def validate_optimization_parameters(
    n_iterations: int,
    learning_rate: float,
    tolerance: float
) -> None:
    """
    Validate optimization parameters.
    
    Args:
        n_iterations: Maximum number of iterations
        learning_rate: Optimization learning rate
        tolerance: Convergence tolerance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Iterations validation
        if not isinstance(n_iterations, int):
            raise ValidationError(
                "Number of iterations must be integer",
                context={'type': type(n_iterations)},
                suggestions=["Use integer value for n_iterations"]
            )
        
        if n_iterations <= 0:
            raise ValidationError(
                f"Number of iterations {n_iterations} ≤ 0",
                context={'n_iterations': n_iterations},
                suggestions=["Use positive number of iterations"]
            )
        
        if n_iterations > 100000:
            warnings.warn(f"Very large iteration count: {n_iterations}")
        
        # Learning rate validation
        if not isinstance(learning_rate, (int, float)):
            raise ValidationError(
                "Learning rate must be numeric",
                context={'type': type(learning_rate)},
                suggestions=["Use float or int value"]
            )
        
        if learning_rate <= 0:
            raise ValidationError(
                f"Learning rate {learning_rate:.6f} ≤ 0",
                context={'learning_rate': learning_rate},
                suggestions=["Use positive learning rate"]
            )
        
        if learning_rate > 1.0:
            warnings.warn(f"Large learning rate may cause instability: {learning_rate:.3f}")
        elif learning_rate < 1e-6:
            warnings.warn(f"Very small learning rate may converge slowly: {learning_rate:.6f}")
        
        # Tolerance validation
        if not isinstance(tolerance, (int, float)):
            raise ValidationError(
                "Tolerance must be numeric",
                context={'type': type(tolerance)},
                suggestions=["Use float or int value"]
            )
        
        if tolerance <= 0:
            raise ValidationError(
                f"Tolerance {tolerance:.9f} ≤ 0",
                context={'tolerance': tolerance},
                suggestions=["Use positive tolerance value"]
            )
        
        if tolerance > 1.0:
            warnings.warn(f"Large tolerance may prevent convergence: {tolerance:.3f}")
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Unexpected error during optimization parameter validation: {str(e)}",
            context={'original_error': str(e)},
            suggestions=["Check optimization parameter values"]
        )


def validate_file_path(
    file_path: str,
    must_exist: bool = False,
    allowed_extensions: Optional[List[str]] = None,
    max_path_length: int = 255
) -> None:
    """
    Validate file path with security checks.
    
    Args:
        file_path: File path to validate
        must_exist: Whether file must already exist
        allowed_extensions: List of allowed file extensions
        max_path_length: Maximum path length
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if not isinstance(file_path, str):
            raise ValidationError(
                "File path must be string",
                context={'type': type(file_path)},
                suggestions=["Provide string file path"]
            )
        
        if len(file_path) == 0:
            raise ValidationError(
                "File path is empty",
                suggestions=["Provide non-empty file path"]
            )
        
        if len(file_path) > max_path_length:
            raise ValidationError(
                f"File path too long: {len(file_path)} > {max_path_length}",
                context={'path_length': len(file_path), 'max_length': max_path_length},
                suggestions=["Use shorter file path"]
            )
        
        # Security checks
        dangerous_patterns = ['..', '~', '$', '|', ';', '&', '`']
        for pattern in dangerous_patterns:
            if pattern in file_path:
                raise ValidationError(
                    f"File path contains dangerous pattern: '{pattern}'",
                    context={'file_path': file_path, 'pattern': pattern},
                    suggestions=["Use safe file path without special characters"]
                )
        
        # Extension validation
        if allowed_extensions:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in allowed_extensions:
                raise ValidationError(
                    f"File extension '{file_ext}' not allowed",
                    context={'file_extension': file_ext, 'allowed': allowed_extensions},
                    suggestions=[f"Use one of: {', '.join(allowed_extensions)}"]
                )
        
        # Existence check
        if must_exist and not os.path.exists(file_path):
            raise ValidationError(
                f"File does not exist: {file_path}",
                context={'file_path': file_path},
                suggestions=["Check file path and ensure file exists"]
            )
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Unexpected error during file path validation: {str(e)}",
            context={'original_error': str(e)},
            suggestions=["Check file path format"]
        )


def validate_device_string(device: str) -> None:
    """
    Validate computation device string.
    
    Args:
        device: Device string ('cpu', 'cuda', 'cuda:0', etc.)
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if not isinstance(device, str):
            raise ValidationError(
                "Device must be string",
                context={'type': type(device)},
                suggestions=["Use string like 'cpu', 'cuda', or 'cuda:0'"]
            )
        
        device_lower = device.lower().strip()
        
        valid_patterns = [
            r'^cpu$',
            r'^cuda$',
            r'^cuda:[0-9]+$',
        ]
        
        is_valid = any(re.match(pattern, device_lower) for pattern in valid_patterns)
        
        if not is_valid:
            raise ValidationError(
                f"Invalid device string: '{device}'",
                context={'device': device},
                suggestions=["Use 'cpu', 'cuda', or 'cuda:N' where N is device number"]
            )
        
        # Check CUDA availability if CUDA device requested
        if 'cuda' in device_lower:
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.warn(f"CUDA device '{device}' requested but CUDA not available")
            except ImportError:
                warnings.warn(f"CUDA device '{device}' requested but PyTorch not available")
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Unexpected error during device validation: {str(e)}",
            context={'original_error': str(e)},
            suggestions=["Check device string format"]
        )


def comprehensive_system_validation() -> Dict[str, Any]:
    """
    Comprehensive system validation for Generation 2 robustness.
    
    Returns:
        Dict containing validation results for different system components
    """
    results = {}
    
    # Core imports validation
    try:
        import liquid_metal_antenna
        from liquid_metal_antenna import AntennaSpec, LMAOptimizer
        results['core_imports'] = {
            'status': 'PASSED',
            'message': 'Core imports successful'
        }
    except Exception as e:
        results['core_imports'] = {
            'status': 'FAILED',
            'message': f'Core import error: {str(e)}'
        }
    
    # Dependencies validation
    try:
        import numpy as np
        import scipy
        numpy_version = np.__version__
        scipy_version = scipy.__version__
        
        results['dependencies'] = {
            'status': 'PASSED',
            'message': f'NumPy {numpy_version}, SciPy {scipy_version}',
            'numpy_version': numpy_version,
            'scipy_version': scipy_version
        }
    except Exception as e:
        results['dependencies'] = {
            'status': 'FAILED',
            'message': f'Dependency error: {str(e)}'
        }
    
    # Memory validation
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1e9
        memory_available_gb = memory.available / 1e9
        
        results['memory'] = {
            'status': 'PASSED' if memory_available_gb > 1.0 else 'WARNING',
            'message': f'{memory_available_gb:.1f}GB available of {memory_gb:.1f}GB total',
            'total_gb': memory_gb,
            'available_gb': memory_available_gb
        }
    except ImportError:
        results['memory'] = {
            'status': 'WARNING',
            'message': 'psutil not available for memory check'
        }
    except Exception as e:
        results['memory'] = {
            'status': 'FAILED',
            'message': f'Memory check error: {str(e)}'
        }
    
    # GPU validation
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else 'Unknown'
            results['gpu'] = {
                'status': 'AVAILABLE',
                'message': f'{gpu_count} GPU(s) available: {gpu_name}',
                'count': gpu_count,
                'name': gpu_name
            }
        else:
            results['gpu'] = {
                'status': 'UNAVAILABLE',
                'message': 'CUDA not available, using CPU',
                'count': 0
            }
    except ImportError:
        results['gpu'] = {
            'status': 'UNAVAILABLE',
            'message': 'PyTorch not available',
            'count': 0
        }
    except Exception as e:
        results['gpu'] = {
            'status': 'ERROR',
            'message': f'GPU check error: {str(e)}'
        }
    
    # File system validation
    try:
        import tempfile
        import os
        
        # Test write permissions
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(b'test')
            tmp.flush()
            
        results['filesystem'] = {
            'status': 'PASSED',
            'message': 'File system access OK'
        }
    except Exception as e:
        results['filesystem'] = {
            'status': 'FAILED',
            'message': f'File system error: {str(e)}'
        }
    
    # Overall status
    failed_checks = sum(1 for r in results.values() if r['status'] in ['FAILED', 'ERROR'])
    warning_checks = sum(1 for r in results.values() if r['status'] == 'WARNING')
    
    if failed_checks == 0:
        overall_status = 'HEALTHY' if warning_checks == 0 else 'HEALTHY_WITH_WARNINGS'
    else:
        overall_status = 'DEGRADED' if failed_checks < len(results) / 2 else 'CRITICAL'
    
    results['overall'] = {
        'status': overall_status,
        'failed_checks': failed_checks,
        'warning_checks': warning_checks,
        'total_checks': len(results) - 1  # Exclude 'overall' itself
    }
    
    return results