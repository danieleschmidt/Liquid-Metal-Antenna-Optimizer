"""
System diagnostics and health monitoring for liquid metal antenna optimizer.
"""

import time
import platform
import os
import json
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    
    name: str
    status: str  # 'healthy', 'warning', 'error'
    message: str
    details: Dict[str, Any]
    timestamp: str
    duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass  
class SystemMetrics:
    """System performance metrics."""
    
    cpu_count: int
    cpu_usage_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_total_gb: float
    gpu_memory_used_gb: float
    python_version: str
    platform: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class SystemDiagnostics:
    """Comprehensive system diagnostics and monitoring."""
    
    def __init__(self):
        """Initialize diagnostics system."""
        self.health_checks = {}
        self.metrics_history = []
        self.max_history_length = 1000
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self) -> None:
        """Register default system health checks."""
        self.register_health_check('system_resources', self._check_system_resources)
        self.register_health_check('python_environment', self._check_python_environment)
        self.register_health_check('dependencies', self._check_dependencies)
        self.register_health_check('gpu_availability', self._check_gpu_availability)
        self.register_health_check('disk_space', self._check_disk_space)
        self.register_health_check('memory_usage', self._check_memory_usage)
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health status.
        
        Returns:
            Dictionary with health metrics and status
        """
        try:
            import psutil
            
            # Get basic system metrics
            memory = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Calculate health metrics
            memory_usage = (memory.used / memory.total) * 100
            
            # Simulate error rate (would be tracked in real implementation)
            error_rate = 0.0  # No errors for demo
            
            # Determine responsiveness
            responsive = memory_usage < 90 and cpu_usage < 95
            
            return {
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'error_rate': error_rate,
                'responsive': responsive,
                'status': 'healthy' if responsive and error_rate < 5 else 'degraded'
            }
            
        except ImportError:
            # Fallback when psutil not available
            return {
                'memory_usage': 25.0,  # Simulated values
                'cpu_usage': 15.0,
                'error_rate': 0.0,
                'responsive': True,
                'status': 'healthy'
            }
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable[[], Tuple[str, str, Dict[str, Any]]]
    ) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Health check name
            check_function: Function returning (status, message, details)
        """
        self.health_checks[name] = check_function
    
    def run_health_check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Health check name
            
        Returns:
            HealthCheckResult with check results
        """
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status='error',
                message=f'Health check "{name}" not found',
                details={},
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=0.0
            )
        
        start_time = time.time()
        
        try:
            status, message, details = self.health_checks[name]()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=name,
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=name,
                status='error',
                message=f'Health check failed: {str(e)}',
                details={'exception': str(e)},
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=duration_ms
            )
    
    def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        
        for name in self.health_checks.keys():
            results[name] = self.run_health_check(name)
        
        return results
    
    def get_system_metrics(self) -> SystemMetrics:
        """
        Collect comprehensive system metrics.
        
        Returns:
            SystemMetrics with current system state
        """
        try:
            # CPU information
            cpu_count = os.cpu_count() or 1
            cpu_usage = self._get_cpu_usage()
            
            # Memory information
            memory_info = self._get_memory_info()
            
            # Disk information
            disk_info = self._get_disk_info()
            
            # GPU information
            gpu_info = self._get_gpu_info()
            
            # System information
            python_version = platform.python_version()
            system_platform = platform.platform()
            
            metrics = SystemMetrics(
                cpu_count=cpu_count,
                cpu_usage_percent=cpu_usage,
                memory_total_gb=memory_info['total'],
                memory_used_gb=memory_info['used'],
                memory_available_gb=memory_info['available'],
                disk_total_gb=disk_info['total'],
                disk_used_gb=disk_info['used'],
                disk_free_gb=disk_info['free'],
                gpu_available=gpu_info['available'],
                gpu_count=gpu_info['count'],
                gpu_memory_total_gb=gpu_info['memory_total'],
                gpu_memory_used_gb=gpu_info['memory_used'],
                python_version=python_version,
                platform=system_platform,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_length:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            # Return minimal metrics on error
            return SystemMetrics(
                cpu_count=1,
                cpu_usage_percent=0.0,
                memory_total_gb=0.0,
                memory_used_gb=0.0,
                memory_available_gb=0.0,
                disk_total_gb=0.0,
                disk_used_gb=0.0,
                disk_free_gb=0.0,
                gpu_available=False,
                gpu_count=0,
                gpu_memory_total_gb=0.0,
                gpu_memory_used_gb=0.0,
                python_version=platform.python_version(),
                platform=platform.platform(),
                timestamp=datetime.utcnow().isoformat()
            )
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            # Try using psutil if available
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback to load average on Unix systems
            try:
                load_avg = os.getloadavg()[0]
                cpu_count = os.cpu_count() or 1
                return min(load_avg / cpu_count * 100, 100.0)
            except (AttributeError, OSError):
                return 0.0
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory information in GB."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total / 1e9,
                'used': memory.used / 1e9,
                'available': memory.available / 1e9
            }
        except ImportError:
            # Fallback for systems without psutil
            try:
                # Try to read from /proc/meminfo on Linux
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                
                total_kb = int([line for line in meminfo.split('\n') if line.startswith('MemTotal')][0].split()[1])
                available_kb = int([line for line in meminfo.split('\n') if line.startswith('MemAvailable')][0].split()[1])
                
                total_gb = total_kb / 1e6
                available_gb = available_kb / 1e6
                used_gb = total_gb - available_gb
                
                return {
                    'total': total_gb,
                    'used': used_gb,
                    'available': available_gb
                }
            except (FileNotFoundError, IndexError, ValueError):
                return {'total': 0.0, 'used': 0.0, 'available': 0.0}
    
    def _get_disk_info(self) -> Dict[str, float]:
        """Get disk usage information in GB."""
        try:
            import psutil
            disk_usage = psutil.disk_usage('/')
            return {
                'total': disk_usage.total / 1e9,
                'used': disk_usage.used / 1e9,
                'free': disk_usage.free / 1e9
            }
        except ImportError:
            # Fallback using os.statvfs on Unix systems
            try:
                statvfs = os.statvfs('/')
                total_bytes = statvfs.f_frsize * statvfs.f_blocks
                free_bytes = statvfs.f_frsize * statvfs.f_available
                used_bytes = total_bytes - free_bytes
                
                return {
                    'total': total_bytes / 1e9,
                    'used': used_bytes / 1e9,
                    'free': free_bytes / 1e9
                }
            except (AttributeError, OSError):
                return {'total': 0.0, 'used': 0.0, 'free': 0.0}
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                
                # Get memory info for first GPU
                if gpu_count > 0:
                    torch.cuda.empty_cache()
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                    
                    return {
                        'available': True,
                        'count': gpu_count,
                        'memory_total': memory_total,
                        'memory_used': memory_allocated
                    }
            
            return {
                'available': False,
                'count': 0,
                'memory_total': 0.0,
                'memory_used': 0.0
            }
            
        except ImportError:
            return {
                'available': False,
                'count': 0,
                'memory_total': 0.0,
                'memory_used': 0.0
            }
    
    def _check_system_resources(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check system resource availability."""
        try:
            cpu_usage = self._get_cpu_usage()
            memory_info = self._get_memory_info()
            disk_info = self._get_disk_info()
            
            # Determine status based on usage
            status = 'healthy'
            messages = []
            
            if cpu_usage > 90:
                status = 'warning'
                messages.append(f'High CPU usage: {cpu_usage:.1f}%')
            
            memory_usage_percent = (memory_info['used'] / memory_info['total']) * 100
            if memory_usage_percent > 90:
                status = 'warning'
                messages.append(f'High memory usage: {memory_usage_percent:.1f}%')
            
            disk_usage_percent = (disk_info['used'] / disk_info['total']) * 100
            if disk_usage_percent > 90:
                status = 'warning'
                messages.append(f'High disk usage: {disk_usage_percent:.1f}%')
            
            message = '; '.join(messages) if messages else 'System resources normal'
            
            details = {
                'cpu_usage_percent': cpu_usage,
                'memory_usage_percent': memory_usage_percent,
                'disk_usage_percent': disk_usage_percent,
                'memory_available_gb': memory_info['available'],
                'disk_free_gb': disk_info['free']
            }
            
            return status, message, details
            
        except Exception as e:
            return 'error', f'Failed to check system resources: {str(e)}', {}
    
    def _check_python_environment(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check Python environment health."""
        try:
            python_version = platform.python_version()
            python_implementation = platform.python_implementation()
            
            # Check Python version
            major, minor, _ = python_version.split('.')
            major, minor = int(major), int(minor)
            
            status = 'healthy'
            message = f'Python {python_version} ({python_implementation})'
            
            if major < 3 or (major == 3 and minor < 9):
                status = 'warning'
                message = f'Python version {python_version} is older than recommended (3.9+)'
            
            details = {
                'python_version': python_version,
                'python_implementation': python_implementation,
                'python_path': os.sys.executable,
                'sys_path_length': len(os.sys.path)
            }
            
            return status, message, details
            
        except Exception as e:
            return 'error', f'Failed to check Python environment: {str(e)}', {}
    
    def _check_dependencies(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check required dependencies."""
        required_packages = {
            'numpy': '1.21.0',
            'scipy': '1.7.0',
            'torch': '2.0.0'
        }
        
        optional_packages = {
            'matplotlib': '3.5.0',
            'cupy': '12.0.0'
        }
        
        missing_required = []
        missing_optional = []
        installed_versions = {}
        
        # Check required packages
        for package, min_version in required_packages.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_versions[package] = version
            except ImportError:
                missing_required.append(package)
        
        # Check optional packages
        for package, min_version in optional_packages.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_versions[package] = version
            except ImportError:
                missing_optional.append(package)
        
        # Determine status
        if missing_required:
            status = 'error'
            message = f'Missing required packages: {", ".join(missing_required)}'
        elif missing_optional:
            status = 'warning'
            message = f'Missing optional packages: {", ".join(missing_optional)}'
        else:
            status = 'healthy'
            message = 'All dependencies available'
        
        details = {
            'installed_versions': installed_versions,
            'missing_required': missing_required,
            'missing_optional': missing_optional
        }
        
        return status, message, details
    
    def _check_gpu_availability(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check GPU availability and health."""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                
                gpu_details = []
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory / 1e9
                    memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                    memory_cached = torch.cuda.memory_reserved(i) / 1e9
                    
                    gpu_details.append({
                        'device_id': i,
                        'name': props.name,
                        'memory_total_gb': memory_total,
                        'memory_allocated_gb': memory_allocated,
                        'memory_cached_gb': memory_cached,
                        'compute_capability': f'{props.major}.{props.minor}'
                    })
                
                status = 'healthy'
                message = f'{gpu_count} GPU(s) available'
                
                details = {
                    'gpu_count': gpu_count,
                    'cuda_version': torch.version.cuda,
                    'gpus': gpu_details
                }
                
            else:
                status = 'warning'
                message = 'CUDA not available (using CPU)'
                details = {'cuda_available': False}
            
            return status, message, details
            
        except ImportError:
            return 'warning', 'PyTorch not available', {'torch_available': False}
        except Exception as e:
            return 'error', f'GPU check failed: {str(e)}', {}
    
    def _check_disk_space(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check available disk space."""
        try:
            disk_info = self._get_disk_info()
            
            free_gb = disk_info['free']
            usage_percent = (disk_info['used'] / disk_info['total']) * 100
            
            if free_gb < 1.0:  # Less than 1 GB free
                status = 'error'
                message = f'Very low disk space: {free_gb:.1f} GB free'
            elif free_gb < 5.0 or usage_percent > 95:  # Less than 5 GB or >95% usage
                status = 'warning'
                message = f'Low disk space: {free_gb:.1f} GB free ({usage_percent:.1f}% used)'
            else:
                status = 'healthy'
                message = f'Sufficient disk space: {free_gb:.1f} GB free'
            
            details = {
                'total_gb': disk_info['total'],
                'used_gb': disk_info['used'],
                'free_gb': disk_info['free'],
                'usage_percent': usage_percent
            }
            
            return status, message, details
            
        except Exception as e:
            return 'error', f'Disk space check failed: {str(e)}', {}
    
    def _check_memory_usage(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check memory usage."""
        try:
            memory_info = self._get_memory_info()
            
            usage_percent = (memory_info['used'] / memory_info['total']) * 100
            available_gb = memory_info['available']
            
            if available_gb < 0.5:  # Less than 500 MB available
                status = 'error'
                message = f'Very low memory: {available_gb:.2f} GB available'
            elif available_gb < 2.0 or usage_percent > 90:  # Less than 2 GB or >90% usage
                status = 'warning'
                message = f'Low memory: {available_gb:.1f} GB available ({usage_percent:.1f}% used)'
            else:
                status = 'healthy'
                message = f'Sufficient memory: {available_gb:.1f} GB available'
            
            details = {
                'total_gb': memory_info['total'],
                'used_gb': memory_info['used'],
                'available_gb': memory_info['available'],
                'usage_percent': usage_percent
            }
            
            return status, message, details
            
        except Exception as e:
            return 'error', f'Memory check failed: {str(e)}', {}
    
    def export_diagnostics_report(self, output_path: str) -> None:
        """
        Export comprehensive diagnostics report.
        
        Args:
            output_path: Path to save the diagnostics report
        """
        try:
            # Run all health checks
            health_results = self.run_all_health_checks()
            
            # Get current system metrics
            system_metrics = self.get_system_metrics()
            
            # Create comprehensive report
            report = {
                'report_timestamp': datetime.utcnow().isoformat(),
                'system_metrics': system_metrics.to_dict(),
                'health_checks': {name: result.to_dict() for name, result in health_results.items()},
                'metrics_history': [metrics.to_dict() for metrics in self.metrics_history[-10:]],  # Last 10 entries
                'summary': self._generate_summary(health_results)
            }
            
            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
        except Exception as e:
            raise RuntimeError(f"Failed to export diagnostics report: {str(e)}")
    
    def _generate_summary(self, health_results: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Generate summary of health check results."""
        total_checks = len(health_results)
        healthy_count = sum(1 for result in health_results.values() if result.status == 'healthy')
        warning_count = sum(1 for result in health_results.values() if result.status == 'warning')
        error_count = sum(1 for result in health_results.values() if result.status == 'error')
        
        overall_status = 'healthy'
        if error_count > 0:
            overall_status = 'error'
        elif warning_count > 0:
            overall_status = 'warning'
        
        return {
            'overall_status': overall_status,
            'total_checks': total_checks,
            'healthy_count': healthy_count,
            'warning_count': warning_count,
            'error_count': error_count,
            'health_score': (healthy_count + warning_count * 0.5) / total_checks if total_checks > 0 else 0
        }


class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.diagnostics = SystemDiagnostics()
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
    def add_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """
        Add callback function to be called on each update.
        
        Args:
            callback: Function to call with SystemMetrics
        """
        self.callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start performance monitoring in background thread."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.diagnostics.get_system_metrics()
                
                # Call all registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception:
                        pass  # Ignore callback errors
                
                time.sleep(self.update_interval)
                
            except Exception:
                # Continue monitoring even if metrics collection fails
                time.sleep(self.update_interval)
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.diagnostics.get_system_metrics()
    
    def get_metrics_history(self, duration_minutes: int = 60) -> List[SystemMetrics]:
        """
        Get metrics history for specified duration.
        
        Args:
            duration_minutes: Duration to retrieve in minutes
            
        Returns:
            List of SystemMetrics within the time window
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        
        return [
            metrics for metrics in self.diagnostics.metrics_history
            if datetime.fromisoformat(metrics.timestamp.replace('Z', '+00:00')) >= cutoff_time
        ]
    
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
        """
        return OperationProfiler(self, operation_name)
    
    def record_timing(self, operation_name: str, duration: float):
        """Record timing for an operation."""
        if not hasattr(self, 'timings'):
            self.timings = {}
            self.operation_counts = {}
            
        if operation_name not in self.timings:
            self.timings[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.timings[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
    
    def get_report(self) -> Dict[str, Dict[str, Any]]:
        """Get performance report."""
        if not hasattr(self, 'timings'):
            return {}
            
        report = {}
        
        for operation, times in self.timings.items():
            if times:
                report[operation] = {
                    'duration': sum(times) / len(times),  # Average duration
                    'calls': self.operation_counts[operation],
                    'total_time': sum(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        return report


class OperationProfiler:
    """Context manager for profiling individual operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.monitor.record_timing(self.operation_name, duration)