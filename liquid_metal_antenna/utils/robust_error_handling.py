"""
Generation 2 Enhancement: Robust Error Handling System
======================================================

Comprehensive error handling, logging, monitoring, and recovery mechanisms
for production-grade liquid metal antenna optimization.

Features:
- Circuit breaker pattern for fault tolerance
- Comprehensive logging with structured data
- Health monitoring and diagnostics
- Automatic error recovery
- Security validation and sanitization
- Performance monitoring and alerting
"""

import logging
import traceback
import time
import hashlib
import json
import os
import sys
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
import threading
from collections import defaultdict, deque
import asyncio


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemHealth(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: float
    error_id: str
    component: str
    operation: str
    user_input: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'timestamp': self.timestamp,
            'error_id': self.error_id,
            'component': self.component,
            'operation': self.operation,
            'user_input': self.user_input,
            'system_state': self.system_state,
            'stack_trace': self.stack_trace,
            'severity': self.severity.value
        }


class RobustErrorHandler:
    """
    Comprehensive error handling system with recovery mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize monitoring
        self.health_monitor = HealthMonitor()
        
        self.logger.info("RobustErrorHandler initialized")
    
    def _setup_logging(self):
        """Setup structured logging"""
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger("liquid_metal_antenna.errors")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for errors
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter with structured data
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def handle_error(
        self, 
        exception: Exception, 
        component: str, 
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> ErrorContext:
        """Handle an error with comprehensive logging and recovery"""
        
        # Generate unique error ID
        error_id = self._generate_error_id(exception, component, operation)
        
        # Create error context
        error_context = ErrorContext(
            timestamp=time.time(),
            error_id=error_id,
            component=component,
            operation=operation,
            user_input=context,
            system_state=self._capture_system_state(),
            stack_trace=traceback.format_exc(),
            severity=severity
        )
        
        # Store in history
        self.error_history.append(error_context)
        self.error_counts[f"{component}.{operation}"] += 1
        
        # Log the error
        self._log_error(error_context, exception)
        
        # Update health monitoring
        self.health_monitor.record_error(error_context)
        
        # Attempt recovery if strategy exists
        recovery_result = self._attempt_recovery(error_context, exception)
        
        # Check if circuit breaker should trigger
        self._check_circuit_breaker(component, operation)
        
        # Send alerts for critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._send_critical_alert(error_context)
        
        return error_context
    
    def _generate_error_id(self, exception: Exception, component: str, operation: str) -> str:
        """Generate unique error identifier"""
        
        error_string = f"{component}:{operation}:{type(exception).__name__}:{str(exception)}"
        return hashlib.md5(error_string.encode()).hexdigest()[:12]
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for debugging"""
        
        try:
            import psutil
            
            state = {
                'memory_usage': psutil.virtual_memory()._asdict(),
                'cpu_usage': psutil.cpu_percent(interval=1),
                'disk_usage': psutil.disk_usage('/')._asdict(),
                'python_version': sys.version,
                'timestamp': time.time()
            }
            
            # Add process-specific information
            process = psutil.Process()
            state['process_info'] = {
                'pid': process.pid,
                'memory_info': process.memory_info()._asdict(),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
            
            return state
            
        except ImportError:
            # Fallback without psutil
            return {
                'python_version': sys.version,
                'timestamp': time.time(),
                'note': 'Limited system info (psutil not available)'
            }
        except Exception as e:
            return {'error': f"Could not capture system state: {e}"}
    
    def _log_error(self, context: ErrorContext, exception: Exception):
        """Log error with structured data"""
        
        log_data = context.to_dict()
        log_data['exception_type'] = type(exception).__name__
        log_data['exception_message'] = str(exception)
        
        # Choose log level based on severity
        if context.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif context.severity == ErrorSeverity.HIGH:
            log_level = logging.ERROR
        elif context.severity == ErrorSeverity.MEDIUM:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        self.logger.log(
            log_level,
            f"Error in {context.component}.{context.operation}: {exception}",
            extra={'error_context': log_data}
        )
    
    def _attempt_recovery(self, context: ErrorContext, exception: Exception) -> Dict[str, Any]:
        """Attempt to recover from error using registered strategies"""
        
        recovery_key = f"{context.component}.{context.operation}"
        
        if recovery_key in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[recovery_key]
                result = recovery_func(context, exception)
                
                self.logger.info(f"Recovery attempted for {recovery_key}: {result}")
                return {'success': True, 'result': result}
                
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {recovery_key}: {recovery_error}")
                return {'success': False, 'error': str(recovery_error)}
        
        return {'success': False, 'reason': 'No recovery strategy registered'}
    
    def _check_circuit_breaker(self, component: str, operation: str):
        """Check if circuit breaker should be triggered"""
        
        key = f"{component}.{operation}"
        current_count = self.error_counts[key]
        
        # Default threshold: 10 errors in 5 minutes
        threshold = self.config.get('circuit_breaker_threshold', 10)
        window = self.config.get('circuit_breaker_window', 300)  # 5 minutes
        
        if current_count >= threshold:
            if key not in self.circuit_breakers:
                self.circuit_breakers[key] = {
                    'triggered_at': time.time(),
                    'count': current_count
                }
                
                self.logger.critical(
                    f"Circuit breaker triggered for {key} "
                    f"({current_count} errors >= {threshold} threshold)"
                )
    
    def _send_critical_alert(self, context: ErrorContext):
        """Send alert for critical errors"""
        
        alert_message = {
            'alert_type': 'critical_error',
            'component': context.component,
            'operation': context.operation,
            'error_id': context.error_id,
            'timestamp': context.timestamp,
            'message': f"Critical error in {context.component}.{context.operation}"
        }
        
        # In production, this would send to monitoring system
        # For now, just log the alert
        self.logger.critical(f"CRITICAL ALERT: {json.dumps(alert_message)}")
    
    def register_recovery_strategy(
        self, 
        component: str, 
        operation: str, 
        recovery_func: Callable
    ):
        """Register a recovery strategy for specific component/operation"""
        
        key = f"{component}.{operation}"
        self.recovery_strategies[key] = recovery_func
        self.logger.info(f"Recovery strategy registered for {key}")
    
    def is_circuit_breaker_open(self, component: str, operation: str) -> bool:
        """Check if circuit breaker is open for component/operation"""
        
        key = f"{component}.{operation}"
        
        if key not in self.circuit_breakers:
            return False
        
        # Check if circuit breaker has been open too long
        breaker_info = self.circuit_breakers[key]
        elapsed = time.time() - breaker_info['triggered_at']
        timeout = self.config.get('circuit_breaker_timeout', 600)  # 10 minutes
        
        if elapsed > timeout:
            # Reset circuit breaker
            del self.circuit_breakers[key]
            self.error_counts[key] = 0
            self.logger.info(f"Circuit breaker reset for {key}")
            return False
        
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics"""
        
        total_errors = sum(self.error_counts.values())
        recent_errors = len([e for e in self.error_history if time.time() - e.timestamp < 3600])
        
        stats = {
            'total_errors': total_errors,
            'recent_errors_1h': recent_errors,
            'error_counts_by_component': dict(self.error_counts),
            'active_circuit_breakers': list(self.circuit_breakers.keys()),
            'system_health': self.health_monitor.get_health_status().value,
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
        
        return stats


class HealthMonitor:
    """System health monitoring and diagnostics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.error_rate_window = deque(maxlen=100)
        self.performance_metrics = deque(maxlen=1000)
        self.health_checks = {}
        
    def record_error(self, error_context: ErrorContext):
        """Record error for health monitoring"""
        
        self.error_rate_window.append({
            'timestamp': error_context.timestamp,
            'severity': error_context.severity,
            'component': error_context.component
        })
    
    def record_performance_metric(self, operation: str, duration: float, success: bool):
        """Record performance metric"""
        
        self.performance_metrics.append({
            'timestamp': time.time(),
            'operation': operation,
            'duration': duration,
            'success': success
        })
    
    def get_health_status(self) -> SystemHealth:
        """Get overall system health status"""
        
        current_time = time.time()
        
        # Count recent errors (last 5 minutes)
        recent_errors = [
            e for e in self.error_rate_window 
            if current_time - e['timestamp'] < 300
        ]
        
        critical_errors = [e for e in recent_errors if e['severity'] == ErrorSeverity.CRITICAL]
        high_errors = [e for e in recent_errors if e['severity'] == ErrorSeverity.HIGH]
        
        # Health assessment
        if len(critical_errors) > 0:
            return SystemHealth.CRITICAL
        elif len(high_errors) > 5:
            return SystemHealth.UNHEALTHY
        elif len(recent_errors) > 20:
            return SystemHealth.DEGRADED
        else:
            return SystemHealth.HEALTHY
    
    def run_health_check(self, check_name: str, check_func: Callable) -> Dict[str, Any]:
        """Run a specific health check"""
        
        start_time = time.time()
        
        try:
            result = check_func()
            duration = time.time() - start_time
            
            health_result = {
                'check_name': check_name,
                'status': 'pass',
                'duration': duration,
                'result': result,
                'timestamp': time.time()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            
            health_result = {
                'check_name': check_name,
                'status': 'fail',
                'duration': duration,
                'error': str(e),
                'timestamp': time.time()
            }
        
        self.health_checks[check_name] = health_result
        return health_result


class SecurityValidator:
    """Security validation and input sanitization"""
    
    def __init__(self):
        self.blocked_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'__import__\s*\('
        ]
    
    def validate_input(self, input_data: Any, context: str = "unknown") -> Dict[str, Any]:
        """Validate and sanitize input data"""
        
        validation_result = {
            'valid': True,
            'sanitized_data': input_data,
            'security_issues': [],
            'context': context
        }
        
        if isinstance(input_data, str):
            # Check for malicious patterns
            import re
            
            for pattern in self.blocked_patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    validation_result['valid'] = False
                    validation_result['security_issues'].append(f"Blocked pattern detected: {pattern}")
            
            # Basic sanitization
            sanitized = input_data.replace('<', '&lt;').replace('>', '&gt;')
            validation_result['sanitized_data'] = sanitized
        
        elif isinstance(input_data, dict):
            # Recursively validate dictionary values
            sanitized_dict = {}
            
            for key, value in input_data.items():
                key_result = self.validate_input(str(key), f"{context}.key")
                value_result = self.validate_input(value, f"{context}.{key}")
                
                if not key_result['valid'] or not value_result['valid']:
                    validation_result['valid'] = False
                    validation_result['security_issues'].extend(
                        key_result['security_issues'] + value_result['security_issues']
                    )
                
                sanitized_dict[key_result['sanitized_data']] = value_result['sanitized_data']
            
            validation_result['sanitized_data'] = sanitized_dict
        
        return validation_result


def robust_operation(
    component: str, 
    operation: str, 
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    timeout: Optional[float] = None,
    retry_count: int = 0
):
    """
    Decorator for robust error handling around operations
    """
    
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            # Get or create error handler
            if not hasattr(wrapper, '_error_handler'):
                wrapper._error_handler = RobustErrorHandler()
            
            error_handler = wrapper._error_handler
            
            # Check circuit breaker
            if error_handler.is_circuit_breaker_open(component, operation):
                raise RuntimeError(
                    f"Circuit breaker open for {component}.{operation}"
                )
            
            # Security validation for kwargs
            security_validator = SecurityValidator()
            
            for key, value in kwargs.items():
                validation = security_validator.validate_input(value, f"{operation}.{key}")
                if not validation['valid']:
                    raise ValueError(
                        f"Security validation failed for {key}: {validation['security_issues']}"
                    )
                kwargs[key] = validation['sanitized_data']
            
            # Execution with retry logic
            for attempt in range(retry_count + 1):
                start_time = time.time()
                
                try:
                    # Execute with timeout if specified
                    if timeout:
                        # Simple timeout implementation
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Operation {operation} timed out after {timeout}s")
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout))
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Record successful operation
                    duration = time.time() - start_time
                    error_handler.health_monitor.record_performance_metric(
                        operation, duration, True
                    )
                    
                    if timeout:
                        signal.alarm(0)  # Cancel timeout
                    
                    return result
                
                except Exception as e:
                    if timeout:
                        signal.alarm(0)  # Cancel timeout
                    
                    # Record failed operation
                    duration = time.time() - start_time
                    error_handler.health_monitor.record_performance_metric(
                        operation, duration, False
                    )
                    
                    # Handle the error
                    context = {
                        'args': str(args)[:200],  # Limit context size
                        'kwargs': str(kwargs)[:200],
                        'attempt': attempt + 1,
                        'max_attempts': retry_count + 1
                    }
                    
                    error_context = error_handler.handle_error(
                        e, component, operation, context, severity
                    )
                    
                    # Retry if attempts remaining
                    if attempt < retry_count:
                        wait_time = 2 ** attempt  # Exponential backoff
                        time.sleep(wait_time)
                        continue
                    
                    # Re-raise if all retries exhausted
                    raise
            
        return wrapper
    return decorator


# Example recovery strategies
def solver_recovery_strategy(error_context: ErrorContext, exception: Exception) -> Dict[str, Any]:
    """Recovery strategy for solver errors"""
    
    if "memory" in str(exception).lower():
        # Memory error - try with reduced precision
        return {
            'recovery_action': 'reduce_precision',
            'recommendation': 'Use float32 instead of float64'
        }
    elif "cuda" in str(exception).lower():
        # CUDA error - fallback to CPU
        return {
            'recovery_action': 'fallback_to_cpu',
            'recommendation': 'Continue with CPU computation'
        }
    else:
        return {
            'recovery_action': 'restart_solver',
            'recommendation': 'Reinitialize solver with default parameters'
        }


def optimization_recovery_strategy(error_context: ErrorContext, exception: Exception) -> Dict[str, Any]:
    """Recovery strategy for optimization errors"""
    
    if "convergence" in str(exception).lower():
        return {
            'recovery_action': 'increase_iterations',
            'recommendation': 'Increase maximum iterations and reduce tolerance'
        }
    else:
        return {
            'recovery_action': 'restart_optimization',
            'recommendation': 'Restart with different initial conditions'
        }


# Global error handler instance
_global_error_handler = None

def get_error_handler() -> RobustErrorHandler:
    """Get global error handler instance"""
    
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
        
        # Register default recovery strategies
        _global_error_handler.register_recovery_strategy(
            'solver', 'simulate', solver_recovery_strategy
        )
        _global_error_handler.register_recovery_strategy(
            'optimizer', 'optimize', optimization_recovery_strategy
        )
    
    return _global_error_handler


# Health check functions
def check_memory_usage() -> Dict[str, Any]:
    """Check system memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        return {
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'status': 'ok' if memory.percent < 85 else 'warning'
        }
    except ImportError:
        return {'status': 'skipped', 'reason': 'psutil not available'}


def check_disk_space() -> Dict[str, Any]:
    """Check available disk space"""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        
        free_gb = disk.free / (1024**3)
        percent_used = (disk.used / disk.total) * 100
        
        return {
            'disk_free_gb': free_gb,
            'disk_percent_used': percent_used,
            'status': 'ok' if percent_used < 90 else 'warning'
        }
    except ImportError:
        return {'status': 'skipped', 'reason': 'psutil not available'}


def check_python_environment() -> Dict[str, Any]:
    """Check Python environment health"""
    
    return {
        'python_version': sys.version,
        'platform': sys.platform,
        'executable': sys.executable,
        'status': 'ok'
    }


# Example usage demonstration
if __name__ == "__main__":
    
    # Initialize error handler
    error_handler = get_error_handler()
    
    # Example of robust operation
    @robust_operation(
        component="demo", 
        operation="test_function",
        severity=ErrorSeverity.MEDIUM,
        retry_count=2
    )
    def test_function(value: int, name: str = "default"):
        """Test function that might fail"""
        
        if value < 0:
            raise ValueError(f"Negative value not allowed: {value}")
        
        if value > 100:
            raise RuntimeError(f"Value too large: {value}")
        
        return f"Success: {name} = {value}"
    
    # Test the robust operation
    print("Testing robust error handling...")
    
    try:
        result = test_function(50, name="test")
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    try:
        result = test_function(-5)  # Should trigger error handling
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Failed after error handling: {e}")
    
    # Run health checks
    print("\nRunning health checks...")
    health_monitor = error_handler.health_monitor
    
    checks = [
        ('memory', check_memory_usage),
        ('disk', check_disk_space),
        ('python', check_python_environment)
    ]
    
    for check_name, check_func in checks:
        result = health_monitor.run_health_check(check_name, check_func)
        print(f"Health check {check_name}: {result['status']}")
    
    # Display statistics
    stats = error_handler.get_error_statistics()
    print(f"\nError statistics: {stats}")
    
    print("\n✅ Generation 2 robust error handling demonstration complete")