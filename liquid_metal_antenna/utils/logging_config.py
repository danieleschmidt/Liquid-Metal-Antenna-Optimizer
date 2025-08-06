"""
Comprehensive logging configuration for liquid metal antenna optimizer.
"""

import logging
import logging.handlers
import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_id': record.thread if hasattr(record, 'thread') else None
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry['extra'] = log_entry.get('extra', {})
                log_entry['extra'][key] = value
        
        return json.dumps(log_entry, default=str)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.utcnow()
        
    def filter(self, record: logging.LogRecord) -> bool:
        # Add runtime information
        current_time = datetime.utcnow()
        record.runtime_seconds = (current_time - self.start_time).total_seconds()
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
        except (ImportError, Exception):
            record.memory_mb = None
            record.cpu_percent = None
        
        return True


class AntennaOptimizerLogger:
    """Centralized logger for antenna optimizer with multiple handlers."""
    
    def __init__(
        self,
        name: str = 'liquid_metal_antenna',
        log_dir: Optional[Union[str, Path]] = None,
        console_level: str = 'INFO',
        file_level: str = 'DEBUG',
        structured_output: bool = False,
        max_file_size_mb: int = 10,
        backup_count: int = 5
    ):
        """
        Initialize logger with multiple handlers.
        
        Args:
            name: Logger name
            log_dir: Directory for log files (None = no file logging)
            console_level: Console logging level
            file_level: File logging level  
            structured_output: Use structured JSON format
            max_file_size_mb: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        self.structured_output = structured_output
        self.performance_filter = PerformanceFilter()
        
        # Setup console handler
        self._setup_console_handler(console_level)
        
        # Setup file handlers if directory provided
        if log_dir:
            self._setup_file_handlers(log_dir, file_level, max_file_size_mb, backup_count)
        
        # Add performance filter to all handlers
        for handler in self.logger.handlers:
            handler.addFilter(self.performance_filter)
    
    def _setup_console_handler(self, level: str) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if self.structured_output:
            console_handler.setFormatter(StructuredFormatter())
        else:
            # Human-readable format for console
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(
        self,
        log_dir: Union[str, Path],
        level: str,
        max_file_size_mb: int,
        backup_count: int
    ) -> None:
        """Setup file logging handlers."""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # General log file with rotation
        general_log_file = log_path / 'antenna_optimizer.log'
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_path / 'antenna_optimizer_errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
        
        # Performance log file
        perf_log_file = log_path / 'antenna_optimizer_performance.log'
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())
        
        # Custom filter for performance logs
        class PerformanceLogFilter(logging.Filter):
            def filter(self, record):
                return 'performance' in record.getMessage().lower() or hasattr(record, 'performance_data')
        
        perf_handler.addFilter(PerformanceLogFilter())
        self.logger.addHandler(perf_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


# Global logger instance
_global_logger = None


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    console_level: str = 'INFO',
    file_level: str = 'DEBUG',
    structured_output: bool = False,
    **kwargs
) -> None:
    """
    Setup global logging configuration.
    
    Args:
        log_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level
        structured_output: Use structured JSON format
        **kwargs: Additional arguments for AntennaOptimizerLogger
    """
    global _global_logger
    
    _global_logger = AntennaOptimizerLogger(
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        structured_output=structured_output,
        **kwargs
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (None for root antenna optimizer logger)
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        setup_logging()
    
    if name is None:
        return _global_logger.get_logger()
    else:
        # Create child logger
        return logging.getLogger(f'liquid_metal_antenna.{name}')


def log_performance(
    operation: str,
    duration_seconds: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log performance metrics for operations.
    
    Args:
        operation: Name of operation
        duration_seconds: Operation duration
        additional_metrics: Additional performance metrics
        logger: Logger instance (None for default)
    """
    if logger is None:
        logger = get_logger('performance')
    
    metrics = {
        'operation': operation,
        'duration_seconds': duration_seconds,
        'performance_data': True
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    logger.info(f"Performance: {operation} completed in {duration_seconds:.3f}s", extra=metrics)


def log_optimization_progress(
    iteration: int,
    objective_value: float,
    constraint_violation: float,
    additional_data: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log optimization progress.
    
    Args:
        iteration: Current iteration
        objective_value: Current objective value
        constraint_violation: Current constraint violation
        additional_data: Additional optimization data
        logger: Logger instance (None for default)
    """
    if logger is None:
        logger = get_logger('optimization')
    
    data = {
        'iteration': iteration,
        'objective_value': objective_value,
        'constraint_violation': constraint_violation,
        'optimization_data': True
    }
    
    if additional_data:
        data.update(additional_data)
    
    logger.info(
        f"Optimization iteration {iteration}: objective={objective_value:.6f}, "
        f"violation={constraint_violation:.6f}",
        extra=data
    )


def log_simulation_metrics(
    frequency: float,
    gain_dbi: float,
    vswr: float,
    efficiency: float,
    computation_time: float,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log electromagnetic simulation metrics.
    
    Args:
        frequency: Simulation frequency in Hz
        gain_dbi: Antenna gain in dBi
        vswr: Voltage standing wave ratio
        efficiency: Radiation efficiency (0-1)
        computation_time: Simulation time in seconds
        logger: Logger instance (None for default)
    """
    if logger is None:
        logger = get_logger('simulation')
    
    data = {
        'frequency_ghz': frequency / 1e9,
        'gain_dbi': gain_dbi,
        'vswr': vswr,
        'efficiency': efficiency,
        'computation_time': computation_time,
        'simulation_data': True
    }
    
    logger.info(
        f"Simulation @ {frequency/1e9:.2f}GHz: gain={gain_dbi:.1f}dBi, "
        f"VSWR={vswr:.2f}, eff={efficiency:.1%}, time={computation_time:.2f}s",
        extra=data
    )


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    operation: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log error with detailed context information.
    
    Args:
        error: Exception that occurred
        context: Context information
        operation: Operation that was being performed
        logger: Logger instance (None for default)
    """
    if logger is None:
        logger = get_logger('error')
    
    error_data = {
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'error_data': True
    }
    
    logger.error(
        f"Error in {operation}: {type(error).__name__}: {str(error)}",
        exc_info=True,
        extra=error_data
    )


class LoggingContextManager:
    """Context manager for operation logging with automatic timing."""
    
    def __init__(
        self,
        operation: str,
        logger: Optional[logging.Logger] = None,
        log_start: bool = True,
        log_end: bool = True,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize logging context manager.
        
        Args:
            operation: Operation name
            logger: Logger instance
            log_start: Log operation start
            log_end: Log operation completion
            extra_data: Additional data to log
        """
        self.operation = operation
        self.logger = logger or get_logger()
        self.log_start = log_start
        self.log_end = log_end
        self.extra_data = extra_data or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        
        if self.log_start:
            self.logger.info(f"Starting {self.operation}", extra=self.extra_data)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is not None:
            # Log error
            error_data = self.extra_data.copy()
            error_data.update({
                'duration_seconds': duration,
                'operation': self.operation
            })
            
            log_error_with_context(exc_val, error_data, self.operation, self.logger)
        elif self.log_end:
            # Log successful completion
            completion_data = self.extra_data.copy()
            completion_data.update({
                'duration_seconds': duration,
                'operation': self.operation,
                'performance_data': True
            })
            
            self.logger.info(
                f"Completed {self.operation} in {duration:.3f}s",
                extra=completion_data
            )


# Convenient decorator for automatic operation logging
def logged_operation(
    operation: str = None,
    logger: Optional[logging.Logger] = None,
    log_performance: bool = True
):
    """
    Decorator for automatic operation logging.
    
    Args:
        operation: Operation name (defaults to function name)
        logger: Logger instance
        log_performance: Whether to log performance metrics
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            
            with LoggingContextManager(
                operation=op_name,
                logger=logger,
                log_end=log_performance
            ):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator