"""
Performance optimization and resource management.
"""

import os
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import queue

from ..utils.logging_config import get_logger
from ..utils.diagnostics import SystemDiagnostics, PerformanceMonitor


@dataclass
class ResourceLimits:
    """Resource usage limits for optimization."""
    
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_gpu_memory_percent: float = 90.0
    max_concurrent_tasks: int = None  # Will be set based on system
    max_threads_per_task: int = 4
    
    def __post_init__(self):
        """Set default values based on system."""
        if self.max_concurrent_tasks is None:
            self.max_concurrent_tasks = min(mp.cpu_count(), 8)


class ResourceManager:
    """Manages system resources for optimal performance."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize resource manager.
        
        Args:
            limits: Resource usage limits
        """
        self.limits = limits or ResourceLimits()
        self.diagnostics = SystemDiagnostics()
        self.performance_monitor = PerformanceMonitor(update_interval=1.0)
        
        self.logger = get_logger('resource_manager')
        
        # Resource tracking
        self.active_tasks = 0
        self.task_lock = threading.Lock()
        
        # GPU resource management
        self.gpu_available = self._check_gpu_availability()
        self.gpu_memory_pool = []
        
        # Thread pools
        self._cpu_pool = None
        self._io_pool = None
        
        self.logger.info(f"Resource manager initialized: CPU cores={mp.cpu_count()}, "
                        f"GPU available={self.gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available and functional."""
        try:
            import torch
            if torch.cuda.is_available():
                # Test basic GPU operation
                test_tensor = torch.ones(100, device='cuda')
                result = torch.sum(test_tensor).item()
                return abs(result - 100) < 1e-6
            return False
        except Exception:
            return False
    
    @contextmanager
    def acquire_resources(self, resource_type: str = 'cpu', memory_mb: float = 0):
        """
        Context manager for resource acquisition.
        
        Args:
            resource_type: Type of resources needed ('cpu', 'gpu', 'mixed')
            memory_mb: Expected memory usage in MB
        """
        acquired = False
        
        try:
            # Wait for resources to become available
            while not self._can_acquire_resources(resource_type, memory_mb):
                time.sleep(0.1)
            
            # Acquire resources
            with self.task_lock:
                self.active_tasks += 1
            
            acquired = True
            self.logger.debug(f"Acquired {resource_type} resources (active tasks: {self.active_tasks})")
            
            yield
            
        finally:
            if acquired:
                with self.task_lock:
                    self.active_tasks -= 1
                self.logger.debug(f"Released {resource_type} resources (active tasks: {self.active_tasks})")
    
    def _can_acquire_resources(self, resource_type: str, memory_mb: float) -> bool:
        """Check if resources can be acquired."""
        # Check task limit
        with self.task_lock:
            if self.active_tasks >= self.limits.max_concurrent_tasks:
                return False
        
        # Check system resources
        metrics = self.diagnostics.get_system_metrics()
        
        # CPU check
        if metrics.cpu_usage_percent > self.limits.max_cpu_percent:
            return False
        
        # Memory check
        memory_usage_percent = (metrics.memory_used_gb / metrics.memory_total_gb) * 100
        if memory_usage_percent > self.limits.max_memory_percent:
            return False
        
        # Additional memory check for requested amount
        available_memory_mb = metrics.memory_available_gb * 1024
        if memory_mb > available_memory_mb:
            return False
        
        # GPU-specific checks
        if resource_type in ['gpu', 'mixed'] and self.gpu_available:
            gpu_usage_percent = (metrics.gpu_memory_used_gb / metrics.gpu_memory_total_gb) * 100
            if gpu_usage_percent > self.limits.max_gpu_memory_percent:
                return False
        
        return True
    
    def get_optimal_thread_count(self, task_type: str = 'cpu_bound') -> int:
        """
        Get optimal thread count for task type.
        
        Args:
            task_type: Type of task ('cpu_bound', 'io_bound', 'mixed')
            
        Returns:
            Optimal thread count
        """
        cpu_count = mp.cpu_count()
        
        if task_type == 'cpu_bound':
            # For CPU-bound tasks, use number of physical cores
            return min(cpu_count, self.limits.max_threads_per_task)
        
        elif task_type == 'io_bound':
            # For I/O-bound tasks, can use more threads
            return min(cpu_count * 2, self.limits.max_threads_per_task * 2)
        
        else:  # mixed
            return min(int(cpu_count * 1.5), self.limits.max_threads_per_task)
    
    def get_cpu_pool(self) -> ThreadPoolExecutor:
        """Get CPU thread pool for computation."""
        if self._cpu_pool is None:
            max_workers = self.get_optimal_thread_count('cpu_bound')
            self._cpu_pool = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix='cpu_worker'
            )
        return self._cpu_pool
    
    def get_io_pool(self) -> ThreadPoolExecutor:
        """Get I/O thread pool for file operations."""
        if self._io_pool is None:
            max_workers = self.get_optimal_thread_count('io_bound')
            self._io_pool = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix='io_worker'
            )
        return self._io_pool
    
    def optimize_for_workload(self, workload_profile: Dict[str, Any]) -> None:
        """
        Optimize resource allocation for specific workload.
        
        Args:
            workload_profile: Workload characteristics
        """
        workload_type = workload_profile.get('type', 'mixed')
        expected_tasks = workload_profile.get('expected_tasks', 10)
        memory_per_task = workload_profile.get('memory_per_task_mb', 100)
        
        # Adjust limits based on workload
        if workload_type == 'compute_intensive':
            self.limits.max_cpu_percent = 95.0
            self.limits.max_memory_percent = 90.0
            self.limits.max_concurrent_tasks = min(mp.cpu_count(), 4)
            
        elif workload_type == 'memory_intensive':
            self.limits.max_cpu_percent = 70.0
            self.limits.max_memory_percent = 95.0
            
            # Calculate max tasks based on memory
            metrics = self.diagnostics.get_system_metrics()
            available_memory_mb = metrics.memory_available_gb * 1024
            max_memory_tasks = int(available_memory_mb / memory_per_task)
            
            self.limits.max_concurrent_tasks = min(max_memory_tasks, 8)
            
        elif workload_type == 'gpu_intensive':
            self.limits.max_gpu_memory_percent = 95.0
            self.limits.max_concurrent_tasks = min(2, mp.cpu_count())  # Limit for GPU tasks
        
        self.logger.info(f"Optimized for {workload_type} workload: "
                        f"max_tasks={self.limits.max_concurrent_tasks}")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        def log_metrics(metrics):
            if metrics.cpu_usage_percent > 90 or metrics.memory_usage_percent > 90:
                self.logger.warning(f"High resource usage: CPU={metrics.cpu_usage_percent:.1f}%, "
                                  f"Memory={metrics.memory_usage_percent:.1f}%")
        
        self.performance_monitor.add_callback(log_metrics)
        self.performance_monitor.start_monitoring()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.performance_monitor.stop_monitoring()
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        metrics = self.diagnostics.get_system_metrics()
        
        with self.task_lock:
            active_tasks = self.active_tasks
        
        return {
            'active_tasks': active_tasks,
            'max_concurrent_tasks': self.limits.max_concurrent_tasks,
            'cpu_usage_percent': metrics.cpu_usage_percent,
            'memory_usage_percent': (metrics.memory_used_gb / metrics.memory_total_gb) * 100,
            'memory_available_gb': metrics.memory_available_gb,
            'gpu_available': self.gpu_available,
            'gpu_memory_usage_percent': (metrics.gpu_memory_used_gb / metrics.gpu_memory_total_gb) * 100 if self.gpu_available else 0,
            'resource_limits': {
                'max_cpu_percent': self.limits.max_cpu_percent,
                'max_memory_percent': self.limits.max_memory_percent,
                'max_gpu_memory_percent': self.limits.max_gpu_memory_percent
            }
        }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_monitoring()
        
        if self._cpu_pool:
            self._cpu_pool.shutdown(wait=True)
            self._cpu_pool = None
        
        if self._io_pool:
            self._io_pool.shutdown(wait=True)
            self._io_pool = None


class PerformanceOptimizer:
    """Optimizes performance across the entire system."""
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize performance optimizer.
        
        Args:
            resource_manager: Resource manager instance
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.logger = get_logger('performance_optimizer')
        
        # Performance profiles
        self.profiles = {
            'balanced': {'cpu_weight': 0.5, 'memory_weight': 0.3, 'gpu_weight': 0.2},
            'cpu_optimized': {'cpu_weight': 0.8, 'memory_weight': 0.1, 'gpu_weight': 0.1},
            'memory_optimized': {'cpu_weight': 0.2, 'memory_weight': 0.7, 'gpu_weight': 0.1},
            'gpu_optimized': {'cpu_weight': 0.1, 'memory_weight': 0.1, 'gpu_weight': 0.8}
        }
        
        self.current_profile = 'balanced'
        
        # Performance metrics
        self.optimization_history = []
        
    def auto_optimize(self, workload_sample: List[Dict[str, Any]]) -> str:
        """
        Automatically optimize based on workload analysis.
        
        Args:
            workload_sample: Sample of recent tasks with performance data
            
        Returns:
            Selected optimization profile
        """
        if not workload_sample:
            return self.current_profile
        
        # Analyze workload characteristics
        total_cpu_time = sum(task.get('cpu_time', 0) for task in workload_sample)
        total_memory_usage = sum(task.get('memory_mb', 0) for task in workload_sample)
        total_gpu_time = sum(task.get('gpu_time', 0) for task in workload_sample)
        
        total_time = total_cpu_time + total_gpu_time
        
        # Determine optimal profile
        if total_time > 0:
            cpu_ratio = total_cpu_time / total_time
            gpu_ratio = total_gpu_time / total_time
            
            if gpu_ratio > 0.6:
                profile = 'gpu_optimized'
            elif cpu_ratio > 0.8:
                profile = 'cpu_optimized'
            elif total_memory_usage / len(workload_sample) > 1000:  # High memory per task
                profile = 'memory_optimized'
            else:
                profile = 'balanced'
        else:
            profile = 'balanced'
        
        if profile != self.current_profile:
            self.apply_profile(profile)
            self.logger.info(f"Auto-optimized to {profile} profile based on workload analysis")
        
        return profile
    
    def apply_profile(self, profile_name: str) -> None:
        """
        Apply performance optimization profile.
        
        Args:
            profile_name: Name of profile to apply
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        profile = self.profiles[profile_name]
        self.current_profile = profile_name
        
        # Apply profile-specific optimizations
        if profile_name == 'cpu_optimized':
            self.resource_manager.limits.max_cpu_percent = 95.0
            self.resource_manager.limits.max_concurrent_tasks = mp.cpu_count()
            self._optimize_cpu_settings()
            
        elif profile_name == 'memory_optimized':
            self.resource_manager.limits.max_memory_percent = 90.0
            self.resource_manager.limits.max_concurrent_tasks = max(1, mp.cpu_count() // 2)
            self._optimize_memory_settings()
            
        elif profile_name == 'gpu_optimized':
            self.resource_manager.limits.max_gpu_memory_percent = 95.0
            self.resource_manager.limits.max_concurrent_tasks = 2
            self._optimize_gpu_settings()
        
        else:  # balanced
            self._apply_balanced_settings()
        
        self.logger.info(f"Applied {profile_name} optimization profile")
    
    def _optimize_cpu_settings(self) -> None:
        """Apply CPU-specific optimizations."""
        # Set CPU affinity if available
        try:
            import psutil
            process = psutil.Process()
            
            # Use all available CPU cores
            cpu_count = mp.cpu_count()
            if hasattr(process, 'cpu_affinity'):
                process.cpu_affinity(list(range(cpu_count)))
        except Exception:
            pass
        
        # Set environment variables for numerical libraries
        os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
        os.environ['NUMEXPR_NUM_THREADS'] = str(mp.cpu_count())
    
    def _optimize_memory_settings(self) -> None:
        """Apply memory-specific optimizations."""
        # Conservative memory settings
        os.environ['OMP_NUM_THREADS'] = str(max(1, mp.cpu_count() // 2))
        
        # Enable memory optimization for PyTorch if available
        try:
            import torch
            torch.backends.cudnn.benchmark = False  # Save memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def _optimize_gpu_settings(self) -> None:
        """Apply GPU-specific optimizations."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Enable optimizations for GPU
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # Set memory growth
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
                
        except Exception:
            pass
    
    def _apply_balanced_settings(self) -> None:
        """Apply balanced optimization settings."""
        # Reset to moderate settings
        self.resource_manager.limits.max_cpu_percent = 80.0
        self.resource_manager.limits.max_memory_percent = 80.0
        self.resource_manager.limits.max_gpu_memory_percent = 90.0
        self.resource_manager.limits.max_concurrent_tasks = min(mp.cpu_count(), 8)
        
        # Moderate thread counts
        thread_count = max(1, mp.cpu_count() // 2)
        os.environ['OMP_NUM_THREADS'] = str(thread_count)
        os.environ['MKL_NUM_THREADS'] = str(thread_count)
    
    def benchmark_system(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Run system benchmark to characterize performance.
        
        Args:
            duration_seconds: Benchmark duration
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Running system benchmark for {duration_seconds} seconds...")
        
        start_time = time.time()
        results = {
            'duration_seconds': duration_seconds,
            'cpu_benchmark': self._benchmark_cpu(),
            'memory_benchmark': self._benchmark_memory(),
            'gpu_benchmark': self._benchmark_gpu() if self.resource_manager.gpu_available else None
        }
        
        actual_duration = time.time() - start_time
        results['actual_duration_seconds'] = actual_duration
        
        self.logger.info(f"Benchmark completed in {actual_duration:.1f} seconds")
        
        return results
    
    def _benchmark_cpu(self) -> Dict[str, float]:
        """Benchmark CPU performance."""
        import numpy as np
        
        # Matrix multiplication benchmark
        size = 1000
        start_time = time.time()
        
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
        
        cpu_time = time.time() - start_time
        flops = 2 * size ** 3  # Approximate FLOPS for matrix multiplication
        gflops = (flops / cpu_time) / 1e9
        
        return {
            'matrix_multiply_time_seconds': cpu_time,
            'estimated_gflops': gflops
        }
    
    def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory performance."""
        import numpy as np
        
        # Memory bandwidth test
        size = 100 * 1024 * 1024  # 100 MB
        
        start_time = time.time()
        data = np.random.bytes(size)
        copy_data = data[:]  # Memory copy
        memory_time = time.time() - start_time
        
        bandwidth_gbps = (size * 2 / memory_time) / 1e9  # Read + write
        
        return {
            'memory_copy_time_seconds': memory_time,
            'estimated_bandwidth_gbps': bandwidth_gbps
        }
    
    def _benchmark_gpu(self) -> Optional[Dict[str, float]]:
        """Benchmark GPU performance."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return None
            
            device = torch.device('cuda')
            size = 2048
            
            # GPU matrix multiplication
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warm up
            torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            flops = 2 * size ** 3
            gflops = (flops / gpu_time) / 1e9
            
            return {
                'gpu_matrix_multiply_time_seconds': gpu_time,
                'estimated_gpu_gflops': gflops,
                'gpu_name': torch.cuda.get_device_name(0)
            }
            
        except Exception as e:
            self.logger.warning(f"GPU benchmark failed: {str(e)}")
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        stats = self.resource_manager.get_resource_stats()
        
        return {
            'current_profile': self.current_profile,
            'resource_stats': stats,
            'optimization_profiles': list(self.profiles.keys()),
            'gpu_available': self.resource_manager.gpu_available,
            'system_capabilities': {
                'cpu_cores': mp.cpu_count(),
                'total_memory_gb': stats.get('memory_available_gb', 0) + 
                                 stats.get('memory_usage_percent', 0) / 100 * stats.get('memory_available_gb', 1)
            }
        }