"""
Advanced Performance Optimization Framework.

This module implements cutting-edge performance optimization techniques for
large-scale liquid-metal antenna design and optimization:

Features:
- GPU-accelerated computation with CUDA/OpenCL
- Distributed computing with MPI and cluster management  
- Memory optimization with intelligent caching
- Parallel execution with thread pools and async operations
- Load balancing and resource management
- Performance profiling and bottleneck analysis
- Auto-scaling based on workload
- Real-time monitoring and alerting
"""

import os
import sys
import time
import threading
import multiprocessing
import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import json
import pickle
import psutil
import queue
import weakref
import gc
from functools import lru_cache, wraps
from contextlib import contextmanager

import numpy as np

from ..utils.logging_config import get_logger
from ..utils.error_handling import handle_errors, robust_operation


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    execution_time: float = 0.0
    throughput: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ResourceAllocation:
    """Resource allocation configuration."""
    
    cpu_cores: int = 1
    memory_mb: int = 1024
    gpu_devices: List[int] = field(default_factory=list)
    thread_pool_size: int = 4
    process_pool_size: int = 2
    priority: str = "normal"  # low, normal, high, critical
    max_execution_time: float = 3600.0  # 1 hour


class GPUAccelerator:
    """GPU acceleration manager for computational operations."""
    
    def __init__(self, device_preference: str = "auto"):
        """
        Initialize GPU accelerator.
        
        Args:
            device_preference: "cuda", "opencl", "auto", or "cpu"
        """
        self.logger = get_logger('gpu_accelerator')
        self.device_preference = device_preference
        self.available_devices = []
        self.current_device = None
        
        # Initialize GPU support
        self._initialize_gpu_support()
    
    def _initialize_gpu_support(self) -> None:
        """Initialize GPU support based on available libraries."""
        
        # Try CUDA support
        if self.device_preference in ["cuda", "auto"]:
            try:
                import cupy as cp
                self.cupy = cp
                self.cuda_available = True
                
                # Get CUDA device info
                device_count = cp.cuda.runtime.getDeviceCount()
                for i in range(device_count):
                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                    self.available_devices.append({
                        'type': 'cuda',
                        'id': i,
                        'name': device_props['name'].decode(),
                        'memory': device_props['totalGlobalMem'],
                        'compute_capability': (device_props['major'], device_props['minor'])
                    })
                
                if self.available_devices:
                    self.current_device = self.available_devices[0]
                    self.logger.info(f"CUDA initialized with {device_count} devices")
                
            except ImportError:
                self.cuda_available = False
                self.logger.info("CUDA not available")
        
        # Try OpenCL support
        if self.device_preference in ["opencl", "auto"] and not self.available_devices:
            try:
                import pyopencl as cl
                self.opencl = cl
                self.opencl_available = True
                
                # Get OpenCL platforms and devices
                for platform in cl.get_platforms():
                    for device in platform.get_devices():
                        self.available_devices.append({
                            'type': 'opencl',
                            'platform': platform.name,
                            'name': device.name,
                            'memory': device.global_mem_size,
                            'compute_units': device.max_compute_units
                        })
                
                if self.available_devices:
                    self.current_device = self.available_devices[0]
                    self.logger.info(f"OpenCL initialized with {len(self.available_devices)} devices")
                
            except ImportError:
                self.opencl_available = False
                self.logger.info("OpenCL not available")
        
        # Fallback to CPU
        if not self.available_devices:
            self.current_device = {
                'type': 'cpu',
                'name': 'CPU',
                'cores': multiprocessing.cpu_count(),
                'memory': psutil.virtual_memory().total
            }
            self.logger.info("Using CPU computation (no GPU available)")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information."""
        return {
            'current_device': self.current_device,
            'available_devices': self.available_devices,
            'device_preference': self.device_preference
        }
    
    @contextmanager
    def gpu_context(self, device_id: Optional[int] = None):
        """Context manager for GPU operations."""
        if self.current_device['type'] == 'cuda' and hasattr(self, 'cupy'):
            original_device = self.cupy.cuda.Device()
            try:
                if device_id is not None:
                    self.cupy.cuda.Device(device_id).use()
                yield
            finally:
                original_device.use()
        else:
            yield
    
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """Transfer array to GPU memory."""
        if self.current_device['type'] == 'cuda' and hasattr(self, 'cupy'):
            return self.cupy.asarray(array)
        else:
            return array
    
    def to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """Transfer array back to CPU memory."""
        if self.current_device['type'] == 'cuda' and hasattr(self, 'cupy'):
            if hasattr(array, 'get'):
                return array.get()
        
        if isinstance(array, np.ndarray):
            return array
        else:
            return np.asarray(array)
    
    def accelerated_operation(self, operation: str, *args, **kwargs) -> Any:
        """Perform accelerated operation on GPU if available."""
        
        if self.current_device['type'] == 'cuda' and hasattr(self, 'cupy'):
            return self._cuda_operation(operation, *args, **kwargs)
        elif self.current_device['type'] == 'opencl' and hasattr(self, 'opencl'):
            return self._opencl_operation(operation, *args, **kwargs)
        else:
            return self._cpu_operation(operation, *args, **kwargs)
    
    def _cuda_operation(self, operation: str, *args, **kwargs) -> Any:
        """Perform CUDA-accelerated operation."""
        cp = self.cupy
        
        if operation == "matrix_multiply":
            a, b = args
            a_gpu = self.to_gpu(a)
            b_gpu = self.to_gpu(b)
            result_gpu = cp.dot(a_gpu, b_gpu)
            return self.to_cpu(result_gpu)
        
        elif operation == "fft":
            data = args[0]
            data_gpu = self.to_gpu(data)
            result_gpu = cp.fft.fftn(data_gpu)
            return self.to_cpu(result_gpu)
        
        elif operation == "convolution":
            signal, kernel = args
            signal_gpu = self.to_gpu(signal)
            kernel_gpu = self.to_gpu(kernel)
            # Use FFT-based convolution for large arrays
            result_gpu = cp.convolve(signal_gpu, kernel_gpu, mode=kwargs.get('mode', 'full'))
            return self.to_cpu(result_gpu)
        
        elif operation == "gradient":
            data = args[0]
            data_gpu = self.to_gpu(data)
            gradients = []
            for axis in range(data.ndim):
                grad_gpu = cp.gradient(data_gpu, axis=axis)
                gradients.append(self.to_cpu(grad_gpu))
            return gradients
        
        else:
            self.logger.warning(f"Unknown CUDA operation: {operation}")
            return self._cpu_operation(operation, *args, **kwargs)
    
    def _opencl_operation(self, operation: str, *args, **kwargs) -> Any:
        """Perform OpenCL-accelerated operation."""
        # Simplified OpenCL operations
        # In practice, would implement custom OpenCL kernels
        self.logger.info(f"OpenCL operation: {operation}")
        return self._cpu_operation(operation, *args, **kwargs)
    
    def _cpu_operation(self, operation: str, *args, **kwargs) -> Any:
        """Perform CPU operation as fallback."""
        
        if operation == "matrix_multiply":
            a, b = args
            return np.dot(a, b)
        
        elif operation == "fft":
            data = args[0]
            return np.fft.fftn(data)
        
        elif operation == "convolution":
            signal, kernel = args
            return np.convolve(signal, kernel, mode=kwargs.get('mode', 'full'))
        
        elif operation == "gradient":
            data = args[0]
            return np.gradient(data)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class DistributedComputing:
    """Distributed computing manager for cluster operations."""
    
    def __init__(self, cluster_config: Optional[Dict[str, Any]] = None):
        """Initialize distributed computing."""
        self.logger = get_logger('distributed_computing')
        self.cluster_config = cluster_config or {}
        self.is_distributed = self._check_distributed_environment()
        
        if self.is_distributed:
            self._initialize_mpi()
        
        self.worker_pool = None
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def _check_distributed_environment(self) -> bool:
        """Check if running in distributed environment."""
        # Check for MPI environment
        if 'OMPI_COMM_WORLD_SIZE' in os.environ or 'PMI_SIZE' in os.environ:
            return True
        
        # Check for cluster configuration
        if self.cluster_config.get('nodes', []):
            return True
        
        return False
    
    def _initialize_mpi(self) -> None:
        """Initialize MPI if available."""
        try:
            from mpi4py import MPI
            self.mpi = MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.mpi_available = True
            
            self.logger.info(f"MPI initialized: rank {self.rank}/{self.size}")
            
        except ImportError:
            self.mpi_available = False
            self.logger.info("MPI not available")
    
    def distribute_computation(
        self,
        computation_func: Callable,
        data_chunks: List[Any],
        gather_results: bool = True
    ) -> List[Any]:
        """Distribute computation across cluster nodes."""
        
        if self.is_distributed and hasattr(self, 'mpi'):
            return self._mpi_distribute(computation_func, data_chunks, gather_results)
        else:
            return self._local_parallel_distribute(computation_func, data_chunks)
    
    def _mpi_distribute(
        self,
        computation_func: Callable,
        data_chunks: List[Any],
        gather_results: bool
    ) -> List[Any]:
        """Distribute computation using MPI."""
        comm = self.comm
        rank = self.rank
        size = self.size
        
        # Scatter data to workers
        if rank == 0:
            # Master process distributes data
            chunks_per_worker = len(data_chunks) // size
            scattered_data = []
            
            for i in range(size):
                start_idx = i * chunks_per_worker
                end_idx = start_idx + chunks_per_worker
                if i == size - 1:  # Last worker gets remaining chunks
                    end_idx = len(data_chunks)
                
                worker_data = data_chunks[start_idx:end_idx]
                scattered_data.append(worker_data)
        else:
            scattered_data = None
        
        # Scatter data to all processes
        local_data = comm.scatter(scattered_data, root=0)
        
        # Process local data
        local_results = []
        for chunk in local_data:
            try:
                result = computation_func(chunk)
                local_results.append(result)
            except Exception as e:
                self.logger.error(f"Worker {rank} computation failed: {e}")
                local_results.append(None)
        
        # Gather results
        if gather_results:
            all_results = comm.gather(local_results, root=0)
            
            if rank == 0:
                # Flatten results
                flattened_results = []
                for worker_results in all_results:
                    if worker_results:
                        flattened_results.extend(worker_results)
                return flattened_results
            else:
                return []
        else:
            return local_results
    
    def _local_parallel_distribute(
        self,
        computation_func: Callable,
        data_chunks: List[Any]
    ) -> List[Any]:
        """Distribute computation using local multiprocessing."""
        
        max_workers = min(multiprocessing.cpu_count(), len(data_chunks))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(computation_func, chunk): chunk 
                for chunk in data_chunks
            }
            
            results = []
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result(timeout=600)  # 10 minute timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Computation failed for chunk: {e}")
                    results.append(None)
            
            return results
    
    def async_computation(
        self,
        computation_func: Callable,
        data: Any,
        callback: Optional[Callable] = None
    ) -> concurrent.futures.Future:
        """Submit asynchronous computation."""
        
        if not self.worker_pool:
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        future = self.worker_pool.submit(computation_func, data)
        
        if callback:
            future.add_done_callback(callback)
        
        return future
    
    def batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10,
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Process items in batches with parallel execution."""
        
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)
        
        # Create batches
        batches = [
            items[i:i + batch_size] 
            for i in range(0, len(items), batch_size)
        ]
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_futures = []
            
            for batch in batches:
                future = executor.submit(self._process_batch, batch, process_func)
                batch_futures.append(future)
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(batch_futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
            
            return all_results
    
    def _process_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process a single batch of items."""
        results = []
        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Item processing failed: {e}")
                results.append(None)
        
        return results


class IntelligentCache:
    """Intelligent caching system with automatic memory management."""
    
    def __init__(
        self, 
        max_memory_mb: int = 1024,
        eviction_policy: str = "lru",
        compression: bool = True
    ):
        """Initialize intelligent cache."""
        self.logger = get_logger('intelligent_cache')
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.compression = compression
        
        # Cache storage
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._memory_usage: Dict[str, int] = {}
        self._total_memory = 0
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _get_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                # Use pickle to estimate size
                pickled = pickle.dumps(obj)
                return len(pickled)
        except Exception:
            # Fallback estimate
            return sys.getsizeof(obj)
    
    def _compress_object(self, obj: Any) -> bytes:
        """Compress object for storage."""
        if not self.compression:
            return pickle.dumps(obj)
        
        import gzip
        pickled = pickle.dumps(obj)
        compressed = gzip.compress(pickled)
        
        compression_ratio = len(compressed) / len(pickled)
        self.logger.debug(f"Compression ratio: {compression_ratio:.2f}")
        
        return compressed
    
    def _decompress_object(self, compressed_data: bytes) -> Any:
        """Decompress object from storage."""
        if not self.compression:
            return pickle.loads(compressed_data)
        
        import gzip
        pickled = gzip.decompress(compressed_data)
        return pickle.loads(pickled)
    
    def _generate_cache_key(self, func: Callable, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for function call."""
        import hashlib
        
        # Create key from function name and arguments
        key_data = {
            'func_name': func.__name__,
            'func_module': func.__module__,
            'args': args,
            'kwargs': kwargs
        }
        
        # Handle numpy arrays in arguments
        serializable_data = self._make_hashable(key_data)
        key_string = json.dumps(serializable_data, sort_keys=True)
        
        # Create hash
        hash_obj = hashlib.sha256(key_string.encode())
        return hash_obj.hexdigest()[:16]  # 16 character key
    
    def _make_hashable(self, obj: Any) -> Any:
        """Convert objects to hashable representations."""
        if isinstance(obj, np.ndarray):
            return {
                'type': 'ndarray',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'hash': hash(obj.data.tobytes())
            }
        elif isinstance(obj, dict):
            return {k: self._make_hashable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_hashable(item) for item in obj]
        else:
            return obj
    
    def _evict_if_needed(self, required_size: int) -> None:
        """Evict cache entries if memory limit would be exceeded."""
        
        if self._total_memory + required_size <= self.max_memory_bytes:
            return
        
        # Calculate how much memory to free
        memory_to_free = self._total_memory + required_size - self.max_memory_bytes
        freed_memory = 0
        
        # Get eviction candidates based on policy
        if self.eviction_policy == "lru":
            candidates = sorted(self._access_times.items(), key=lambda x: x[1])
        elif self.eviction_policy == "lfu":
            candidates = sorted(self._access_counts.items(), key=lambda x: x[1])
        else:  # FIFO
            candidates = list(self._access_times.items())
        
        # Evict entries
        for key, _ in candidates:
            if freed_memory >= memory_to_free:
                break
            
            if key in self._cache:
                freed_memory += self._memory_usage[key]
                self._evict_key(key)
                self.evictions += 1
    
    def _evict_key(self, key: str) -> None:
        """Evict specific key from cache."""
        if key in self._cache:
            self._total_memory -= self._memory_usage[key]
            
            del self._cache[key]
            del self._access_times[key]
            del self._access_counts[key]
            del self._memory_usage[key]
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Cache hit
                self.hits += 1
                self._access_times[key] = time.time()
                self._access_counts[key] += 1
                
                # Decompress if needed
                cached_data = self._cache[key]
                if isinstance(cached_data, bytes) and self.compression:
                    return True, self._decompress_object(cached_data)
                else:
                    return True, cached_data
            else:
                # Cache miss
                self.misses += 1
                return False, None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            # Compress if needed
            if self.compression:
                stored_value = self._compress_object(value)
            else:
                stored_value = value
            
            # Calculate memory usage
            memory_usage = self._get_object_size(stored_value)
            
            # Check if item already exists
            if key in self._cache:
                old_memory = self._memory_usage[key]
                self._total_memory += memory_usage - old_memory
            else:
                self._total_memory += memory_usage
            
            # Evict if necessary
            self._evict_if_needed(0)
            
            # Store item
            self._cache[key] = stored_value
            self._access_times[key] = time.time()
            self._access_counts[key] = 1
            self._memory_usage[key] = memory_usage
    
    def cached_function(self, func: Callable) -> Callable:
        """Decorator for caching function results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func, args, kwargs)
            
            # Try to get from cache
            hit, result = self.get(cache_key)
            if hit:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            self.put(cache_key, result)
            
            return result
        
        return wrapper
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / max(total_requests, 1)) * 100
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'cache_size': len(self._cache),
                'memory_usage_mb': self._total_memory / (1024 * 1024),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': (self._total_memory / self.max_memory_bytes) * 100
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._memory_usage.clear()
            self._total_memory = 0


class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize performance monitor."""
        self.logger = get_logger('performance_monitor')
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.operation_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Resource monitoring
        self.cpu_history: deque = deque(maxlen=100)
        self.memory_history: deque = deque(maxlen=100)
        self.gpu_history: deque = deque(maxlen=100)
        
        # Thresholds for alerts
        self.cpu_threshold = 80.0  # 80%
        self.memory_threshold = 85.0  # 85%
        self.gpu_threshold = 90.0  # 90%
        
        # Monitoring thread
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Store metrics
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_percent)
                
                # GPU metrics (if available)
                gpu_percent = self._get_gpu_usage()
                if gpu_percent is not None:
                    self.gpu_history.append(gpu_percent)
                
                # Check thresholds and alert
                self._check_thresholds(cpu_percent, memory_percent, gpu_percent)
                
                # Sleep until next monitoring interval
                self._stop_event.wait(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                self._stop_event.wait(self.monitoring_interval)
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        try:
            # Try nvidia-ml-py for NVIDIA GPUs
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except:
            return None
    
    def _check_thresholds(
        self, 
        cpu_percent: float, 
        memory_percent: float, 
        gpu_percent: Optional[float]
    ) -> None:
        """Check resource thresholds and generate alerts."""
        
        if cpu_percent > self.cpu_threshold:
            self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory_percent > self.memory_threshold:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
        
        if gpu_percent and gpu_percent > self.gpu_threshold:
            self.logger.warning(f"High GPU usage: {gpu_percent:.1f}%")
    
    @contextmanager
    def measure_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to measure operation performance."""
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        start_gpu = self._get_gpu_usage()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            cpu_usage=start_cpu,
            memory_usage=start_memory,
            gpu_usage=start_gpu or 0.0,
            metadata=metadata or {}
        )
        
        try:
            yield metrics
            metrics.success = True
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            raise
        
        finally:
            # Record final metrics
            end_time = time.time()
            metrics.end_time = end_time
            metrics.execution_time = end_time - start_time
            
            # Calculate throughput if applicable
            if 'data_size' in metrics.metadata:
                data_size = metrics.metadata['data_size']
                metrics.throughput = data_size / max(metrics.execution_time, 0.001)
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.operation_metrics[operation_name].append(metrics)
            
            # Log performance
            self.logger.debug(
                f"Operation {operation_name}: {metrics.execution_time:.3f}s, "
                f"Success: {metrics.success}"
            )
    
    def get_operation_statistics(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for specific operation."""
        
        if operation_name not in self.operation_metrics:
            return {'error': f'No data for operation: {operation_name}'}
        
        metrics_list = self.operation_metrics[operation_name]
        
        if not metrics_list:
            return {'error': f'No metrics available for: {operation_name}'}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in metrics_list if m.success]
        success_count = sum(1 for m in metrics_list if m.success)
        total_count = len(metrics_list)
        
        if execution_times:
            avg_time = np.mean(execution_times)
            min_time = np.min(execution_times)
            max_time = np.max(execution_times)
            std_time = np.std(execution_times)
            
            # Percentiles
            p50_time = np.percentile(execution_times, 50)
            p95_time = np.percentile(execution_times, 95)
            p99_time = np.percentile(execution_times, 99)
        else:
            avg_time = min_time = max_time = std_time = 0.0
            p50_time = p95_time = p99_time = 0.0
        
        # Throughput statistics
        throughputs = [m.throughput for m in metrics_list if m.success and m.throughput > 0]
        avg_throughput = np.mean(throughputs) if throughputs else 0.0
        
        return {
            'operation_name': operation_name,
            'total_calls': total_count,
            'successful_calls': success_count,
            'success_rate': (success_count / max(total_count, 1)) * 100,
            'avg_execution_time': avg_time,
            'min_execution_time': min_time,
            'max_execution_time': max_time,
            'std_execution_time': std_time,
            'p50_execution_time': p50_time,
            'p95_execution_time': p95_time,
            'p99_execution_time': p99_time,
            'avg_throughput': avg_throughput,
            'last_24h_calls': sum(
                1 for m in metrics_list[-100:] 
                if (time.time() - m.start_time) < 86400
            )
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        
        # Current resource usage
        current_cpu = self.cpu_history[-1] if self.cpu_history else 0
        current_memory = self.memory_history[-1] if self.memory_history else 0
        current_gpu = self.gpu_history[-1] if self.gpu_history else 0
        
        # Average usage over monitoring period
        avg_cpu = np.mean(self.cpu_history) if self.cpu_history else 0
        avg_memory = np.mean(self.memory_history) if self.memory_history else 0
        avg_gpu = np.mean(self.gpu_history) if self.gpu_history else 0
        
        # System health score (0-100)
        health_score = 100
        health_score -= min(current_cpu / 100 * 30, 30)  # CPU penalty
        health_score -= min(current_memory / 100 * 40, 40)  # Memory penalty  
        health_score -= min(current_gpu / 100 * 20, 20)  # GPU penalty
        health_score = max(0, health_score)
        
        # Recent failures
        recent_failures = sum(
            1 for m in list(self.metrics_history)[-100:] 
            if not m.success and (time.time() - m.start_time) < 3600
        )
        
        return {
            'health_score': health_score,
            'current_cpu_usage': current_cpu,
            'current_memory_usage': current_memory,
            'current_gpu_usage': current_gpu,
            'avg_cpu_usage': avg_cpu,
            'avg_memory_usage': avg_memory,
            'avg_gpu_usage': avg_gpu,
            'recent_failures_1h': recent_failures,
            'monitoring_active': self.monitoring_active,
            'uptime_seconds': time.time() - (self.metrics_history[0].start_time if self.metrics_history else time.time())
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # System health
        system_health = self.get_system_health()
        
        # Operation statistics
        operation_stats = {}
        for op_name in self.operation_metrics:
            operation_stats[op_name] = self.get_operation_statistics(op_name)
        
        # Top performing and problematic operations
        sorted_ops = sorted(
            operation_stats.items(),
            key=lambda x: x[1].get('avg_execution_time', 0),
            reverse=True
        )
        
        slowest_operations = sorted_ops[:5]
        fastest_operations = sorted_ops[-5:]
        
        # Performance trends (simplified)
        performance_trend = "stable"
        if len(self.metrics_history) > 50:
            recent_times = [m.execution_time for m in list(self.metrics_history)[-25:] if m.success]
            older_times = [m.execution_time for m in list(self.metrics_history)[-50:-25] if m.success]
            
            if recent_times and older_times:
                recent_avg = np.mean(recent_times)
                older_avg = np.mean(older_times)
                
                if recent_avg > older_avg * 1.2:
                    performance_trend = "degrading"
                elif recent_avg < older_avg * 0.8:
                    performance_trend = "improving"
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'system_health': system_health,
            'operation_statistics': operation_stats,
            'performance_summary': {
                'total_operations': len(self.metrics_history),
                'unique_operations': len(self.operation_metrics),
                'performance_trend': performance_trend,
                'slowest_operations': [op[0] for op in slowest_operations],
                'fastest_operations': [op[0] for op in fastest_operations]
            },
            'recommendations': self._generate_performance_recommendations(system_health, operation_stats)
        }
    
    def _generate_performance_recommendations(
        self, 
        system_health: Dict[str, Any], 
        operation_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        
        recommendations = []
        
        # System-level recommendations
        if system_health['current_cpu_usage'] > 80:
            recommendations.append("High CPU usage detected. Consider scaling CPU resources or optimizing algorithms.")
        
        if system_health['current_memory_usage'] > 85:
            recommendations.append("High memory usage detected. Implement memory optimization or increase available RAM.")
        
        if system_health['current_gpu_usage'] > 90:
            recommendations.append("GPU utilization is very high. Consider load balancing across multiple GPUs.")
        
        # Operation-level recommendations
        slow_operations = [
            (name, stats) for name, stats in operation_stats.items()
            if stats.get('avg_execution_time', 0) > 5.0  # Slower than 5 seconds
        ]
        
        if slow_operations:
            recommendations.append(f"Slow operations detected: {[op[0] for op in slow_operations[:3]]}. Consider optimization.")
        
        # Success rate recommendations
        low_success_ops = [
            (name, stats) for name, stats in operation_stats.items()
            if stats.get('success_rate', 100) < 90  # Less than 90% success
        ]
        
        if low_success_ops:
            recommendations.append(f"Low success rates for: {[op[0] for op in low_success_ops[:3]]}. Review error handling.")
        
        # General recommendations
        if system_health['health_score'] < 70:
            recommendations.append("Overall system health is poor. Review resource allocation and optimization strategies.")
        
        if not recommendations:
            recommendations.append("System performance is within normal parameters.")
        
        return recommendations


class AutoScaler:
    """Automatic scaling based on workload and performance."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """Initialize auto-scaler."""
        self.logger = get_logger('auto_scaler')
        self.performance_monitor = performance_monitor
        
        # Scaling configuration
        self.scaling_enabled = True
        self.min_resources = ResourceAllocation(cpu_cores=1, memory_mb=512)
        self.max_resources = ResourceAllocation(cpu_cores=16, memory_mb=16384)
        self.current_resources = ResourceAllocation(cpu_cores=4, memory_mb=2048)
        
        # Scaling thresholds
        self.scale_up_cpu_threshold = 75.0
        self.scale_down_cpu_threshold = 30.0
        self.scale_up_memory_threshold = 80.0
        self.scale_down_memory_threshold = 40.0
        
        # Scaling history
        self.scaling_actions = []
        self.last_scaling_time = 0
        self.scaling_cooldown = 300  # 5 minutes
    
    def should_scale_up(self) -> Tuple[bool, str]:
        """Check if system should scale up."""
        
        system_health = self.performance_monitor.get_system_health()
        
        cpu_usage = system_health['avg_cpu_usage']
        memory_usage = system_health['avg_memory_usage']
        
        if cpu_usage > self.scale_up_cpu_threshold:
            return True, f"High CPU usage: {cpu_usage:.1f}%"
        
        if memory_usage > self.scale_up_memory_threshold:
            return True, f"High memory usage: {memory_usage:.1f}%"
        
        # Check for performance degradation
        if system_health['health_score'] < 60:
            return True, f"Low health score: {system_health['health_score']:.1f}"
        
        return False, ""
    
    def should_scale_down(self) -> Tuple[bool, str]:
        """Check if system should scale down."""
        
        system_health = self.performance_monitor.get_system_health()
        
        cpu_usage = system_health['avg_cpu_usage']
        memory_usage = system_health['avg_memory_usage']
        
        if (cpu_usage < self.scale_down_cpu_threshold and 
            memory_usage < self.scale_down_memory_threshold):
            return True, f"Low resource usage: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%"
        
        return False, ""
    
    def scale_up(self, reason: str) -> bool:
        """Scale up resources."""
        
        if not self.scaling_enabled:
            return False
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return False
        
        # Calculate new resource allocation
        new_cpu = min(self.current_resources.cpu_cores * 2, self.max_resources.cpu_cores)
        new_memory = min(self.current_resources.memory_mb * 2, self.max_resources.memory_mb)
        
        if (new_cpu == self.current_resources.cpu_cores and 
            new_memory == self.current_resources.memory_mb):
            # Already at maximum
            return False
        
        # Apply scaling
        old_resources = self.current_resources
        self.current_resources = ResourceAllocation(cpu_cores=new_cpu, memory_mb=new_memory)
        
        # Record scaling action
        scaling_action = {
            'timestamp': datetime.now().isoformat(),
            'action': 'scale_up',
            'reason': reason,
            'old_resources': {'cpu': old_resources.cpu_cores, 'memory': old_resources.memory_mb},
            'new_resources': {'cpu': new_cpu, 'memory': new_memory}
        }
        
        self.scaling_actions.append(scaling_action)
        self.last_scaling_time = time.time()
        
        self.logger.info(f"Scaled up: CPU {old_resources.cpu_cores} -> {new_cpu}, Memory {old_resources.memory_mb} -> {new_memory}MB. Reason: {reason}")
        
        return True
    
    def scale_down(self, reason: str) -> bool:
        """Scale down resources."""
        
        if not self.scaling_enabled:
            return False
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return False
        
        # Calculate new resource allocation
        new_cpu = max(self.current_resources.cpu_cores // 2, self.min_resources.cpu_cores)
        new_memory = max(self.current_resources.memory_mb // 2, self.min_resources.memory_mb)
        
        if (new_cpu == self.current_resources.cpu_cores and 
            new_memory == self.current_resources.memory_mb):
            # Already at minimum
            return False
        
        # Apply scaling
        old_resources = self.current_resources
        self.current_resources = ResourceAllocation(cpu_cores=new_cpu, memory_mb=new_memory)
        
        # Record scaling action
        scaling_action = {
            'timestamp': datetime.now().isoformat(),
            'action': 'scale_down',
            'reason': reason,
            'old_resources': {'cpu': old_resources.cpu_cores, 'memory': old_resources.memory_mb},
            'new_resources': {'cpu': new_cpu, 'memory': new_memory}
        }
        
        self.scaling_actions.append(scaling_action)
        self.last_scaling_time = time.time()
        
        self.logger.info(f"Scaled down: CPU {old_resources.cpu_cores} -> {new_cpu}, Memory {old_resources.memory_mb} -> {new_memory}MB. Reason: {reason}")
        
        return True
    
    def auto_scale_check(self) -> None:
        """Perform automatic scaling check."""
        
        if not self.scaling_enabled:
            return
        
        # Check if we should scale up
        scale_up, up_reason = self.should_scale_up()
        if scale_up:
            self.scale_up(up_reason)
            return
        
        # Check if we should scale down
        scale_down, down_reason = self.should_scale_down()
        if scale_down:
            self.scale_down(down_reason)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        
        return {
            'scaling_enabled': self.scaling_enabled,
            'current_resources': {
                'cpu_cores': self.current_resources.cpu_cores,
                'memory_mb': self.current_resources.memory_mb
            },
            'resource_limits': {
                'min_cpu': self.min_resources.cpu_cores,
                'max_cpu': self.max_resources.cpu_cores,
                'min_memory': self.min_resources.memory_mb,
                'max_memory': self.max_resources.memory_mb
            },
            'recent_actions': self.scaling_actions[-10:],  # Last 10 actions
            'last_scaling_time': datetime.fromtimestamp(self.last_scaling_time).isoformat() if self.last_scaling_time else None,
            'cooldown_remaining': max(0, self.scaling_cooldown - (time.time() - self.last_scaling_time))
        }


# Performance optimization decorators
def gpu_accelerated(device_preference: str = "auto"):
    """Decorator for GPU acceleration."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            gpu_accelerator = GPUAccelerator(device_preference)
            
            # Add GPU context to kwargs
            kwargs['gpu_accelerator'] = gpu_accelerator
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cached(cache_size_mb: int = 512, ttl_seconds: Optional[float] = None):
    """Decorator for intelligent caching."""
    cache = IntelligentCache(max_memory_mb=cache_size_mb)
    
    def decorator(func: Callable) -> Callable:
        return cache.cached_function(func)
    
    return decorator


def monitored(operation_name: Optional[str] = None):
    """Decorator for performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal operation_name
            if operation_name is None:
                operation_name = func.__name__
            
            monitor = PerformanceMonitor()
            
            with monitor.measure_operation(operation_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global instances
global_gpu_accelerator = GPUAccelerator()
global_distributed_computing = DistributedComputing()
global_cache = IntelligentCache()
global_performance_monitor = PerformanceMonitor()
global_auto_scaler = AutoScaler(global_performance_monitor)


# Export main classes and functions
__all__ = [
    'PerformanceMetrics', 'ResourceAllocation',
    'GPUAccelerator', 'DistributedComputing', 'IntelligentCache',
    'PerformanceMonitor', 'AutoScaler',
    'gpu_accelerated', 'cached', 'monitored',
    'global_gpu_accelerator', 'global_distributed_computing',
    'global_cache', 'global_performance_monitor', 'global_auto_scaler'
]