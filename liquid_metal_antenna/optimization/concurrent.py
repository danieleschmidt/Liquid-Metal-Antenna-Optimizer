"""
Concurrent processing infrastructure for liquid metal antenna optimization.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Iterator
from dataclasses import dataclass
from queue import Queue, Empty
import pickle
import traceback

from ..utils.logging_config import get_logger
from .performance import ResourceManager


@dataclass
class Task:
    """Represents a computational task."""
    
    task_id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: int = 0  # Higher values = higher priority
    estimated_time: float = 1.0  # Estimated execution time in seconds
    memory_mb: float = 100  # Estimated memory usage in MB
    requires_gpu: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskResult:
    """Result of task execution."""
    
    task_id: str
    success: bool
    result: Any
    error: Optional[Exception]
    execution_time: float
    memory_used_mb: float
    worker_id: str
    start_time: float
    end_time: float


class TaskQueue:
    """Thread-safe priority task queue."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize task queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self.queue = Queue(max_size)
        self.priority_tasks = []  # For priority handling
        self.lock = threading.Lock()
        self._closed = False
    
    def put(self, task: Task, timeout: Optional[float] = None) -> bool:
        """
        Add task to queue.
        
        Args:
            task: Task to add
            timeout: Timeout in seconds
            
        Returns:
            True if task was added successfully
        """
        if self._closed:
            return False
        
        try:
            with self.lock:
                # Insert task in priority order
                inserted = False
                for i, (priority, existing_task) in enumerate(self.priority_tasks):
                    if task.priority > priority:
                        self.priority_tasks.insert(i, (task.priority, task))
                        inserted = True
                        break
                
                if not inserted:
                    self.priority_tasks.append((task.priority, task))
                
                # Move highest priority task to queue
                if self.priority_tasks:
                    _, next_task = self.priority_tasks.pop(0)
                    self.queue.put(next_task, timeout=timeout)
                    return True
                
            return True
            
        except Exception:
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Get next task from queue."""
        if self._closed and self.queue.empty():
            return None
        
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize() + len(self.priority_tasks)
    
    def close(self) -> None:
        """Close the queue."""
        self._closed = True
    
    def is_closed(self) -> bool:
        """Check if queue is closed."""
        return self._closed


class Worker:
    """Individual worker for task execution."""
    
    def __init__(
        self,
        worker_id: str,
        task_queue: TaskQueue,
        result_callback: Callable[[TaskResult], None],
        resource_manager: ResourceManager
    ):
        """
        Initialize worker.
        
        Args:
            worker_id: Unique worker identifier
            task_queue: Task queue to process
            result_callback: Callback for results
            resource_manager: Resource manager
        """
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_callback = result_callback
        self.resource_manager = resource_manager
        
        self.logger = get_logger(f'worker_{worker_id}')
        
        self.running = False
        self.current_task = None
        self.thread = None
        
        # Statistics
        self.tasks_completed = 0
        self.total_execution_time = 0.0
        self.errors = 0
    
    def start(self) -> None:
        """Start worker thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
            self.logger.debug(f"Worker {self.worker_id} started")
    
    def stop(self) -> None:
        """Stop worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            self.logger.debug(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self) -> None:
        """Main worker processing loop."""
        while self.running:
            try:
                # Get next task
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:
                    continue
                
                # Execute task
                self.current_task = task
                result = self._execute_task(task)
                
                # Report result
                self.result_callback(result)
                
                # Update statistics
                self.tasks_completed += 1
                self.total_execution_time += result.execution_time
                
                if not result.success:
                    self.errors += 1
                
                self.current_task = None
                
            except Exception as e:
                self.logger.error(f"Worker error: {str(e)}")
                self.errors += 1
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Acquire resources
            resource_type = 'gpu' if task.requires_gpu else 'cpu'
            
            with self.resource_manager.acquire_resources(resource_type, task.memory_mb):
                # Execute function
                result = task.function(*task.args, **task.kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    error=None,
                    execution_time=execution_time,
                    memory_used_mb=task.memory_mb,  # Simplified
                    worker_id=self.worker_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error=e,
                execution_time=execution_time,
                memory_used_mb=0,
                worker_id=self.worker_id,
                start_time=start_time,
                end_time=end_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        avg_execution_time = (self.total_execution_time / self.tasks_completed 
                             if self.tasks_completed > 0 else 0)
        
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'tasks_completed': self.tasks_completed,
            'total_execution_time': self.total_execution_time,
            'avg_execution_time': avg_execution_time,
            'errors': self.errors,
            'error_rate': self.errors / max(self.tasks_completed, 1),
            'current_task': self.current_task.task_id if self.current_task else None
        }


class ConcurrentProcessor:
    """High-level concurrent processing coordinator."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        """
        Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of workers
            resource_manager: Resource manager instance
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.max_workers = max_workers or self.resource_manager.limits.max_concurrent_tasks
        
        self.logger = get_logger('concurrent_processor')
        
        # Task management
        self.task_queue = TaskQueue()
        self.workers = []
        self.results = {}  # task_id -> TaskResult
        self.result_lock = threading.Lock()
        
        # Statistics
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.start_time = None
        
        # Initialize workers
        self._create_workers()
        
        self.logger.info(f"Concurrent processor initialized with {len(self.workers)} workers")
    
    def _create_workers(self) -> None:
        """Create worker threads."""
        for i in range(self.max_workers):
            worker_id = f"worker_{i:02d}"
            worker = Worker(
                worker_id=worker_id,
                task_queue=self.task_queue,
                result_callback=self._handle_result,
                resource_manager=self.resource_manager
            )
            self.workers.append(worker)
    
    def _handle_result(self, result: TaskResult) -> None:
        """Handle task result."""
        with self.result_lock:
            self.results[result.task_id] = result
            self.tasks_completed += 1
        
        self.logger.debug(f"Task {result.task_id} completed by {result.worker_id} "
                         f"in {result.execution_time:.3f}s")
    
    def start(self) -> None:
        """Start all workers."""
        if self.start_time is None:
            self.start_time = time.time()
        
        for worker in self.workers:
            worker.start()
        
        self.logger.info("Concurrent processor started")
    
    def stop(self) -> None:
        """Stop all workers."""
        self.task_queue.close()
        
        for worker in self.workers:
            worker.stop()
        
        self.logger.info("Concurrent processor stopped")
    
    def submit_task(
        self,
        function: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: int = 0,
        estimated_time: float = 1.0,
        memory_mb: float = 100,
        requires_gpu: bool = False,
        **kwargs
    ) -> str:
        """
        Submit task for concurrent execution.
        
        Args:
            function: Function to execute
            *args: Function arguments
            task_id: Optional task identifier
            priority: Task priority
            estimated_time: Estimated execution time
            memory_mb: Estimated memory usage
            requires_gpu: Whether task requires GPU
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{self.tasks_submitted:06d}_{int(time.time() * 1000) % 100000}"
        
        task = Task(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            estimated_time=estimated_time,
            memory_mb=memory_mb,
            requires_gpu=requires_gpu
        )
        
        success = self.task_queue.put(task)
        if success:
            self.tasks_submitted += 1
            self.logger.debug(f"Submitted task {task_id}")
        else:
            self.logger.warning(f"Failed to submit task {task_id}")
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Get result for specific task.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Task result or None if not available
        """
        start_time = time.time()
        
        while True:
            with self.result_lock:
                if task_id in self.results:
                    return self.results[task_id]
            
            if timeout is not None and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.01)  # Short sleep to avoid busy waiting
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all submitted tasks to complete.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if all tasks completed
        """
        start_time = time.time()
        
        while self.tasks_completed < self.tasks_submitted:
            if timeout is not None and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)
        
        return True
    
    def get_all_results(self) -> Dict[str, TaskResult]:
        """Get all completed results."""
        with self.result_lock:
            return self.results.copy()
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        # Worker statistics
        worker_stats = [worker.get_stats() for worker in self.workers]
        
        # Aggregate statistics
        total_tasks_by_workers = sum(stats['tasks_completed'] for stats in worker_stats)
        total_errors = sum(stats['errors'] for stats in worker_stats)
        
        with self.result_lock:
            completed_tasks = self.tasks_completed
            total_results = len(self.results)
        
        return {
            'processor_info': {
                'max_workers': self.max_workers,
                'active_workers': sum(1 for stats in worker_stats if stats['running']),
                'runtime_seconds': runtime
            },
            'task_stats': {
                'tasks_submitted': self.tasks_submitted,
                'tasks_completed': completed_tasks,
                'tasks_in_queue': self.task_queue.size(),
                'completion_rate': completed_tasks / max(self.tasks_submitted, 1),
                'tasks_per_second': completed_tasks / max(runtime, 1)
            },
            'error_stats': {
                'total_errors': total_errors,
                'error_rate': total_errors / max(total_tasks_by_workers, 1)
            },
            'worker_stats': worker_stats,
            'resource_stats': self.resource_manager.get_resource_stats()
        }
    
    def optimize_worker_count(self) -> int:
        """
        Dynamically optimize number of workers based on performance.
        
        Returns:
            New optimal worker count
        """
        stats = self.get_processor_stats()
        
        # Simple optimization based on queue size and completion rate
        queue_size = stats['task_stats']['tasks_in_queue']
        completion_rate = stats['task_stats']['completion_rate']
        
        current_workers = len(self.workers)
        optimal_workers = current_workers
        
        # Increase workers if queue is growing and system has capacity
        if queue_size > current_workers * 2:
            resource_stats = stats['resource_stats']
            if (resource_stats['cpu_usage_percent'] < 80 and 
                resource_stats['memory_usage_percent'] < 80):
                optimal_workers = min(current_workers + 2, self.max_workers)
        
        # Decrease workers if system is over-utilized
        elif queue_size < current_workers // 2:
            resource_stats = stats['resource_stats']
            if (resource_stats['cpu_usage_percent'] > 90 or 
                resource_stats['memory_usage_percent'] > 90):
                optimal_workers = max(current_workers - 1, 1)
        
        # Adjust worker count if different
        if optimal_workers != current_workers:
            self._adjust_worker_count(optimal_workers)
            self.logger.info(f"Optimized worker count: {current_workers} -> {optimal_workers}")
        
        return optimal_workers
    
    def _adjust_worker_count(self, target_count: int) -> None:
        """Adjust number of active workers."""
        current_count = len(self.workers)
        
        if target_count > current_count:
            # Add workers
            for i in range(current_count, target_count):
                worker_id = f"worker_{i:02d}"
                worker = Worker(
                    worker_id=worker_id,
                    task_queue=self.task_queue,
                    result_callback=self._handle_result,
                    resource_manager=self.resource_manager
                )
                worker.start()
                self.workers.append(worker)
        
        elif target_count < current_count:
            # Remove workers
            workers_to_remove = self.workers[target_count:]
            self.workers = self.workers[:target_count]
            
            for worker in workers_to_remove:
                worker.stop()


class TaskPool:
    """High-level task pool with automatic load balancing."""
    
    def __init__(
        self,
        pool_type: str = 'mixed',  # 'cpu', 'io', 'mixed'
        max_workers: Optional[int] = None
    ):
        """
        Initialize task pool.
        
        Args:
            pool_type: Type of pool ('cpu', 'io', 'mixed')
            max_workers: Maximum number of workers
        """
        self.pool_type = pool_type
        self.resource_manager = ResourceManager()
        
        if max_workers is None:
            if pool_type == 'cpu':
                max_workers = self.resource_manager.get_optimal_thread_count('cpu_bound')
            elif pool_type == 'io':
                max_workers = self.resource_manager.get_optimal_thread_count('io_bound')
            else:  # mixed
                max_workers = self.resource_manager.get_optimal_thread_count('mixed')
        
        self.processor = ConcurrentProcessor(max_workers, self.resource_manager)
        self.logger = get_logger(f'task_pool_{pool_type}')
        
    def __enter__(self):
        """Context manager entry."""
        self.processor.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.processor.stop()
    
    def map(
        self,
        function: Callable,
        iterable: Iterator,
        chunk_size: int = 1,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Map function over iterable with concurrent execution.
        
        Args:
            function: Function to apply
            iterable: Input data
            chunk_size: Items per task
            timeout: Overall timeout
            
        Returns:
            Results in order
        """
        self.processor.start()
        
        try:
            # Submit tasks
            items = list(iterable)
            task_ids = []
            
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                if chunk_size == 1:
                    task_id = self.processor.submit_task(function, chunk[0])
                else:
                    task_id = self.processor.submit_task(self._map_chunk, function, chunk)
                task_ids.append(task_id)
            
            # Wait for completion
            if not self.processor.wait_for_completion(timeout):
                raise TimeoutError("Task pool mapping timeout")
            
            # Collect results in order
            results = []
            for task_id in task_ids:
                result = self.processor.get_result(task_id)
                if result and result.success:
                    if chunk_size == 1:
                        results.append(result.result)
                    else:
                        results.extend(result.result)
                else:
                    error = result.error if result else TimeoutError("Task failed")
                    raise error
            
            return results
            
        finally:
            self.processor.stop()
    
    def _map_chunk(self, function: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [function(item) for item in chunk]
    
    def starmap(
        self,
        function: Callable,
        iterable: Iterator[Tuple],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Starmap function over iterable of argument tuples.
        
        Args:
            function: Function to apply
            iterable: Input tuples
            timeout: Overall timeout
            
        Returns:
            Results in order
        """
        return self.map(lambda args: function(*args), iterable, timeout=timeout)
    
    def submit(self, function: Callable, *args, **kwargs) -> Future:
        """
        Submit single task and return Future.
        
        Args:
            function: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object
        """
        self.processor.start()
        
        task_id = self.processor.submit_task(function, *args, **kwargs)
        
        # Create a custom Future-like object
        class TaskFuture:
            def __init__(self, processor, task_id):
                self.processor = processor
                self.task_id = task_id
            
            def result(self, timeout=None):
                result = self.processor.get_result(self.task_id, timeout)
                if result is None:
                    raise TimeoutError("Task timeout")
                if not result.success:
                    raise result.error
                return result.result
            
            def done(self):
                return self.task_id in self.processor.results
        
        return TaskFuture(self.processor, task_id)