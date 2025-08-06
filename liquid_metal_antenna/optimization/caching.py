"""
Advanced caching system for liquid metal antenna optimization.
"""

import os
import time
import pickle
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

from ..utils.logging_config import get_logger
from ..utils.security import SecureFileHandler, SecurityError
from ..solvers.base import SolverResult


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    data: Any
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    size_bytes: int
    metadata: Dict[str, Any]
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > ttl_seconds
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheStorage:
    """Base class for cache storage backends."""
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        raise NotImplementedError
    
    def put(self, key: str, entry: CacheEntry) -> None:
        """Store entry."""
        raise NotImplementedError
    
    def delete(self, key: str) -> None:
        """Delete entry."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear all entries."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        raise NotImplementedError


class MemoryStorage(CacheStorage):
    """In-memory cache storage with LRU eviction."""
    
    def __init__(self, max_size_mb: float = 1000):
        """
        Initialize memory storage.
        
        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.entries: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.current_size = 0
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        with self.lock:
            entry = self.entries.get(key)
            if entry:
                entry.update_access()
            return entry
    
    def put(self, key: str, entry: CacheEntry) -> None:
        """Store entry with LRU eviction."""
        with self.lock:
            # Remove existing entry if present
            if key in self.entries:
                self.current_size -= self.entries[key].size_bytes
                del self.entries[key]
            
            # Check if we need to evict entries
            while (self.current_size + entry.size_bytes > self.max_size_bytes 
                   and self.entries):
                self._evict_lru()
            
            # Add new entry
            self.entries[key] = entry
            self.current_size += entry.size_bytes
    
    def delete(self, key: str) -> None:
        """Delete entry."""
        with self.lock:
            if key in self.entries:
                self.current_size -= self.entries[key].size_bytes
                del self.entries[key]
    
    def clear(self) -> None:
        """Clear all entries."""
        with self.lock:
            self.entries.clear()
            self.current_size = 0
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.entries:
            return
        
        # Find LRU entry
        lru_key = min(self.entries.keys(), 
                     key=lambda k: self.entries[k].last_accessed)
        
        # Remove it
        self.current_size -= self.entries[lru_key].size_bytes
        del self.entries[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = sum(entry.access_count for entry in self.entries.values())
            
            return {
                'storage_type': 'memory',
                'total_entries': len(self.entries),
                'current_size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization_percent': (self.current_size / self.max_size_bytes) * 100,
                'total_accesses': total_accesses,
                'avg_accesses_per_entry': total_accesses / len(self.entries) if self.entries else 0
            }


class DiskStorage(CacheStorage):
    """Disk-based cache storage with SQLite metadata."""
    
    def __init__(self, cache_dir: str, max_size_mb: float = 10000):
        """
        Initialize disk storage.
        
        Args:
            cache_dir: Cache directory path
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.file_handler = SecureFileHandler(str(self.cache_dir))
        self.lock = threading.RLock()
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    timestamp TEXT,
                    access_count INTEGER,
                    last_accessed TEXT,
                    size_bytes INTEGER,
                    metadata TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)')
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        'SELECT timestamp, access_count, last_accessed, size_bytes, metadata FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Load data from file
                data_file = self.data_dir / f"{self._hash_key(key)}.pkl"
                if not data_file.exists():
                    # Cleanup orphaned metadata
                    self.delete(key)
                    return None
                
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    data=data,
                    timestamp=datetime.fromisoformat(row[0]),
                    access_count=row[1],
                    last_accessed=datetime.fromisoformat(row[2]),
                    size_bytes=row[3],
                    metadata=pickle.loads(row[4]) if row[4] else {}
                )
                
                # Update access statistics
                entry.update_access()
                self._update_access_stats(key, entry)
                
                return entry
                
            except Exception:
                return None
    
    def put(self, key: str, entry: CacheEntry) -> None:
        """Store entry on disk."""
        with self.lock:
            try:
                # Check if we need to evict entries
                self._ensure_space(entry.size_bytes)
                
                # Save data to file
                data_file = self.data_dir / f"{self._hash_key(key)}.pkl"
                with open(data_file, 'wb') as f:
                    pickle.dump(entry.data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save metadata to database
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (key, timestamp, access_count, last_accessed, size_bytes, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        key,
                        entry.timestamp.isoformat(),
                        entry.access_count,
                        entry.last_accessed.isoformat(),
                        entry.size_bytes,
                        pickle.dumps(entry.metadata)
                    ))
                    
            except Exception as e:
                # Cleanup on failure
                data_file = self.data_dir / f"{self._hash_key(key)}.pkl"
                if data_file.exists():
                    data_file.unlink()
                raise e
    
    def delete(self, key: str) -> None:
        """Delete entry."""
        with self.lock:
            # Remove data file
            data_file = self.data_dir / f"{self._hash_key(key)}.pkl"
            if data_file.exists():
                data_file.unlink()
            
            # Remove metadata
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
    
    def clear(self) -> None:
        """Clear all entries."""
        with self.lock:
            # Remove all data files
            for data_file in self.data_dir.glob("*.pkl"):
                data_file.unlink()
            
            # Clear database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('DELETE FROM cache_entries')
    
    def _hash_key(self, key: str) -> str:
        """Create filename-safe hash of key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Ensure sufficient space by evicting old entries."""
        current_size = self._get_current_size()
        
        while current_size + required_bytes > self.max_size_bytes:
            # Find oldest entry
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    'SELECT key FROM cache_entries ORDER BY last_accessed LIMIT 1'
                )
                row = cursor.fetchone()
            
            if not row:
                break
            
            # Delete oldest entry
            self.delete(row[0])
            current_size = self._get_current_size()
    
    def _get_current_size(self) -> int:
        """Get current cache size in bytes."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute('SELECT SUM(size_bytes) FROM cache_entries')
            result = cursor.fetchone()[0]
            return result or 0
    
    def _update_access_stats(self, key: str, entry: CacheEntry) -> None:
        """Update access statistics in database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                UPDATE cache_entries 
                SET access_count = ?, last_accessed = ?
                WHERE key = ?
            ''', (entry.access_count, entry.last_accessed.isoformat(), key))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses
                    FROM cache_entries
                ''')
                stats = cursor.fetchone()
            
            return {
                'storage_type': 'disk',
                'cache_dir': str(self.cache_dir),
                'total_entries': stats[0] or 0,
                'current_size_mb': (stats[1] or 0) / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization_percent': ((stats[1] or 0) / self.max_size_bytes) * 100,
                'total_accesses': stats[2] or 0,
                'avg_accesses_per_entry': stats[3] or 0
            }


class SimulationCache:
    """High-performance cache for electromagnetic simulation results."""
    
    def __init__(
        self,
        storage: Optional[CacheStorage] = None,
        ttl_hours: int = 24,
        geometry_tolerance: float = 1e-6,
        frequency_tolerance: float = 1e3  # 1 kHz
    ):
        """
        Initialize simulation cache.
        
        Args:
            storage: Cache storage backend
            ttl_hours: Time-to-live in hours
            geometry_tolerance: Tolerance for geometry matching
            frequency_tolerance: Tolerance for frequency matching
        """
        self.storage = storage or MemoryStorage()
        self.ttl_seconds = ttl_hours * 3600
        self.geometry_tolerance = geometry_tolerance
        self.frequency_tolerance = frequency_tolerance
        
        self.logger = get_logger('simulation_cache')
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.puts = 0
        
    def _create_cache_key(
        self,
        geometry: np.ndarray,
        frequency: float,
        solver_params: Dict[str, Any]
    ) -> str:
        """Create unique cache key for simulation parameters."""
        # Normalize geometry for consistent hashing
        geometry_normalized = np.round(geometry / self.geometry_tolerance) * self.geometry_tolerance
        
        # Normalize frequency
        frequency_normalized = round(frequency / self.frequency_tolerance) * self.frequency_tolerance
        
        # Create deterministic hash
        key_data = {
            'geometry_hash': hashlib.sha256(geometry_normalized.tobytes()).hexdigest(),
            'frequency': frequency_normalized,
            'solver_params': sorted(solver_params.items())
        }
        
        key_string = str(key_data)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(
        self,
        geometry: np.ndarray,
        frequency: float,
        solver_params: Dict[str, Any]
    ) -> Optional[SolverResult]:
        """
        Get cached simulation result.
        
        Args:
            geometry: Antenna geometry
            frequency: Simulation frequency
            solver_params: Solver parameters
            
        Returns:
            Cached result or None if not found
        """
        try:
            key = self._create_cache_key(geometry, frequency, solver_params)
            entry = self.storage.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            # Check expiration
            if entry.is_expired(self.ttl_seconds):
                self.storage.delete(key)
                self.misses += 1
                return None
            
            self.hits += 1
            self.logger.debug(f"Cache hit for key {key[:16]}...")
            
            return entry.data
            
        except Exception as e:
            self.logger.warning(f"Cache get error: {str(e)}")
            self.misses += 1
            return None
    
    def put(
        self,
        geometry: np.ndarray,
        frequency: float,
        solver_params: Dict[str, Any],
        result: SolverResult
    ) -> None:
        """
        Store simulation result in cache.
        
        Args:
            geometry: Antenna geometry
            frequency: Simulation frequency
            solver_params: Solver parameters
            result: Simulation result to cache
        """
        try:
            key = self._create_cache_key(geometry, frequency, solver_params)
            
            # Estimate result size
            result_size = self._estimate_result_size(result)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=result,
                timestamp=datetime.now(),
                access_count=0,
                last_accessed=datetime.now(),
                size_bytes=result_size,
                metadata={
                    'frequency': frequency,
                    'geometry_shape': geometry.shape,
                    'solver_params': solver_params,
                    'computation_time': getattr(result, 'computation_time', 0)
                }
            )
            
            self.storage.put(key, entry)
            self.puts += 1
            
            self.logger.debug(f"Cached result for key {key[:16]}... (size: {result_size} bytes)")
            
        except Exception as e:
            self.logger.warning(f"Cache put error: {str(e)}")
    
    def _estimate_result_size(self, result: SolverResult) -> int:
        """Estimate memory size of result object."""
        try:
            import sys
            
            size = sys.getsizeof(result)
            
            # Add array sizes
            if hasattr(result, 's_parameters') and result.s_parameters is not None:
                size += result.s_parameters.nbytes
            
            if hasattr(result, 'radiation_pattern') and result.radiation_pattern is not None:
                size += result.radiation_pattern.nbytes
            
            if hasattr(result, 'frequencies') and result.frequencies is not None:
                size += result.frequencies.nbytes
            
            return size
            
        except Exception:
            return 1000  # Default estimate
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (simple substring matching)
            
        Returns:
            Number of entries invalidated
        """
        # This is a simplified implementation
        # In practice, would need more sophisticated pattern matching
        self.logger.info(f"Pattern-based invalidation not fully implemented: {pattern}")
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        storage_stats = self.storage.get_stats()
        
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        return {
            'cache_type': 'SimulationCache',
            'hit_rate_percent': hit_rate * 100,
            'total_hits': self.hits,
            'total_misses': self.misses,
            'total_puts': self.puts,
            'ttl_hours': self.ttl_seconds / 3600,
            'tolerances': {
                'geometry': self.geometry_tolerance,
                'frequency_hz': self.frequency_tolerance
            },
            'storage_stats': storage_stats
        }
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        # This would need to be implemented based on storage backend
        self.logger.info("Cleanup not implemented for current storage backend")
        return 0


class ResultCache:
    """Specialized cache for optimization results."""
    
    def __init__(self, storage: Optional[CacheStorage] = None):
        """Initialize result cache."""
        self.storage = storage or MemoryStorage(max_size_mb=500)
        self.logger = get_logger('result_cache')
    
    def cache_optimization_result(
        self,
        spec_hash: str,
        objective: str,
        constraints: Dict[str, Any],
        result: Any
    ) -> None:
        """Cache optimization result."""
        key = f"opt_{spec_hash}_{objective}_{hash(str(sorted(constraints.items())))}"
        
        entry = CacheEntry(
            key=key,
            data=result,
            timestamp=datetime.now(),
            access_count=0,
            last_accessed=datetime.now(),
            size_bytes=1000,  # Simplified
            metadata={'type': 'optimization_result'}
        )
        
        self.storage.put(key, entry)
        self.logger.debug(f"Cached optimization result: {key[:16]}...")
    
    def get_optimization_result(
        self,
        spec_hash: str,
        objective: str,
        constraints: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached optimization result."""
        key = f"opt_{spec_hash}_{objective}_{hash(str(sorted(constraints.items())))}"
        entry = self.storage.get(key)
        
        if entry and not entry.is_expired(3600):  # 1 hour TTL
            self.logger.debug(f"Found cached optimization result: {key[:16]}...")
            return entry.data
        
        return None


class GeometryCache:
    """Cache for geometry processing results."""
    
    def __init__(self, storage: Optional[CacheStorage] = None):
        """Initialize geometry cache."""
        self.storage = storage or MemoryStorage(max_size_mb=200)
        self.logger = get_logger('geometry_cache')
    
    def cache_geometry_processing(
        self,
        geometry_hash: str,
        processing_type: str,
        result: Any
    ) -> None:
        """Cache geometry processing result."""
        key = f"geom_{geometry_hash}_{processing_type}"
        
        entry = CacheEntry(
            key=key,
            data=result,
            timestamp=datetime.now(),
            access_count=0,
            last_accessed=datetime.now(),
            size_bytes=len(str(result)),
            metadata={'type': 'geometry_processing'}
        )
        
        self.storage.put(key, entry)
    
    def get_geometry_processing(
        self,
        geometry_hash: str,
        processing_type: str
    ) -> Optional[Any]:
        """Get cached geometry processing result."""
        key = f"geom_{geometry_hash}_{processing_type}"
        entry = self.storage.get(key)
        
        if entry and not entry.is_expired(7200):  # 2 hour TTL
            return entry.data
        
        return None