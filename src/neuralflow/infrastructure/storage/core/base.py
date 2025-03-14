"""
Base storage system for NeuralFlow.
Provides a unified interface for all storage operations.
"""

import logging
import threading
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')

class StorageConfig:
    """Configuration for storage systems."""
    
    def __init__(self,
                 storage_id: str,
                 storage_type: str,
                 root_dir: Optional[Union[str, Path]] = None,
                 **kwargs):
        """Initialize storage configuration.
        
        Args:
            storage_id: Unique identifier for the storage
            storage_type: Type of storage (file, database, vector, etc.)
            root_dir: Optional root directory for storage
            **kwargs: Additional configuration parameters
        """
        self.id = storage_id
        self.type = storage_type
        self.root_dir = Path(root_dir) if root_dir else None
        self.parameters = kwargs
        self.metadata = {}

class BaseStorage(ABC, Generic[T]):
    """Base class for all storage implementations."""
    
    def __init__(self, config: StorageConfig):
        """Initialize the storage system.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._last_access = {}
        self._creation_time = {}
        
        if self.config.root_dir:
            self._ensure_storage_dir()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        if self.config.root_dir:
            self.config.root_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized storage at {self.config.root_dir}")
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage system."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up storage resources."""
        pass
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information.
        
        Returns:
            Dictionary containing storage information
        """
        return {
            'id': self.config.id,
            'type': self.config.type,
            'root_dir': str(self.config.root_dir) if self.config.root_dir else None,
            'created': self.created,
            'modified': self.modified,
            'parameters': self.config.parameters,
            'metadata': self.config.metadata,
            'cache_size': len(self._cache)
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        try:
            stats = {
                'total_entries': len(self._cache),
                'cache_size': sum(len(str(v)) for v in self._cache.values()),
                'last_modified': self.modified
            }
            
            if self.config.root_dir:
                stats.update({
                    'disk_usage': sum(f.stat().st_size for f in self.config.root_dir.rglob('*') if f.is_file()),
                    'file_count': len(list(self.config.root_dir.rglob('*')))
                })
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._cache_lock:
            if key in self._cache:
                self._last_access[key] = datetime.now()
                return self._cache[key]
            return None
    
    def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
        """
        with self._cache_lock:
            self._cache[key] = value
            now = datetime.now()
            self._last_access[key] = now
            self._creation_time[key] = now
    
    def _cache_delete(self, key: str) -> bool:
        """Delete cache value.
        
        Args:
            key: Cache key
            
        Returns:
            True if value was deleted, False otherwise
        """
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                del self._last_access[key]
                del self._creation_time[key]
                return True
            return False
    
    def _cache_cleanup(self, max_age: Optional[int] = None, max_idle: Optional[int] = None) -> int:
        """Clean up expired cache entries.
        
        Args:
            max_age: Maximum age in seconds
            max_idle: Maximum idle time in seconds
            
        Returns:
            Number of entries removed
        """
        removed = 0
        now = datetime.now()
        
        with self._cache_lock:
            keys = list(self._cache.keys())
            for key in keys:
                if max_age and (now - self._creation_time[key]).total_seconds() > max_age:
                    self._cache_delete(key)
                    removed += 1
                elif max_idle and (now - self._last_access[key]).total_seconds() > max_idle:
                    self._cache_delete(key)
                    removed += 1
        
        return removed 