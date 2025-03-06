"""
Base interface for caching system.
"""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

class BaseCache(ABC):
    """Base class for caching system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the cache with configuration.
        
        Args:
            config: Configuration parameters for the cache
        """
        self.config = config
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the cache."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Any:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists, None otherwise
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live for the cached value
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache.
        
        Args:
            key: Cache key to delete
        """
        pass
    
    @abstractmethod
    async def exists(
        self,
        key: str,
        **kwargs: Any
    ) -> bool:
        """Check if a key exists in the cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all values from cache."""
        pass

class CacheConfig:
    """Configuration for a cache."""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')
        self.type = kwargs.get('type', '')
        self.ttl = kwargs.get('ttl', timedelta(hours=1))
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['BaseCache', 'CacheConfig']
