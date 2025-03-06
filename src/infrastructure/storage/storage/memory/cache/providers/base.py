"""
Base provider interfaces for cache stores.
This module provides base classes for cache store provider implementations.
"""

import logging
from typing import Dict, Any, Optional, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheConfig:
    """Configuration for cache stores."""
    
    def __init__(self,
                 max_size: Optional[int] = None,
                 ttl: Optional[int] = None,
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            max_size: Maximum number of items to store
            ttl: Time-to-live in seconds
            **kwargs: Additional configuration parameters
        """
        self.max_size = max_size
        self.ttl = ttl
        self.extra_params = kwargs

class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    
    def __init__(self,
                 key: str,
                 value: T,
                 ttl: Optional[int] = None):
        """Initialize the cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            ttl: Time-to-live in seconds
        """
        self.key = key
        self.value = value
        self.created = datetime.now()
        self.expires = (self.created + timedelta(seconds=ttl)) if ttl else None
        self.hits = 0
        self.last_accessed = self.created
    
    def is_expired(self) -> bool:
        """Check if the entry has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expires is None:
            return False
        return datetime.now() > self.expires
    
    def access(self) -> None:
        """Record an access to this entry."""
        self.hits += 1
        self.last_accessed = datetime.now()

class BaseCacheStore(ABC, Generic[T]):
    """Base class for cache stores."""
    
    def __init__(self, store_id: str,
                 config: CacheConfig,
                 **kwargs):
        """Initialize the cache store.
        
        Args:
            store_id: Unique identifier for the store
            config: Cache store configuration
            **kwargs: Additional initialization parameters
        """
        self.id = store_id
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
        self._entries: Dict[str, CacheEntry[T]] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the cache store."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up cache store resources."""
        pass
    
    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            entry = self._entries.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._entries[key]
                return None
            
            entry.access()
            return entry.value
        except Exception as e:
            logger.error(f"Failed to get value for key {key} from store {self.id}: {e}")
            return None
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check max size
            if (self.config.max_size is not None and
                len(self._entries) >= self.config.max_size):
                self._evict_oldest()
            
            entry = CacheEntry(key, value, ttl or self.config.ttl)
            self._entries[key] = entry
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to set value for key {key} in store {self.id}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if key not in self._entries:
                return False
            
            del self._entries[key]
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to delete key {key} from store {self.id}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all values from the cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._entries.clear()
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to clear store {self.id}: {e}")
            return False
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get store information.
        
        Returns:
            Dictionary containing store information
        """
        return {
            'id': self.id,
            'type': type(self).__name__,
            'created': self.created,
            'modified': self.modified,
            'config': {
                'max_size': self.config.max_size,
                'ttl': self.config.ttl,
                'extra_params': self.config.extra_params
            },
            'stats': {
                'entries': len(self._entries),
                'hits': sum(e.hits for e in self._entries.values())
            }
        }
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self._entries:
            return
        
        oldest_key = min(self._entries.items(),
                        key=lambda x: x[1].last_accessed)[0]
        del self._entries[oldest_key]

class BaseCacheProvider(ABC):
    """Base class for cache providers."""
    
    def __init__(self, provider_id: str,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the provider.
        
        Args:
            provider_id: Unique identifier for the provider
            config: Optional configuration dictionary
        """
        self.id = provider_id
        self.config = config or {}
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._stores: Dict[str, BaseCacheStore] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def create_store(self,
                    store_id: str,
                    max_size: Optional[int] = None,
                    ttl: Optional[int] = None,
                    **kwargs) -> BaseCacheStore:
        """Create a new cache store.
        
        Args:
            store_id: Unique identifier for the store
            max_size: Maximum number of items to store
            ttl: Time-to-live in seconds
            **kwargs: Additional store configuration
            
        Returns:
            Created cache store instance
        """
        try:
            config = CacheConfig(max_size=max_size, ttl=ttl, **kwargs)
            store = self._create_store(store_id, config)
            self._stores[store_id] = store
            logger.info(f"Created cache store {store_id} with provider {self.id}")
            return store
        except Exception as e:
            logger.error(f"Failed to create cache store {store_id} with provider {self.id}: {e}")
            raise
    
    @abstractmethod
    def _create_store(self,
                     store_id: str,
                     config: CacheConfig) -> BaseCacheStore:
        """Create a new cache store instance.
        
        Args:
            store_id: Unique identifier for the store
            config: Cache store configuration
            
        Returns:
            Created cache store instance
        """
        pass
    
    def get_store(self, store_id: str) -> Optional[BaseCacheStore]:
        """Get a store by ID.
        
        Args:
            store_id: Store ID
            
        Returns:
            Store instance or None if not found
        """
        return self._stores.get(store_id)
    
    def remove_store(self, store_id: str) -> bool:
        """Remove a store.
        
        Args:
            store_id: Store ID
            
        Returns:
            True if store was removed, False otherwise
        """
        try:
            if store_id not in self._stores:
                return False
            
            store = self._stores[store_id]
            store.cleanup()
            del self._stores[store_id]
            logger.info(f"Removed cache store {store_id} from provider {self.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove cache store {store_id} from provider {self.id}: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information.
        
        Returns:
            Dictionary containing provider information
        """
        return {
            'id': self.id,
            'type': type(self).__name__,
            'created': self.created,
            'modified': self.modified,
            'config': self.config,
            'store_count': len(self._stores)
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            return {
                'total_stores': len(self._stores),
                'store_types': {
                    store_type: len([s for s in self._stores.values() if isinstance(s, store_type)])
                    for store_type in set(type(s) for s in self._stores.values())
                }
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 