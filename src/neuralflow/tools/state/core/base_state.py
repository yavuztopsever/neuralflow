"""
Base state management system for NeuralFlow.
Provides unified state and context management capabilities.
"""

import logging
import threading
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel

from ..storage.base import BaseStorage, StorageConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class StateConfig:
    """Configuration for state management systems."""
    
    def __init__(self,
                 state_id: str,
                 state_type: str,
                 storage_config: Optional[StorageConfig] = None,
                 ttl: Optional[int] = None,
                 max_size: Optional[int] = None,
                 **kwargs):
        """Initialize state configuration.
        
        Args:
            state_id: Unique identifier for the state manager
            state_type: Type of state manager (state, context, memory)
            storage_config: Optional storage configuration
            ttl: Optional time-to-live in seconds
            max_size: Optional maximum size in bytes
            **kwargs: Additional configuration parameters
        """
        self.id = state_id
        self.type = state_type
        self.storage_config = storage_config
        self.ttl = ttl
        self.max_size = max_size
        self.parameters = kwargs
        self.metadata = {}

class BaseState(BaseModel):
    """Base state model."""
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    ttl: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseContext(BaseModel):
    """Base context model."""
    id: str
    type: str
    data: Dict[str, Any]
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    timestamp: datetime
    ttl: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseStateManager(ABC, Generic[T]):
    """Base class for all state management implementations."""
    
    def __init__(self, config: StateConfig):
        """Initialize the state manager.
        
        Args:
            config: State configuration
        """
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._storage = self._initialize_storage()
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self._last_access: Dict[str, datetime] = {}
        self._creation_time: Dict[str, datetime] = {}
        self._size = 0
    
    def _initialize_storage(self) -> Optional[BaseStorage]:
        """Initialize storage if configured."""
        if self.config.storage_config:
            return BaseStorage(self.config.storage_config)
        return None
    
    async def get_state(self, state_id: str) -> Optional[BaseState]:
        """Get state by ID.
        
        Args:
            state_id: State ID
            
        Returns:
            State if found, None otherwise
        """
        try:
            # Check cache
            state = self._cache_get(state_id)
            if state:
                return state
            
            # Check storage
            if self._storage:
                state = await self._storage.get(state_id)
                if state:
                    self._cache_set(state_id, state)
                return state
            
            return None
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None
    
    async def set_state(self, state: BaseState) -> bool:
        """Set state.
        
        Args:
            state: State to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update cache
            self._cache_set(state.id, state)
            
            # Update storage
            if self._storage:
                await self._storage.store(state)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set state: {e}")
            return False
    
    async def delete_state(self, state_id: str) -> bool:
        """Delete state.
        
        Args:
            state_id: State ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from cache
            self._cache_delete(state_id)
            
            # Remove from storage
            if self._storage:
                await self._storage.delete(state_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            return False
    
    async def get_context(self, context_id: str) -> Optional[BaseContext]:
        """Get context by ID.
        
        Args:
            context_id: Context ID
            
        Returns:
            Context if found, None otherwise
        """
        try:
            # Check cache
            context = self._cache_get(context_id)
            if context:
                return context
            
            # Check storage
            if self._storage:
                context = await self._storage.get(context_id)
                if context:
                    self._cache_set(context_id, context)
                return context
            
            return None
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return None
    
    async def set_context(self, context: BaseContext) -> bool:
        """Set context.
        
        Args:
            context: Context to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update cache
            self._cache_set(context.id, context)
            
            # Update storage
            if self._storage:
                await self._storage.store(context)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set context: {e}")
            return False
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete context.
        
        Args:
            context_id: Context ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from cache
            self._cache_delete(context_id)
            
            # Remove from storage
            if self._storage:
                await self._storage.delete(context_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete context: {e}")
            return False
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._cache_lock:
            if key in self._cache:
                self._last_access[key] = datetime.now()
                return self._cache[key]
            return None
    
    def _cache_set(self, key: str, value: Any) -> None:
        """Set cache value."""
        with self._cache_lock:
            # Check size limit
            if self.config.max_size:
                value_size = len(str(value))
                if self._size + value_size > self.config.max_size:
                    self._cleanup_cache()
                    if self._size + value_size > self.config.max_size:
                        logger.warning("Cache size limit exceeded")
                        return
                self._size += value_size
            
            self._cache[key] = value
            now = datetime.now()
            self._last_access[key] = now
            self._creation_time[key] = now
    
    def _cache_delete(self, key: str) -> bool:
        """Delete cache value."""
        with self._cache_lock:
            if key in self._cache:
                if self.config.max_size:
                    self._size -= len(str(self._cache[key]))
                del self._cache[key]
                del self._last_access[key]
                del self._creation_time[key]
                return True
            return False
    
    def _cleanup_cache(self) -> int:
        """Clean up cache entries."""
        removed = 0
        now = datetime.now()
        
        with self._cache_lock:
            keys = list(self._cache.keys())
            for key in keys:
                # Check TTL
                if self.config.ttl:
                    age = (now - self._creation_time[key]).total_seconds()
                    if age > self.config.ttl:
                        self._cache_delete(key)
                        removed += 1
                        continue
                
                # Remove oldest entries if size limit exceeded
                if self.config.max_size and self._size > self.config.max_size:
                    oldest_key = min(
                        self._last_access.items(),
                        key=lambda x: x[1]
                    )[0]
                    self._cache_delete(oldest_key)
                    removed += 1
                else:
                    break
        
        return removed
    
    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information.
        
        Returns:
            Dictionary containing manager information
        """
        return {
            'id': self.config.id,
            'type': self.config.type,
            'created': self.created,
            'modified': self.modified,
            'parameters': self.config.parameters,
            'metadata': self.config.metadata,
            'cache_size': len(self._cache),
            'storage': self._storage.get_storage_info() if self._storage else None
        }
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics.
        
        Returns:
            Dictionary containing manager statistics
        """
        try:
            stats = {
                'cache_entries': len(self._cache),
                'cache_size': self._size,
                'last_modified': self.modified
            }
            
            if self._storage:
                stats['storage'] = self._storage.get_storage_stats()
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get manager stats: {e}")
            return {} 