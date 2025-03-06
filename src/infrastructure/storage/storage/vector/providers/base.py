"""
Base provider interfaces for vector stores.
This module provides base classes for vector store provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStoreConfig:
    """Configuration for vector stores."""
    
    def __init__(self,
                 dimension: int,
                 metric: str = 'cosine',
                 index_type: str = 'flat',
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, etc.)
            index_type: Index type (flat, ivf, etc.)
            **kwargs: Additional configuration parameters
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.extra_params = kwargs

class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    def __init__(self, store_id: str,
                 config: VectorStoreConfig,
                 **kwargs):
        """Initialize the vector store.
        
        Args:
            store_id: Unique identifier for the store
            config: Vector store configuration
            **kwargs: Additional initialization parameters
        """
        self.id = store_id
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up vector store resources."""
        pass
    
    @abstractmethod
    def add_vectors(self,
                   vectors: List[List[float]],
                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add vectors to the store.
        
        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector
            
        Returns:
            List of vector IDs
        """
        pass
    
    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Get a vector by ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Vector or None if not found
        """
        pass
    
    @abstractmethod
    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Metadata dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def search(self,
              query_vector: List[float],
              k: int = 10,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with scores and metadata
        """
        pass
    
    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            True if vector was deleted, False otherwise
        """
        pass
    
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
                'dimension': self.config.dimension,
                'metric': self.config.metric,
                'index_type': self.config.index_type,
                'extra_params': self.config.extra_params
            }
        }

class BaseVectorStoreProvider(ABC):
    """Base class for vector store providers."""
    
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
        self._stores: Dict[str, BaseVectorStore] = {}
    
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
                    dimension: int,
                    **kwargs) -> BaseVectorStore:
        """Create a new vector store.
        
        Args:
            store_id: Unique identifier for the store
            dimension: Vector dimension
            **kwargs: Additional store configuration
            
        Returns:
            Created vector store instance
        """
        try:
            config = VectorStoreConfig(dimension=dimension, **kwargs)
            store = self._create_store(store_id, config)
            self._stores[store_id] = store
            logger.info(f"Created vector store {store_id} with provider {self.id}")
            return store
        except Exception as e:
            logger.error(f"Failed to create vector store {store_id} with provider {self.id}: {e}")
            raise
    
    @abstractmethod
    def _create_store(self,
                     store_id: str,
                     config: VectorStoreConfig) -> BaseVectorStore:
        """Create a new vector store instance.
        
        Args:
            store_id: Unique identifier for the store
            config: Vector store configuration
            
        Returns:
            Created vector store instance
        """
        pass
    
    def get_store(self, store_id: str) -> Optional[BaseVectorStore]:
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
            logger.info(f"Removed vector store {store_id} from provider {self.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove vector store {store_id} from provider {self.id}: {e}")
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