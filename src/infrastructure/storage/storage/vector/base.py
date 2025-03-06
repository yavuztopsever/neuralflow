"""
Base vector storage interfaces for LangGraph.
"""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def add(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(
        self,
        ids: List[str],
        **kwargs: Any
    ) -> bool:
        """Delete vectors by ID."""
        pass
    
    @abstractmethod
    async def update(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        """Update a vector and its metadata."""
        pass

class VectorStoreConfig:
    """Configuration for a vector store."""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')
        self.type = kwargs.get('type', '')
        self.dimensions = kwargs.get('dimensions', 1536)
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['BaseVectorStore', 'VectorStoreConfig']
