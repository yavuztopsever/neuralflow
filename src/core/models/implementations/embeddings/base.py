"""
Base embedding model interfaces for LangGraph.
"""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from ...models.base import BaseModel

class BaseEmbedding(BaseModel):
    """Base class for embedding models."""
    
    @abstractmethod
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for input text."""
        pass
    
    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs: Any
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

class EmbeddingConfig:
    """Configuration for an embedding model."""
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', '')
        self.model_type = kwargs.get('model_type', '')
        self.dimensions = kwargs.get('dimensions', 1536)
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['BaseEmbedding', 'EmbeddingConfig'] 