"""
Base embedding model interfaces for LangGraph.
"""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime

from core.models.base_model import BaseNamedModel

class BaseEmbedding(BaseNamedModel, ABC):
    """Base class for embedding models."""
    
    def __init__(self, name: str, config: Dict[str, Any], description: Optional[str] = None):
        """Initialize the embedding model.
        
        Args:
            name: Name of the embedding model
            config: Configuration parameters
            description: Optional description of the model
        """
        super().__init__(name=name, description=description)
        self.config = config
        self.last_used: Optional[datetime] = None
        self.dimensions = config.get('dimensions', 1536)
    
    @abstractmethod
    async def embed(
        self,
        text: Union[str, List[str]],
        **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for input text."""
        self.last_used = datetime.now()
        pass
    
    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs: Any
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        self.last_used = datetime.now()
        pass

class EmbeddingConfig(BaseNamedModel):
    """Configuration for an embedding model."""
    
    def __init__(self, name: str, model_type: str, **kwargs):
        super().__init__(name=name)
        self.model_type = model_type
        self.dimensions = kwargs.get('dimensions', 1536)
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['BaseEmbedding', 'EmbeddingConfig'] 