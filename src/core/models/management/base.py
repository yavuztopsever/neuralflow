"""
Base model interfaces for LangGraph.
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model."""
        pass
    
    @abstractmethod
    async def predict(self, inputs: Any) -> Any:
        """Make predictions using the model."""
        pass
    
    @abstractmethod
    async def train(self, data: List[Any], **kwargs) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    async def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @abstractmethod
    async def load(self, path: str) -> None:
        """Load the model from disk."""
        pass

class ModelConfig:
    """Configuration for a model."""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')
        self.description = kwargs.get('description', '')
        self.version = kwargs.get('version', '1.0.0')
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['BaseModel', 'ModelConfig'] 