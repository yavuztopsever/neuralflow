"""
Base interface for Language Models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from core.models.base_model import BaseNamedModel

class BaseLLM(BaseNamedModel, ABC):
    """Base class for Language Models."""
    
    def __init__(self, name: str, config: Dict[str, Any], description: Optional[str] = None):
        """Initialize the LLM with configuration.
        
        Args:
            name: Name of the LLM
            config: Configuration parameters for the LLM
            description: Optional description of the LLM
        """
        super().__init__(name=name, description=description)
        self.config = config
        self.last_used: Optional[datetime] = None
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt for text generation
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        self.last_used = datetime.now()
        pass
        
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        self.last_used = datetime.now()
        pass

class LLMConfig(BaseNamedModel):
    """Configuration for a language model."""
    
    def __init__(self, name: str, model_type: str, **kwargs):
        super().__init__(name=name)
        self.model_type = model_type
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.temperature = kwargs.get('temperature', 0.7)
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['BaseLLM', 'LLMConfig']
