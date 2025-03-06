"""
Base interface for Language Models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLLM(ABC):
    """Base class for Language Models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM with configuration.
        
        Args:
            config: Configuration parameters for the LLM
        """
        self.config = config
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt for text generation
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        pass
        
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        pass

class LLMConfig:
    """Configuration for a language model."""
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', '')
        self.model_type = kwargs.get('model_type', '')
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.temperature = kwargs.get('temperature', 0.7)
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['BaseLLM', 'LLMConfig']
