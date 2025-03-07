"""
Base provider interfaces for language models.
This module provides base classes for language model provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime

from core.models.base_model import BaseNamedModel
from core.models.implementations.llm.base import BaseLLM

logger = logging.getLogger(__name__)

class BaseLLMProvider(BaseNamedModel, ABC):
    """Base class for language model providers."""
    
    def __init__(self, name: str, provider_id: str,
                 config: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None):
        """Initialize the provider.
        
        Args:
            name: Name of the provider
            provider_id: Unique identifier for the provider
            config: Optional configuration dictionary
            description: Optional description of the provider
        """
        super().__init__(name=name, description=description)
        self.id = provider_id
        self.config = config or {}
        self._models: Dict[str, BaseLLM] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def register_model(self, model: BaseLLM) -> None:
        """Register a model with the provider.
        
        Args:
            model: Model instance to register
        """
        try:
            self._models[model.name] = model
            logger.info(f"Registered model {model.name} with provider {self.name}")
        except Exception as e:
            logger.error(f"Failed to register model {model.name} with provider {self.name}: {e}")
            raise
    
    def get_model(self, model_name: str) -> Optional[BaseLLM]:
        """Get a model by name.
        
        Args:
            model_name: Model name
            
        Returns:
            Model instance or None if not found
        """
        return self._models.get(model_name)
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model.
        
        Args:
            model_name: Model name
            
        Returns:
            True if model was removed, False otherwise
        """
        try:
            if model_name not in self._models:
                return False
            
            del self._models[model_name]
            logger.info(f"Removed model {model_name} from provider {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove model {model_name} from provider {self.name}: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information.
        
        Returns:
            Dictionary containing provider information
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'type': type(self).__name__,
            'config': self.config,
            'model_count': len(self._models)
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            return {
                'total_models': len(self._models),
                'model_types': {
                    model_type: len([m for m in self._models.values() if isinstance(m, model_type)])
                    for model_type in set(type(m) for m in self._models.values())
                }
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 