"""
Base provider interfaces for embedding models.
This module provides base classes for embedding model provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime

from models.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

class BaseEmbeddingProvider(ABC):
    """Base class for embedding model providers."""
    
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
        self._models: Dict[str, BaseEmbeddingModel] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def register_model(self, model: BaseEmbeddingModel) -> None:
        """Register a model with the provider.
        
        Args:
            model: Model instance to register
        """
        try:
            self._models[model.id] = model
            logger.info(f"Registered model {model.id} with provider {self.id}")
        except Exception as e:
            logger.error(f"Failed to register model {model.id} with provider {self.id}: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[BaseEmbeddingModel]:
        """Get a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model instance or None if not found
        """
        return self._models.get(model_id)
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if model was removed, False otherwise
        """
        try:
            if model_id not in self._models:
                return False
            
            del self._models[model_id]
            logger.info(f"Removed model {model_id} from provider {self.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove model {model_id} from provider {self.id}: {e}")
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