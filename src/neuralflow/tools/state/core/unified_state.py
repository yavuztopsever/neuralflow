"""
Unified state system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from .base_state import BaseStateManager
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class StateConfig(BaseModel):
    """State configuration."""
    ttl: Optional[float] = None
    max_size: Optional[int] = None
    cache_enabled: bool = True
    persistence_enabled: bool = True

class UnifiedState(BaseStateManager):
    """Unified state with integrated functionality."""
    
    def __init__(
        self,
        config: Optional[StateConfig] = None,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize unified state.
        
        Args:
            config: Optional state configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        super().__init__(config or StateConfig())
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        self.providers = {}
        self.handlers = {}
    
    def register_provider(self, name: str, provider: Any) -> None:
        """Register a state provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
    
    def register_handler(self, name: str, handler: Any) -> None:
        """Register a state handler.
        
        Args:
            name: Handler name
            handler: Handler instance
        """
        self.handlers[name] = handler
    
    async def get_state(
        self,
        state_id: str,
        provider: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get state by ID.
        
        Args:
            state_id: State ID
            provider: Provider to use
            
        Returns:
            State if found
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            return await self.providers[provider].get_state(state_id)
        except Exception as e:
            self.error_handler.handle_error(
                "STATE_ERROR",
                f"Failed to get state: {e}",
                details={"state_id": state_id, "provider": provider}
            )
            raise
    
    async def set_state(
        self,
        state_id: str,
        state: Dict[str, Any],
        provider: str = "default",
        ttl: Optional[float] = None
    ) -> None:
        """Set state.
        
        Args:
            state_id: State ID
            state: State data
            provider: Provider to use
            ttl: Optional time-to-live
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            await self.providers[provider].set_state(state_id, state, ttl)
        except Exception as e:
            self.error_handler.handle_error(
                "STATE_ERROR",
                f"Failed to set state: {e}",
                details={"state_id": state_id, "provider": provider}
            )
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            super().cleanup()
            for provider in self.providers.values():
                if hasattr(provider, "cleanup"):
                    provider.cleanup()
            for handler in self.handlers.values():
                if hasattr(handler, "cleanup"):
                    handler.cleanup()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            )
