"""
Unified service system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from .base_service import BaseServiceTool
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class UnifiedService(BaseServiceTool[T]):
    """Unified service with integrated functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize unified service.
        
        Args:
            config: Optional service configuration
        """
        super().__init__(config or {})
        self.providers = {}
        self.handlers = {}
    
    def register_provider(self, name: str, provider: Any) -> None:
        """Register a service provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
        self._record_history(
            "register_provider",
            details={"provider_name": name}
        )
    
    def register_handler(self, name: str, handler: Any) -> None:
        """Register a service handler.
        
        Args:
            name: Handler name
            handler: Handler instance
        """
        self.handlers[name] = handler
        self._record_history(
            "register_handler",
            details={"handler_name": name}
        )
    
    async def execute_provider(
        self,
        provider_name: str,
        action: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a provider action.
        
        Args:
            provider_name: Name of the provider
            action: Action to execute
            data: Input data
            context: Optional execution context
            
        Returns:
            Dict[str, Any]: Action results
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider not found: {provider_name}")
        
        try:
            provider = self.providers[provider_name]
            result = await provider.execute(action, data, context)
            
            self._record_history(
                "execute_provider",
                details={
                    "provider_name": provider_name,
                    "action": action,
                    "result": result
                }
            )
            
            return result
            
        except Exception as e:
            self.error_handler.handle_error(
                "PROVIDER_ERROR",
                f"Provider execution failed: {e}",
                details={
                    "provider_name": provider_name,
                    "action": action,
                    "data": data
                }
            )
            raise
    
    async def handle_event(
        self,
        handler_name: str,
        event: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle an event.
        
        Args:
            handler_name: Name of the handler
            event: Event data
            context: Optional handling context
            
        Returns:
            Dict[str, Any]: Handling results
        """
        if handler_name not in self.handlers:
            raise ValueError(f"Handler not found: {handler_name}")
        
        try:
            handler = self.handlers[handler_name]
            result = await handler.handle(event, context)
            
            self._record_history(
                "handle_event",
                details={
                    "handler_name": handler_name,
                    "event": event,
                    "result": result
                }
            )
            
            return result
            
        except Exception as e:
            self.error_handler.handle_error(
                "HANDLER_ERROR",
                f"Event handling failed: {e}",
                details={
                    "handler_name": handler_name,
                    "event": event
                }
            )
            raise
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information.
        
        Returns:
            Dict[str, Any]: Service information
        """
        return {
            "providers": list(self.providers.keys()),
            "handlers": list(self.handlers.keys()),
            "state_count": len(self.state),
            "history_count": len(self.history)
        }
    
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
