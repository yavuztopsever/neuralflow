"""
Unified monitoring system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from .base_monitor import BaseMonitor, MonitoringConfig
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class UnifiedMonitor(BaseMonitor):
    """Unified monitoring with integrated functionality."""
    
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize unified monitoring.
        
        Args:
            config: Optional monitoring configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        super().__init__(config or MonitoringConfig(
            monitor_id="unified",
            monitor_type="unified"
        ))
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        self.providers = {}
        self.handlers = {}
    
    def register_provider(self, name: str, provider: Any) -> None:
        """Register a monitoring provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
    
    def register_handler(self, name: str, handler: Any) -> None:
        """Register a monitoring handler.
        
        Args:
            name: Handler name
            handler: Handler instance
        """
        self.handlers[name] = handler
    
    async def record_metric(
        self,
        metric: Dict[str, Any],
        provider: str = "default"
    ) -> bool:
        """Record a metric.
        
        Args:
            metric: Metric data
            provider: Provider to use
            
        Returns:
            True if successful
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            return await self.providers[provider].record_metric(metric)
        except Exception as e:
            self.error_handler.handle_error(
                "METRIC_ERROR",
                f"Failed to record metric: {e}",
                details={"metric": metric, "provider": provider}
            )
            raise
    
    async def record_event(
        self,
        event: Dict[str, Any],
        provider: str = "default"
    ) -> bool:
        """Record an event.
        
        Args:
            event: Event data
            provider: Provider to use
            
        Returns:
            True if successful
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            return await self.providers[provider].record_event(event)
        except Exception as e:
            self.error_handler.handle_error(
                "EVENT_ERROR",
                f"Failed to record event: {e}",
                details={"event": event, "provider": provider}
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
