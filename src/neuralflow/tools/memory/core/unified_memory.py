"""
Unified memory system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from .base_memory import MemoryManager
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager
from ...monitoring.core.unified_monitor import UnifiedMonitor, MonitoringConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class MemoryConfig(BaseModel):
    """Memory configuration."""
    max_items: int = 1000
    ttl: float = 3600.0
    cleanup_interval: float = 300.0
    cache_enabled: bool = True
    persistence_enabled: bool = True
    monitor_enabled: bool = True

class UnifiedMemory(MemoryManager):
    """Unified memory with integrated functionality."""
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize unified memory.
        
        Args:
            config: Optional memory configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        super().__init__(config or MemoryConfig())
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        self.monitor = UnifiedMonitor(MonitoringConfig(
            monitor_id="memory",
            monitor_type="memory_system"
        )) if self.config.monitor_enabled else None
        self.providers = {}
        self.handlers = {}
    
    def register_provider(self, name: str, provider: Any) -> None:
        """Register a memory provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
    
    def register_handler(self, name: str, handler: Any) -> None:
        """Register a memory handler.
        
        Args:
            name: Handler name
            handler: Handler instance
        """
        self.handlers[name] = handler
    
    async def store(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[float] = None,
        provider: str = "default"
    ) -> None:
        """Store a value in memory.
        
        Args:
            key: Memory key
            value: Value to store
            ttl: Optional time-to-live
            provider: Provider to use
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            await self.providers[provider].store(key, value, ttl or self.config.ttl)
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "memory_store",
                    "key": key,
                    "provider": provider,
                    "ttl": ttl,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.error_handler.handle_error(
                "MEMORY_ERROR",
                f"Failed to store value: {e}",
                details={"key": key, "provider": provider}
            )
            raise
    
    async def retrieve(
        self,
        key: str,
        provider: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a value from memory.
        
        Args:
            key: Memory key
            provider: Provider to use
            
        Returns:
            Stored value if found
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            value = await self.providers[provider].retrieve(key)
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "memory_retrieve",
                    "key": key,
                    "provider": provider,
                    "found": value is not None,
                    "timestamp": datetime.now().isoformat()
                })
            
            return value
            
        except Exception as e:
            self.error_handler.handle_error(
                "MEMORY_ERROR",
                f"Failed to retrieve value: {e}",
                details={"key": key, "provider": provider}
            )
            raise
    
    async def delete(
        self,
        key: str,
        provider: str = "default"
    ) -> bool:
        """Delete a value from memory.
        
        Args:
            key: Memory key
            provider: Provider to use
            
        Returns:
            True if deleted
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            deleted = await self.providers[provider].delete(key)
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "memory_delete",
                    "key": key,
                    "provider": provider,
                    "deleted": deleted,
                    "timestamp": datetime.now().isoformat()
                })
            
            return deleted
            
        except Exception as e:
            self.error_handler.handle_error(
                "MEMORY_ERROR",
                f"Failed to delete value: {e}",
                details={"key": key, "provider": provider}
            )
            raise
    
    async def cleanup_expired(self) -> int:
        """Clean up expired items.
        
        Returns:
            Number of items cleaned up
        """
        try:
            total_cleaned = 0
            
            for provider in self.providers.values():
                cleaned = await provider.cleanup_expired()
                total_cleaned += cleaned
            
            if self.monitor:
                await self.monitor.record_metric({
                    "type": "memory_cleanup",
                    "items_cleaned": total_cleaned,
                    "timestamp": datetime.now().isoformat()
                })
            
            return total_cleaned
            
        except Exception as e:
            self.error_handler.handle_error(
                "MEMORY_ERROR",
                f"Failed to clean up expired items: {e}"
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
            if self.monitor:
                self.monitor.cleanup()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            )
