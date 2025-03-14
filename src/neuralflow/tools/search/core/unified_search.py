"""
Unified search system for NeuralFlow.
"""

from typing import Dict, Any, Optional, List, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from .base_search import SearchTool
from ...utils.error.base_handler import BaseErrorHandler
from ...utils.logging.base_manager import BaseLogManager
from ...monitoring.core.unified_monitor import UnifiedMonitor, MonitoringConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class SearchConfig(BaseModel):
    """Search configuration."""
    max_results: int = 100
    min_score: float = 0.5
    cache_enabled: bool = True
    parallel_search: bool = True
    monitor_enabled: bool = True

class SearchResult(BaseModel):
    """Search result model."""
    id: str
    score: float
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class UnifiedSearch(SearchTool):
    """Unified search with integrated functionality."""
    
    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        error_handler: Optional[BaseErrorHandler] = None,
        log_manager: Optional[BaseLogManager] = None
    ):
        """Initialize unified search.
        
        Args:
            config: Optional search configuration
            error_handler: Optional error handler
            log_manager: Optional log manager
        """
        super().__init__()
        self.config = config or SearchConfig()
        self.error_handler = error_handler or BaseErrorHandler()
        self.log_manager = log_manager or BaseLogManager()
        self.monitor = UnifiedMonitor(MonitoringConfig(
            monitor_id="search",
            monitor_type="search_system"
        )) if self.config.monitor_enabled else None
        self.providers = {}
        self.handlers = {}
    
    def register_provider(self, name: str, provider: Any) -> None:
        """Register a search provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider
    
    def register_handler(self, name: str, handler: Any) -> None:
        """Register a search handler.
        
        Args:
            name: Handler name
            handler: Handler instance
        """
        self.handlers[name] = handler
    
    async def search(
        self,
        query: str,
        provider: str = "default",
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Search using the specified provider.
        
        Args:
            query: Search query
            provider: Provider to use
            filters: Optional search filters
            limit: Optional result limit
            
        Returns:
            List of search results
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            # Apply configuration limits
            limit = min(limit or self.config.max_results, self.config.max_results)
            
            # Execute search
            results = await self.providers[provider].search(
                query=query,
                filters=filters,
                limit=limit
            )
            
            # Filter by minimum score
            filtered_results = [
                result for result in results
                if result.score >= self.config.min_score
            ]
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "search_executed",
                    "query": query,
                    "provider": provider,
                    "total_results": len(results),
                    "filtered_results": len(filtered_results),
                    "timestamp": datetime.now().isoformat()
                })
            
            return filtered_results
            
        except Exception as e:
            self.error_handler.handle_error(
                "SEARCH_ERROR",
                f"Search failed: {e}",
                details={
                    "query": query,
                    "provider": provider,
                    "filters": filters
                }
            )
            raise
    
    async def index(
        self,
        items: List[Dict[str, Any]],
        provider: str = "default"
    ) -> bool:
        """Index items for search.
        
        Args:
            items: Items to index
            provider: Provider to use
            
        Returns:
            True if successful
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            success = await self.providers[provider].index(items)
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "items_indexed",
                    "provider": provider,
                    "items_count": len(items),
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                })
            
            return success
            
        except Exception as e:
            self.error_handler.handle_error(
                "INDEX_ERROR",
                f"Indexing failed: {e}",
                details={
                    "provider": provider,
                    "items_count": len(items)
                }
            )
            raise
    
    async def delete_index(
        self,
        provider: str = "default"
    ) -> bool:
        """Delete search index.
        
        Args:
            provider: Provider to use
            
        Returns:
            True if successful
        """
        if provider not in self.providers:
            raise ValueError(f"Provider not found: {provider}")
        
        try:
            success = await self.providers[provider].delete_index()
            
            if self.monitor:
                await self.monitor.record_event({
                    "type": "index_deleted",
                    "provider": provider,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                })
            
            return success
            
        except Exception as e:
            self.error_handler.handle_error(
                "INDEX_ERROR",
                f"Failed to delete index: {e}",
                details={"provider": provider}
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
