"""
Search registry for NeuralFlow.
"""

from typing import Dict, Any, Optional
import logging
from .unified_search import UnifiedSearch

logger = logging.getLogger(__name__)

class SearchRegistry:
    """Registry for managing search instances."""
    
    _instance = None
    _searches: Dict[str, UnifiedSearch] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_search(
        self,
        name: str,
        search: UnifiedSearch,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a search instance.
        
        Args:
            name: Search name
            search: Search instance
            config: Optional search configuration
        """
        if name in self._searches:
            logger.warning(f"Search already registered: {name}")
            return
        
        self._searches[name] = search
        if config:
            search.update_config(config)
        
        logger.info(f"Search registered: {name}")
    
    def get_search(self, name: str) -> Optional[UnifiedSearch]:
        """Get a search instance by name.
        
        Args:
            name: Search name
            
        Returns:
            Optional[UnifiedSearch]: Search if found
        """
        return self._searches.get(name)
    
    def list_searches(self) -> Dict[str, Dict[str, Any]]:
        """List all registered searches.
        
        Returns:
            Dict[str, Dict[str, Any]]: Search information
        """
        return {
            name: search.get_search_info()
            for name, search in self._searches.items()
        }
    
    def cleanup(self):
        """Clean up all searches."""
        for search in self._searches.values():
            search.cleanup()
        self._searches = {}
