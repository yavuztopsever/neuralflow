"""
Search manager for NeuralFlow.
"""

from typing import Dict, Any, Optional, List
import logging
from .registry import SearchRegistry
from .unified_search import UnifiedSearch, SearchConfig

logger = logging.getLogger(__name__)

class SearchManager:
    """Manager for coordinating search instances."""
    
    def __init__(self):
        """Initialize search manager."""
        self.registry = SearchRegistry()
    
    def create_search(
        self,
        name: str,
        config: Optional[SearchConfig] = None
    ) -> UnifiedSearch:
        """Create and register a new search instance.
        
        Args:
            name: Search name
            config: Optional search configuration
            
        Returns:
            UnifiedSearch: Created search instance
        """
        search = UnifiedSearch(config)
        self.registry.register_search(name, search)
        return search
    
    def get_search(self, name: str) -> Optional[UnifiedSearch]:
        """Get a search instance by name.
        
        Args:
            name: Search name
            
        Returns:
            Optional[UnifiedSearch]: Search if found
        """
        return self.registry.get_search(name)
    
    def list_searches(self) -> Dict[str, Dict[str, Any]]:
        """List all search instances.
        
        Returns:
            Dict[str, Dict[str, Any]]: Search information
        """
        return self.registry.list_searches()
    
    def cleanup(self):
        """Clean up all search instances."""
        self.registry.cleanup()
