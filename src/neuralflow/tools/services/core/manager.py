"""
Service manager for NeuralFlow.
"""

from typing import Dict, Any, Optional, List
import logging
from .registry import ServiceRegistry
from .unified_service import UnifiedService

logger = logging.getLogger(__name__)

class ServiceManager:
    """Manager for coordinating services."""
    
    def __init__(self):
        """Initialize service manager."""
        self.registry = ServiceRegistry()
    
    def create_service(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> UnifiedService:
        """Create and register a new service.
        
        Args:
            name: Service name
            config: Optional service configuration
            
        Returns:
            UnifiedService: Created service
        """
        service = UnifiedService(config)
        self.registry.register_service(name, service, config)
        return service
    
    def get_service(self, name: str) -> Optional[UnifiedService]:
        """Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Optional[UnifiedService]: Service if found
        """
        return self.registry.get_service(name)
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all services.
        
        Returns:
            Dict[str, Dict[str, Any]]: Service information
        """
        return self.registry.list_services()
    
    def cleanup(self):
        """Clean up all services."""
        self.registry.cleanup()
