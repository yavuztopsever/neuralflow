"""
Service registry for NeuralFlow.
"""

from typing import Dict, Any, Optional, Type
import logging
from .unified_service import UnifiedService

logger = logging.getLogger(__name__)

class ServiceRegistry:
    """Registry for managing services."""
    
    _instance = None
    _services: Dict[str, UnifiedService] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_service(
        self,
        name: str,
        service: UnifiedService,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a service.
        
        Args:
            name: Service name
            service: Service instance
            config: Optional service configuration
        """
        if name in self._services:
            logger.warning(f"Service already registered: {name}")
            return
        
        self._services[name] = service
        if config:
            service.update_config(config)
        
        logger.info(f"Service registered: {name}")
    
    def get_service(self, name: str) -> Optional[UnifiedService]:
        """Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Optional[UnifiedService]: Service if found
        """
        return self._services.get(name)
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services.
        
        Returns:
            Dict[str, Dict[str, Any]]: Service information
        """
        return {
            name: service.get_service_info()
            for name, service in self._services.items()
        }
    
    def cleanup(self):
        """Clean up all services."""
        for service in self._services.values():
            service.cleanup()
        self._services = {}
