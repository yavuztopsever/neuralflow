"""
Frontend manager for NeuralFlow.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import FrontendConfig
from ..infrastructure.base import InfrastructureManager

logger = logging.getLogger(__name__)

class FrontendManager:
    """Manages frontend components."""
    
    def __init__(
        self,
        config: Optional[FrontendConfig] = None,
        infrastructure: Optional[InfrastructureManager] = None
    ):
        """Initialize frontend manager.
        
        Args:
            config: Optional frontend configuration
            infrastructure: Optional infrastructure manager
        """
        self.config = config or FrontendConfig()
        self.infrastructure = infrastructure or InfrastructureManager()
        
        # Initialize components
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application.
        
        Returns:
            FastAPI: Application instance
        """
        app = FastAPI(
            title="NeuralFlow API",
            description="API for NeuralFlow platform",
            version="1.0.0",
            prefix=self.config.api.api_prefix
        )
        
        # Add CORS middleware
        if self.config.api.cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.api.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
        
        return app
    
    async def start(self):
        """Start frontend components."""
        try:
            # Initialize routes
            from .api.core.routes import init_routes
            init_routes(self.app)
            
            logger.info("Frontend started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start frontend: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown frontend components."""
        try:
            logger.info("Frontend shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed to shutdown frontend: {e}")
            raise
