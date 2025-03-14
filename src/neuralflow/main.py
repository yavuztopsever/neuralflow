"""
Main application module for NeuralFlow.
"""

import asyncio
import logging
from typing import Optional

from .config import ApplicationConfig
from .infrastructure.base import InfrastructureManager
from .frontend.base import FrontendManager

logger = logging.getLogger(__name__)

class Application:
    """Main application class."""
    
    def __init__(self, config: Optional[ApplicationConfig] = None):
        """Initialize application.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or ApplicationConfig()
        
        # Initialize managers
        self.infrastructure = InfrastructureManager(self.config.infrastructure)
        self.frontend = FrontendManager(self.config.frontend, self.infrastructure)
    
    async def start(self):
        """Start the application."""
        try:
            # Start components
            await self.frontend.start()
            
            logger.info("Application started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the application."""
        try:
            # Shutdown components
            await self.frontend.shutdown()
            self.infrastructure.cleanup()
            
            logger.info("Application shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed to shutdown application: {e}")
            raise

def run_application():
    """Run the application."""
    try:
        app = Application()
        asyncio.run(app.start())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        asyncio.run(app.shutdown())
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    run_application()
