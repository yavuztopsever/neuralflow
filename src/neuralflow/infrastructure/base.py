"""
Base infrastructure manager for NeuralFlow.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .config.core.base_config import InfrastructureConfig
from .logging.core.base_logger import setup_logging
from .utils.core.base_utils import setup_directories

logger = logging.getLogger(__name__)

class InfrastructureManager:
    """Manages infrastructure components."""
    
    def __init__(self, config: Optional[InfrastructureConfig] = None):
        """Initialize infrastructure manager.
        
        Args:
            config: Optional infrastructure configuration
        """
        self.config = config or InfrastructureConfig()
        
        # Setup components
        self._setup_infrastructure()
    
    def _setup_infrastructure(self):
        """Setup infrastructure components."""
        try:
            # Setup logging
            setup_logging(self.config.logging)
            
            # Setup directories
            setup_directories([
                self.config.cache_dir,
                self.config.temp_dir
            ])
            
            logger.info("Infrastructure setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup infrastructure: {e}")
            raise
    
    def cleanup(self):
        """Clean up infrastructure resources."""
        try:
            # Cleanup temp directories
            if Path(self.config.temp_dir).exists():
                import shutil
                shutil.rmtree(self.config.temp_dir)
            
            logger.info("Infrastructure cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup infrastructure: {e}")
            raise
