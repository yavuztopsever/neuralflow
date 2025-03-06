"""
Main entry point for the NeuralFlow application.
This module initializes and runs the application.
"""

import os
import logging
import uvicorn
from pathlib import Path
from config.manager import ConfigManager
from utils.logging.manager import LogManager
from utils.common.hardware import HardwareManager
from core.api import APIManager
from ui.app import NeuralFlowUI

def main():
    """Initialize and run the application."""
    try:
        # Initialize configuration
        config = ConfigManager()
        
        # Initialize logging
        log_manager = LogManager(config)
        logger = logging.getLogger(__name__)
        
        # Initialize hardware manager
        hardware_manager = HardwareManager()
        hardware_manager.optimize_torch_settings()
        hardware_manager.log_hardware_info()
        
        # Create necessary directories
        storage_dir = Path(config.get('storage_dir', 'storage'))
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API with hardware settings
        api_manager = APIManager(config)
        
        # Initialize UI
        ui = NeuralFlowUI()
        
        # Get server configuration
        host = config.get('server_host', '0.0.0.0')
        port = config.get('server_port', 8000)
        reload = config.get('server_reload', False)
        
        # Run server
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            api_manager.app,
            host=host,
            port=port,
            reload=reload
        )
        
        # Launch UI (this will run in a separate process)
        logger.info("Starting UI application")
        ui.launch(share=True)
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

if __name__ == '__main__':
    main() 