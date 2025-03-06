"""
Logging handlers for LangGraph.
"""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

class LogHandler:
    """Base class for log handlers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with the given configuration."""
        logger = logging.getLogger(self.config.get('name', 'langgraph'))
        logger.setLevel(self.config.get('level', logging.INFO))
        
        # Create handlers
        handlers = self._create_handlers()
        for handler in handlers:
            logger.addHandler(handler)
        
        return logger
    
    def _create_handlers(self) -> list[logging.Handler]:
        """Create log handlers based on configuration."""
        handlers = []
        
        # Console handler
        if self.config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config.get('level', logging.INFO))
            console_handler.setFormatter(
                logging.Formatter(self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            )
            handlers.append(console_handler)
        
        # File handler
        if self.config.get('file'):
            file_path = Path(self.config['file'])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(self.config.get('level', logging.INFO))
            file_handler.setFormatter(
                logging.Formatter(self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            )
            handlers.append(file_handler)
        
        return handlers
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message."""
        self.logger.critical(message, extra=kwargs)

class LogConfig:
    """Configuration for logging."""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'langgraph')
        self.level = kwargs.get('level', 'INFO')
        self.format = kwargs.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console = kwargs.get('console', True)
        self.file = kwargs.get('file')
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['LogHandler', 'LogConfig']
