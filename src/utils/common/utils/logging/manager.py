"""
Logging configuration utilities for the LangGraph application.
This module provides functionality for managing application logging.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager

logger = logging.getLogger(__name__)

class LogManager:
    """Manages application logging configuration."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the logging manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.text_processor = TextProcessor()
        self._initialize_logging()
    
    def _initialize_logging(self):
        """Initialize logging configuration."""
        try:
            # Create logs directory
            log_dir = Path(self.config.get('storage.logs_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self._get_log_level())
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Add console handler
            self._add_console_handler(root_logger)
            
            # Add file handler
            self._add_file_handler(root_logger, log_dir)
            
            logger.info("Initialized logging configuration")
        except Exception as e:
            print(f"Failed to initialize logging: {e}")
            raise
    
    def _get_log_level(self) -> int:
        """Get the configured log level.
        
        Returns:
            Logging level as integer
        """
        level_str = self.config.get('app.log_level', 'INFO').upper()
        return getattr(logging, level_str, logging.INFO)
    
    def _add_console_handler(self, logger: logging.Logger):
        """Add console handler to logger.
        
        Args:
            logger: Logger instance to configure
        """
        try:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._get_log_level())
            console_handler.setFormatter(self._get_formatter())
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"Failed to add console handler: {e}")
            raise
    
    def _add_file_handler(self, logger: logging.Logger, log_dir: Path):
        """Add file handler to logger.
        
        Args:
            logger: Logger instance to configure
            log_dir: Directory for log files
        """
        try:
            # Create log file path
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = log_dir / f"app_{timestamp}.log"
            
            # Configure file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('logging.max_file_size', 10 * 1024 * 1024),  # 10MB
                backupCount=self.config.get('logging.backup_count', 5)
            )
            file_handler.setLevel(self._get_log_level())
            file_handler.setFormatter(self._get_formatter())
            
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to add file handler: {e}")
            raise
    
    def _get_formatter(self) -> logging.Formatter:
        """Get the log formatter.
        
        Returns:
            Logging formatter instance
        """
        try:
            format_str = self.config.get(
                'logging.format',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.Formatter(format_str)
        except Exception as e:
            print(f"Failed to create formatter: {e}")
            return logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        try:
            logger = logging.getLogger(name)
            logger.setLevel(self._get_log_level())
            return logger
        except Exception as e:
            print(f"Failed to get logger {name}: {e}")
            return logging.getLogger(name)
    
    def set_log_level(self, level: Union[str, int]):
        """Set the logging level.
        
        Args:
            level: Logging level (string or integer)
        """
        try:
            if isinstance(level, str):
                level = getattr(logging, level.upper(), logging.INFO)
            
            # Update root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            
            # Update all handlers
            for handler in root_logger.handlers:
                handler.setLevel(level)
            
            logger.info(f"Set logging level to {logging.getLevelName(level)}")
        except Exception as e:
            logger.error(f"Failed to set log level: {e}")
            raise
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about logging configuration.
        
        Returns:
            Dictionary containing logging statistics
        """
        try:
            root_logger = logging.getLogger()
            return {
                'level': logging.getLevelName(root_logger.getEffectiveLevel()),
                'handlers': len(root_logger.handlers),
                'handlers_info': [
                    {
                        'type': type(handler).__name__,
                        'level': logging.getLevelName(handler.level),
                        'formatter': type(handler.formatter).__name__ if handler.formatter else None
                    }
                    for handler in root_logger.handlers
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get log stats: {e}")
            return {}
    
    def rotate_logs(self) -> bool:
        """Rotate log files.
        
        Returns:
            True if rotation was successful, False otherwise
        """
        try:
            for handler in logging.getLogger().handlers:
                if isinstance(handler, RotatingFileHandler):
                    handler.doRollover()
            logger.info("Rotated log files")
            return True
        except Exception as e:
            logger.error(f"Failed to rotate logs: {e}")
            return False
    
    def clear_logs(self, days: int = 30) -> int:
        """Clear old log files.
        
        Args:
            days: Number of days to keep logs
            
        Returns:
            Number of files removed
        """
        try:
            log_dir = Path(self.config.get('storage.logs_dir', 'logs'))
            if not log_dir.exists():
                return 0
            
            cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
            removed = 0
            
            for log_file in log_dir.glob('*.log'):
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
                    removed += 1
            
            logger.info(f"Cleared {removed} old log files")
            return removed
        except Exception as e:
            logger.error(f"Failed to clear logs: {e}")
            return 0 