"""
Base logging utilities for the LangGraph project.
This module provides base logging capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
import json
import os
from pathlib import Path
from ..models.base_model import BaseMetadataModel

logger = logging.getLogger(__name__)

class BaseLogEntry(BaseMetadataModel):
    """Base log entry model."""
    level: str
    message: str
    extra: Optional[Dict[str, Any]] = None
    exc_info: Optional[str] = None

class BaseLogManager:
    """Base manager for logging."""
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        log_format: Optional[str] = None
    ):
        """
        Initialize the logging manager.
        
        Args:
            log_dir: Optional directory for storing logs
            log_level: Logging level
            log_format: Optional log format string
        """
        self.logs: List[BaseLogEntry] = []
        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")
        self.log_level = getattr(logging, log_level.upper())
        self.log_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._ensure_log_dir()
        self._setup_logging()
    
    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.log_dir, "app.log"))
            ]
        )
        
        # Log initialization
        self.logger.info(f"Logging initialized at level: {self.log_level}")
    
    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> BaseLogEntry:
        """
        Log a message.
        
        Args:
            level: Log level
            message: Log message
            extra: Optional extra data
            exc_info: Optional exception info
            
        Returns:
            BaseLogEntry: Created log entry
        """
        try:
            # Create log entry
            log_entry = BaseLogEntry(
                level=level,
                message=message,
                extra=extra,
                exc_info=str(exc_info) if exc_info else None
            )
            
            # Add to logs list
            self.logs.append(log_entry)
            
            # Log using appropriate level
            log_func = getattr(self.logger, level.lower())
            log_func(message, extra=extra, exc_info=exc_info)
            
            return log_entry
            
        except Exception as e:
            self.logger.error(f"Failed to log message: {e}")
            raise
    
    def get_logs(
        self,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[BaseLogEntry]:
        """
        Get logs.
        
        Args:
            level: Optional log level to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List[BaseLogEntry]: List of log entries
        """
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        return logs
    
    def save_logs(self, filename: Optional[str] = None) -> None:
        """
        Save logs to file.
        
        Args:
            filename: Optional filename to save to
        """
        try:
            if not filename:
                filename = os.path.join(self.log_dir, "logs.json")
            
            # Convert logs to dict
            log_dict = {
                "logs": [
                    {
                        "level": log.level,
                        "message": log.message,
                        "timestamp": log.timestamp.isoformat(),
                        "extra": log.extra,
                        "exc_info": log.exc_info
                    }
                    for log in self.logs
                ]
            }
            
            # Save to file
            with open(filename, "w") as f:
                json.dump(log_dict, f, indent=2)
            
            self.logger.info(f"Logs saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save logs: {e}")
            raise
    
    def load_logs(self, filename: Optional[str] = None) -> None:
        """
        Load logs from file.
        
        Args:
            filename: Optional filename to load from
        """
        try:
            if not filename:
                filename = os.path.join(self.log_dir, "logs.json")
            
            if not os.path.exists(filename):
                self.logger.warning(f"Log file not found: {filename}")
                return
            
            # Load from file
            with open(filename, "r") as f:
                log_dict = json.load(f)
            
            # Convert to logs
            self.logs = [
                BaseLogEntry(
                    level=log["level"],
                    message=log["message"],
                    timestamp=datetime.fromisoformat(log["timestamp"]),
                    extra=log.get("extra"),
                    exc_info=log.get("exc_info")
                )
                for log in log_dict["logs"]
            ]
            
            self.logger.info(f"Logs loaded from {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to load logs: {e}")
            raise
    
    def clear_logs(
        self,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Clear logs.
        
        Args:
            level: Optional log level to clear
            start_time: Optional start time to clear from
            end_time: Optional end time to clear to
        """
        if level:
            self.logs = [log for log in self.logs if log.level != level]
        elif start_time and end_time:
            self.logs = [
                log for log in self.logs
                if log.timestamp < start_time or log.timestamp > end_time
            ]
        else:
            self.logs = []
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get log summary.
        
        Returns:
            Dict[str, Any]: Log summary
        """
        return {
            "total_logs": len(self.logs),
            "log_levels": {
                level: len([log for log in self.logs if log.level == level])
                for level in set(log.level for log in self.logs)
            },
            "latest_log": self.logs[-1] if self.logs else None
        }
    
    def reset(self) -> None:
        """Reset logging manager."""
        self.logs = []
        self.logger.info("Logging manager reset")

__all__ = ['BaseLogManager', 'BaseLogEntry'] 