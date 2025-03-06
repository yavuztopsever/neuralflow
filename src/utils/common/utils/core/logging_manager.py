"""
Logging utilities specific to the LangGraph project.
These utilities handle logging that is not covered by LangChain's built-in logging.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line: int
    metadata: Optional[Dict[str, Any]] = None

class LogManager:
    """Manages logging for the LangGraph project."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up root logger
        self.logger = logging.getLogger("langgraph")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
        )
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set up file handler
        self.file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "langgraph.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        self.file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(self.file_handler)
        
        # Set up console handler if requested
        if console_output:
            self.console_handler = logging.StreamHandler()
            self.console_handler.setFormatter(self.console_formatter)
            self.logger.addHandler(self.console_handler)
        
        # Initialize log history
        self.log_history: List[LogEntry] = []
        self.max_history_size = 1000

    def log(
        self,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ):
        """Log a message with optional metadata and exception info."""
        # Get caller information
        import inspect
        frame = inspect.currentframe().f_back
        module = frame.f_globals['__name__']
        function = frame.f_code.co_name
        line = frame.f_lineno
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            message=message,
            module=module,
            function=function,
            line=line,
            metadata=metadata
        )
        
        # Add to history
        self.log_history.append(entry)
        if len(self.log_history) > self.max_history_size:
            self.log_history.pop(0)
        
        # Log using appropriate level
        log_func = getattr(self.logger, level.lower())
        if exc_info:
            log_func(message, exc_info=exc_info, extra=metadata or {})
        else:
            log_func(message, extra=metadata or {})

    def debug(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self.log("DEBUG", message, metadata)

    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self.log("INFO", message, metadata)

    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self.log("WARNING", message, metadata)

    def error(self, message: str, metadata: Optional[Dict[str, Any]] = None, exc_info: Optional[Exception] = None):
        """Log an error message."""
        self.log("ERROR", message, metadata, exc_info)

    def critical(self, message: str, metadata: Optional[Dict[str, Any]] = None, exc_info: Optional[Exception] = None):
        """Log a critical message."""
        self.log("CRITICAL", message, metadata, exc_info)

    def get_log_history(
        self,
        level: Optional[str] = None,
        module: Optional[str] = None,
        function: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[LogEntry]:
        """Get log history with optional filters."""
        filtered_logs = self.log_history
        
        if level:
            filtered_logs = [log for log in filtered_logs if log.level == level.upper()]
        
        if module:
            filtered_logs = [log for log in filtered_logs if log.module == module]
        
        if function:
            filtered_logs = [log for log in filtered_logs if log.function == function]
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        return filtered_logs

    def clear_log_history(self):
        """Clear log history."""
        self.log_history.clear()

    def save_log_history(self, filepath: str):
        """Save log history to a file."""
        with open(filepath, 'w') as f:
            json.dump(
                [vars(entry) for entry in self.log_history],
                f,
                indent=2
            )

    def load_log_history(self, filepath: str):
        """Load log history from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.log_history = [
                LogEntry(**entry_data)
                for entry_data in data
            ]

    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about the logs."""
        stats = {
            "total_entries": len(self.log_history),
            "level_counts": {},
            "module_counts": {},
            "function_counts": {},
            "time_range": {
                "start": None,
                "end": None
            }
        }
        
        if self.log_history:
            stats["time_range"]["start"] = self.log_history[0].timestamp
            stats["time_range"]["end"] = self.log_history[-1].timestamp
        
        for entry in self.log_history:
            # Count by level
            stats["level_counts"][entry.level] = stats["level_counts"].get(entry.level, 0) + 1
            
            # Count by module
            stats["module_counts"][entry.module] = stats["module_counts"].get(entry.module, 0) + 1
            
            # Count by function
            stats["function_counts"][entry.function] = stats["function_counts"].get(entry.function, 0) + 1
        
        return stats

def with_logging(logger: Optional[LogManager] = None):
    """Decorator for logging function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                return func(*args, **kwargs)
            
            # Log function entry
            logger.debug(
                f"Entering function: {func.__name__}",
                metadata={
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful exit
                logger.debug(
                    f"Exiting function: {func.__name__}",
                    metadata={"result": str(result)}
                )
                
                return result
                
            except Exception as e:
                # Log error
                logger.error(
                    f"Error in function: {func.__name__}",
                    metadata={
                        "args": str(args),
                        "kwargs": str(kwargs)
                    },
                    exc_info=e
                )
                raise
                
        return wrapper
    return decorator 