"""
Base service for the LangGraph project.
This module provides base service capabilities.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for generic state

class BaseHistoryEntry(BaseModel):
    """Base history entry model."""
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str
    details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseService(Generic[T]):
    """Base service with common functionality."""
    
    def __init__(self):
        """Initialize the base service."""
        self.history: List[BaseHistoryEntry] = []
        self.state: Dict[str, T] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def record_history(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an action in history.
        
        Args:
            action: Action performed
            details: Optional action details
            metadata: Optional metadata
        """
        entry = BaseHistoryEntry(
            action=action,
            details=details,
            metadata=metadata
        )
        self.history.append(entry)
        
        # Log action
        self.logger.info(
            f"Action recorded: {action}",
            extra={
                "action": action,
                "details": details,
                "metadata": metadata
            }
        )
    
    def get_history(
        self,
        filter_key: Optional[str] = None,
        filter_value: Optional[Any] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[BaseHistoryEntry]:
        """
        Get filtered history.
        
        Args:
            filter_key: Optional key to filter by in details
            filter_value: Optional value to match in details[filter_key]
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List[BaseHistoryEntry]: Filtered history entries
        """
        filtered = self.history
        
        if filter_key and filter_value is not None:
            filtered = [
                entry for entry in filtered
                if entry.details and entry.details.get(filter_key) == filter_value
            ]
        
        if start_time:
            filtered = [entry for entry in filtered if entry.timestamp >= start_time]
        
        if end_time:
            filtered = [entry for entry in filtered if entry.timestamp <= end_time]
        
        return filtered
    
    def clear_history(
        self,
        filter_key: Optional[str] = None,
        filter_value: Optional[Any] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Clear history with optional filters.
        
        Args:
            filter_key: Optional key to filter by in details
            filter_value: Optional value to match in details[filter_key]
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
        """
        if not any([filter_key, start_time, end_time]):
            self.history = []
            self.logger.info("History cleared")
            return
        
        if filter_key and filter_value is not None:
            self.history = [
                entry for entry in self.history
                if not entry.details or entry.details.get(filter_key) != filter_value
            ]
        
        if start_time and end_time:
            self.history = [
                entry for entry in self.history
                if entry.timestamp < start_time or entry.timestamp > end_time
            ]
        
        self.logger.info("History partially cleared with filters")
    
    def get_state(self, key: str) -> Optional[T]:
        """
        Get state by key.
        
        Args:
            key: State key
            
        Returns:
            Optional[T]: State if found
        """
        return self.state.get(key)
    
    def set_state(self, key: str, value: T) -> None:
        """
        Set state for key.
        
        Args:
            key: State key
            value: State value
        """
        self.state[key] = value
        self.record_history(
            "set_state",
            details={"key": key}
        )
    
    def delete_state(self, key: str) -> None:
        """
        Delete state for key.
        
        Args:
            key: State key
        """
        if key in self.state:
            del self.state[key]
            self.record_history(
                "delete_state",
                details={"key": key}
            )
    
    def reset(self) -> None:
        """Reset service state."""
        self.state = {}
        self.history = []
        self.logger.info("Service reset")

__all__ = ['BaseService', 'BaseHistoryEntry'] 