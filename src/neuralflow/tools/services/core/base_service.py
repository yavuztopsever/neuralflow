"""
Base service tool for NeuralFlow.
Provides unified service capabilities with tool integration.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseTool, ToolConfig, ToolType, ToolResult
from ..storage.base import StorageConfig
from ..utils.error.base_handler import BaseErrorHandler
from ..utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for generic state

class ServiceHistoryEntry(BaseModel):
    """Service history entry model."""
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str
    details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ServiceState(BaseModel):
    """Service state model."""
    key: str
    value: Any
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class BaseServiceTool(BaseTool, Generic[T]):
    """Base service tool with integrated functionality."""
    
    def __init__(
        self,
        config: ToolConfig,
        storage_config: Optional[StorageConfig] = None
    ):
        """Initialize the service tool.
        
        Args:
            config: Tool configuration
            storage_config: Optional storage configuration
        """
        super().__init__(config, storage_config)
        
        # Initialize state and history
        self.history: List[ServiceHistoryEntry] = []
        self.state: Dict[str, ServiceState] = {}
        
        # Initialize components
        self.error_handler = BaseErrorHandler()
        self.log_manager = BaseLogManager()
    
    async def _execute_impl(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the tool implementation.
        
        Args:
            input_data: Input data containing service action and parameters
            context: Optional execution context
            
        Returns:
            Dict[str, Any]: Service action results
        """
        action = input_data.get("action")
        if not action:
            raise ValueError("Service action is required")
        
        try:
            if action == "get_state":
                result = self._get_state(input_data.get("key"))
            elif action == "set_state":
                result = self._set_state(
                    input_data.get("key"),
                    input_data.get("value"),
                    input_data.get("metadata")
                )
            elif action == "delete_state":
                result = self._delete_state(input_data.get("key"))
            elif action == "get_history":
                result = self._get_history(
                    input_data.get("filter_key"),
                    input_data.get("filter_value"),
                    input_data.get("start_time"),
                    input_data.get("end_time")
                )
            elif action == "clear_history":
                result = self._clear_history(
                    input_data.get("filter_key"),
                    input_data.get("filter_value"),
                    input_data.get("start_time"),
                    input_data.get("end_time")
                )
            elif action == "reset":
                result = self._reset()
            else:
                result = await self._execute_service_action(action, input_data, context)
            
            return {
                "action": action,
                "result": result,
                "metadata": self._generate_metadata(action)
            }
            
        except Exception as e:
            self.error_handler.handle_error(
                "SERVICE_ERROR",
                f"Service action failed: {str(e)}",
                details={
                    "action": action,
                    "input_data": input_data
                }
            )
            raise
    
    def _record_history(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an action in history.
        
        Args:
            action: Action performed
            details: Optional action details
            metadata: Optional metadata
        """
        entry = ServiceHistoryEntry(
            action=action,
            details=details,
            metadata=metadata
        )
        self.history.append(entry)
        
        self.log_manager.log(
            "INFO",
            f"Action recorded: {action}",
            extra={
                "action": action,
                "details": details,
                "metadata": metadata
            }
        )
    
    def _get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state by key.
        
        Args:
            key: State key
            
        Returns:
            Optional[Dict[str, Any]]: State if found
        """
        state = self.state.get(key)
        if state:
            return {
                "key": state.key,
                "value": state.value,
                "metadata": state.metadata,
                "timestamp": state.timestamp
            }
        return None
    
    def _set_state(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Set state for key.
        
        Args:
            key: State key
            value: State value
            metadata: Optional metadata
            
        Returns:
            Dict[str, Any]: Updated state
        """
        state = ServiceState(
            key=key,
            value=value,
            metadata=metadata
        )
        self.state[key] = state
        
        self._record_history(
            "set_state",
            details={"key": key, "metadata": metadata}
        )
        
        return {
            "key": state.key,
            "value": state.value,
            "metadata": state.metadata,
            "timestamp": state.timestamp
        }
    
    def _delete_state(self, key: str) -> Dict[str, Any]:
        """Delete state for key.
        
        Args:
            key: State key
            
        Returns:
            Dict[str, Any]: Deletion result
        """
        if key in self.state:
            state = self.state.pop(key)
            self._record_history(
                "delete_state",
                details={"key": key}
            )
            return {
                "deleted": True,
                "key": key,
                "previous_value": state.value
            }
        return {
            "deleted": False,
            "key": key
        }
    
    def _get_history(
        self,
        filter_key: Optional[str] = None,
        filter_value: Optional[Any] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered history.
        
        Args:
            filter_key: Optional key to filter by in details
            filter_value: Optional value to match in details[filter_key]
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List[Dict[str, Any]]: Filtered history entries
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
        
        return [entry.dict() for entry in filtered]
    
    def _clear_history(
        self,
        filter_key: Optional[str] = None,
        filter_value: Optional[Any] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Clear history with optional filters.
        
        Args:
            filter_key: Optional key to filter by in details
            filter_value: Optional value to match in details[filter_key]
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            Dict[str, Any]: Clear operation result
        """
        initial_count = len(self.history)
        
        if not any([filter_key, start_time, end_time]):
            self.history = []
            self.log_manager.log("INFO", "History cleared")
            return {
                "cleared": True,
                "entries_removed": initial_count
            }
        
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
        
        entries_removed = initial_count - len(self.history)
        self.log_manager.log(
            "INFO",
            "History partially cleared with filters",
            extra={"entries_removed": entries_removed}
        )
        
        return {
            "cleared": True,
            "entries_removed": entries_removed,
            "filters_applied": {
                "filter_key": filter_key,
                "start_time": start_time,
                "end_time": end_time
            }
        }
    
    def _reset(self) -> Dict[str, Any]:
        """Reset service state.
        
        Returns:
            Dict[str, Any]: Reset operation result
        """
        state_count = len(self.state)
        history_count = len(self.history)
        
        self.state = {}
        self.history = []
        
        self.log_manager.log("INFO", "Service reset")
        
        return {
            "reset": True,
            "states_cleared": state_count,
            "history_entries_cleared": history_count
        }
    
    async def _execute_service_action(
        self,
        action: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a custom service action.
        
        Args:
            action: Action to execute
            input_data: Input data for the action
            context: Optional execution context
            
        Returns:
            Dict[str, Any]: Action results
        """
        raise NotImplementedError(f"Service action not implemented: {action}")
    
    def _generate_metadata(self, action: str) -> Dict[str, Any]:
        """Generate metadata about the service action.
        
        Args:
            action: Service action
            
        Returns:
            Dict[str, Any]: Action metadata
        """
        return {
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "state_count": len(self.state),
            "history_count": len(self.history)
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            super().cleanup()
            self.state = {}
            self.history = []
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            ) 