"""
Base tools system for NeuralFlow.
Provides unified tool capabilities with integrated model support.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum

from ..models.base_model import BaseNamedModel, BaseMetadataModel
from ..storage.base import BaseStorage, StorageConfig
from ..utils.error.base_handler import BaseErrorHandler
from ..utils.logging.base_manager import BaseLogManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ToolType(Enum):
    """Types of tools available."""
    SEARCH = "search"
    MEMORY = "memory"
    PROCESSING = "processing"
    CUSTOM = "custom"

class ToolConfig(BaseNamedModel):
    """Configuration for tools."""
    tool_type: ToolType
    parameters: Dict[str, Any]
    enabled: bool = True

class ToolResult(BaseMetadataModel):
    """Base model for tool execution results."""
    tool_name: str
    tool_type: ToolType
    status: str
    result: Dict[str, Any]
    error: Optional[str] = None

class BaseTool(ABC, Generic[T]):
    """Base class for all tools with integrated model support."""
    
    def __init__(
        self,
        config: ToolConfig,
        storage_config: Optional[StorageConfig] = None
    ):
        """Initialize the tool.
        
        Args:
            config: Tool configuration
            storage_config: Optional storage configuration
        """
        self.config = config
        self.storage = BaseStorage(storage_config) if storage_config else None
        
        # Initialize components
        self.error_handler = BaseErrorHandler()
        self.log_manager = BaseLogManager()
        
        # Initialize cache
        self.cache_dir = Path("tools/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute the tool.
        
        Args:
            input_data: Input data for the tool
            context: Optional execution context
            
        Returns:
            ToolResult: Tool execution result
        """
        if not self.config.enabled:
            return ToolResult(
                tool_name=self.config.name,
                tool_type=self.config.tool_type,
                status="disabled",
                result={},
                error="Tool is disabled"
            )
        
        try:
            # Execute tool implementation
            result = await self._execute_impl(input_data, context)
            
            # Create result
            tool_result = ToolResult(
                tool_name=self.config.name,
                tool_type=self.config.tool_type,
                status="success",
                result=result
            )
            
            # Cache result if storage is configured
            if self.storage:
                await self._cache_result(tool_result)
            
            return tool_result
            
        except Exception as e:
            error_msg = str(e)
            self.error_handler.handle_error(
                "TOOL_EXECUTION_ERROR",
                f"Failed to execute tool: {error_msg}",
                details={
                    "tool_name": self.config.name,
                    "tool_type": self.config.tool_type,
                    "input_data": input_data,
                    "context": context
                }
            )
            
            return ToolResult(
                tool_name=self.config.name,
                tool_type=self.config.tool_type,
                status="error",
                result={},
                error=error_msg
            )
    
    @abstractmethod
    async def _execute_impl(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute tool implementation.
        
        Args:
            input_data: Input data for the tool
            context: Optional execution context
            
        Returns:
            Dict[str, Any]: Tool execution results
        """
        pass
    
    async def _cache_result(self, result: ToolResult) -> None:
        """Cache tool execution result.
        
        Args:
            result: Tool execution result to cache
        """
        try:
            if not self.storage:
                return
                
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "result": result.dict()
            }
            
            await self.storage.store(
                f"cache/tools/{result.tool_name}_{result.timestamp}",
                cache_data,
                metadata={"type": "tool_result_cache"}
            )
            
        except Exception as e:
            self.error_handler.handle_error(
                "CACHE_ERROR",
                f"Failed to cache tool result: {e}",
                details={"tool_name": result.tool_name}
            )
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.storage:
                self.storage.cleanup()
        except Exception as e:
            self.error_handler.handle_error(
                "CLEANUP_ERROR",
                f"Failed to clean up resources: {e}"
            ) 