"""
Base tool for the LangGraph project.
This module provides base tool capabilities and common functionality for all tools.

The base tool module defines:
1. ToolMetadata: A model for tool metadata including name, description, version, etc.
2. BaseTool: An abstract base class that all tools must inherit from.

Features:
- Common tool initialization with metadata
- Usage tracking and statistics
- Error handling and logging
- State management
- Abstract execution interface

Example:
    ```python
    from src.core.tools.base import BaseTool
    
    class MyTool(BaseTool):
        def __init__(self, config: Optional[Dict[str, Any]] = None):
            super().__init__(
                name="my_tool",
                description="My custom tool",
                config=config
            )
        
        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            self.record_usage()
            try:
                # Tool implementation
                result = await self._process(*args, **kwargs)
                return result
            except Exception as e:
                self.record_error(e)
                raise
    ```
"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel, Field

class ToolMetadata(BaseModel):
    """
    Tool metadata model.
    
    Attributes:
        name: Tool name
        description: Tool description
        version: Tool version (defaults to "1.0.0")
        created_at: Tool creation timestamp
        updated_at: Tool last update timestamp
        config: Optional tool configuration
    """
    name: str
    description: str
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    config: Optional[Dict[str, Any]] = None

class BaseTool(ABC):
    """
    Base class for all tools.
    
    This class provides common functionality that all tools should inherit:
    - Tool metadata management
    - Usage tracking
    - Error handling
    - Statistics
    
    All tools must implement the execute() method.
    """
    
    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base tool.
        
        Args:
            name: Tool name
            description: Tool description
            config: Optional tool configuration
            
        The initialization process:
        1. Creates tool metadata
        2. Initializes usage tracking
        3. Sets up error tracking
        """
        self.metadata = ToolMetadata(
            name=name,
            description=description,
            config=config
        )
        self.usage_count = 0
        self.last_used = None
        self.errors = []
    
    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool.
        
        This is the main entry point for tool execution. All tools must implement
        this method with their specific functionality.
        
        Args:
            *args: Positional arguments specific to the tool
            **kwargs: Keyword arguments specific to the tool
            
        Returns:
            Any: Tool execution result
            
        Raises:
            NotImplementedError: If the tool doesn't implement this method
            Exception: Any tool-specific exceptions that may occur
        """
        pass
    
    def record_usage(self) -> None:
        """
        Record tool usage.
        
        This method:
        1. Increments the usage counter
        2. Updates the last used timestamp
        """
        self.usage_count += 1
        self.last_used = datetime.now()
    
    def record_error(self, error: Exception) -> None:
        """
        Record tool error.
        
        This method stores error information including:
        1. Error message
        2. Timestamp of occurrence
        
        Args:
            error: Error that occurred during tool execution
        """
        self.errors.append({
            "error": str(error),
            "timestamp": datetime.now()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tool statistics.
        
        Returns a dictionary containing:
        1. Tool name
        2. Usage count
        3. Last used timestamp
        4. Error count
        5. Last error details
        
        Returns:
            Dict[str, Any]: Tool statistics
        """
        return {
            "name": self.metadata.name,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "error_count": len(self.errors),
            "last_error": self.errors[-1] if self.errors else None
        }
    
    def reset_stats(self) -> None:
        """
        Reset tool statistics.
        
        This method:
        1. Resets the usage counter to 0
        2. Clears the last used timestamp
        3. Clears all recorded errors
        """
        self.usage_count = 0
        self.last_used = None
        self.errors = []

__all__ = ['BaseTool', 'ToolMetadata'] 