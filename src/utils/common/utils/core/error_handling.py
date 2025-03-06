"""
Error handling utilities specific to the LangGraph project.
These utilities handle error handling that is not covered by LangChain's built-in error handling.
"""

from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass
import traceback
import logging
from datetime import datetime
import json

@dataclass
class ErrorContext:
    error_type: str
    message: str
    timestamp: str
    traceback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LangGraphError(Exception):
    """Base exception class for LangGraph errors."""
    def __init__(self, message: str, error_type: str = "general", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_type": self.error_type,
            "message": str(self),
            "timestamp": self.timestamp,
            "traceback": self.traceback,
            "metadata": self.metadata
        }

class ValidationError(LangGraphError):
    """Exception raised for validation errors."""
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, "validation", metadata)

class ProcessingError(LangGraphError):
    """Exception raised for processing errors."""
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, "processing", metadata)

class StateError(LangGraphError):
    """Exception raised for state-related errors."""
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, "state", metadata)

class GraphError(LangGraphError):
    """Exception raised for graph-related errors."""
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, "graph", metadata)

class VectorStoreError(LangGraphError):
    """Exception raised for vector store errors."""
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, "vector_store", metadata)

class ErrorHandler:
    """Handles error processing and logging."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle an error and create an error context."""
        error_context = ErrorContext(
            error_type=error.error_type if isinstance(error, LangGraphError) else "unknown",
            message=str(error),
            timestamp=datetime.now().isoformat(),
            traceback=traceback.format_exc(),
            metadata={
                **(context or {}),
                **(error.metadata if isinstance(error, LangGraphError) else {})
            }
        )
        
        # Log error
        self.logger.error(
            f"Error: {error_context.message}",
            extra={
                "error_type": error_context.error_type,
                "metadata": error_context.metadata
            },
            exc_info=True
        )
        
        # Add to history
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        return error_context

    def get_error_history(self, error_type: Optional[str] = None) -> List[ErrorContext]:
        """Get error history, optionally filtered by error type."""
        if error_type:
            return [error for error in self.error_history if error.error_type == error_type]
        return self.error_history

    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()

    def save_error_history(self, filepath: str):
        """Save error history to a file."""
        with open(filepath, 'w') as f:
            json.dump(
                [error.to_dict() for error in self.error_history],
                f,
                indent=2
            )

    def load_error_history(self, filepath: str):
        """Load error history from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.error_history = [
                ErrorContext(**error_data)
                for error_data in data
            ]

def with_error_handling(func):
    """Decorator for handling errors in functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler = ErrorHandler()
            error_context = error_handler.handle_error(e)
            raise
    return wrapper

def create_error_response(error: Exception, include_traceback: bool = False) -> Dict[str, Any]:
    """Create a standardized error response."""
    response = {
        "success": False,
        "error": {
            "type": error.error_type if isinstance(error, LangGraphError) else "unknown",
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if include_traceback:
        response["error"]["traceback"] = traceback.format_exc()
    
    if isinstance(error, LangGraphError):
        response["error"]["metadata"] = error.metadata
    
    return response

def handle_langchain_error(error: Exception) -> Dict[str, Any]:
    """Handle errors from LangChain operations."""
    error_type = "langchain"
    if "openai" in str(error).lower():
        error_type = "openai"
    elif "vectorstore" in str(error).lower():
        error_type = "vectorstore"
    elif "memory" in str(error).lower():
        error_type = "memory"
    
    return create_error_response(
        LangGraphError(
            message=str(error),
            error_type=error_type,
            metadata={"original_error": str(error)}
        )
    ) 