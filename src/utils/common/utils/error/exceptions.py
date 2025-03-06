"""
Custom exceptions for LangGraph.
"""

class BaseError(Exception):
    """Base class for all LangGraph exceptions."""
    pass

class ValidationError(BaseError):
    """Raised when validation fails."""
    pass

class ConfigurationError(BaseError):
    """Raised when configuration is invalid."""
    pass

class ExecutionError(BaseError):
    """Raised when execution fails."""
    pass

class ModelError(BaseError):
    """Raised when model operations fail."""
    pass

class StorageError(BaseError):
    """Raised when storage operations fail."""
    pass

class WorkflowError(BaseError):
    """Raised when workflow operations fail."""
    pass

__all__ = [
    'BaseError',
    'ValidationError',
    'ConfigurationError',
    'ExecutionError',
    'ModelError',
    'StorageError',
    'WorkflowError'
]
