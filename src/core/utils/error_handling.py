"""
Error handling utilities for the LangGraph project.
These utilities provide error handling capabilities integrated with LangChain.
"""

from typing import Any, Dict, Optional, Type, Union, List
import traceback
import logging
from functools import wraps
from langchain.schema import BaseMessage
from langchain.schema.output import LLMResult
from langchain.schema.document import Document
from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.callbacks.base import BaseCallbackHandler

class LangGraphError(Exception):
    """Base exception class for LangGraph errors."""
    pass

class ValidationError(LangGraphError):
    """Exception raised for validation errors."""
    pass

class ProcessingError(LangGraphError):
    """Exception raised for processing errors."""
    pass

class StorageError(LangGraphError):
    """Exception raised for storage errors."""
    pass

class SearchError(LangGraphError):
    """Exception raised for search errors."""
    pass

class GraphError(LangGraphError):
    """Exception raised for graph-related errors."""
    pass

class ErrorHandler:
    """Handler for managing and processing errors in the LangGraph project."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error and return error information.
        
        Args:
            error: The exception to handle
            context: Optional context information
            
        Returns:
            Dict[str, Any]: Error information
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create error info
        error_info = {
            'type': error_type,
            'message': error_message,
            'traceback': error_traceback,
            'context': context or {},
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add to history
        self.error_history.append(error_info)
        
        # Log error
        self.logger.error(f"Error: {error_type} - {error_message}")
        if context:
            self.logger.error(f"Context: {context}")
        self.logger.error(f"Traceback: {error_traceback}")
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dict[str, Any]: Error statistics
        """
        return {
            'counts': self.error_counts,
            'total_errors': sum(self.error_counts.values()),
            'unique_errors': len(self.error_counts)
        }
    
    def clear_errors(self) -> None:
        """Clear error history and counts."""
        self.error_counts.clear()
        self.error_history.clear()

def error_handler(error_types: Optional[List[Type[Exception]]] = None):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_types: Optional list of error types to handle
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_types is None or any(isinstance(e, t) for t in error_types):
                    handler = ErrorHandler()
                    error_info = handler.handle_error(e, {
                        'function': func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    })
                    raise LangGraphError(f"Error in {func.__name__}: {error_info['message']}")
                raise
        return wrapper
    return decorator

def validate_langchain_input(input_data: Any, expected_type: Type) -> None:
    """
    Validate LangChain input data.
    
    Args:
        input_data: The input data to validate
        expected_type: Expected type of the input
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(input_data, expected_type):
        raise ValidationError(f"Expected {expected_type.__name__}, got {type(input_data).__name__}")

def validate_langchain_output(output_data: Any, expected_type: Type) -> None:
    """
    Validate LangChain output data.
    
    Args:
        output_data: The output data to validate
        expected_type: Expected type of the output
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(output_data, expected_type):
        raise ValidationError(f"Expected {expected_type.__name__}, got {type(output_data).__name__}")

def validate_vector_store(vector_store: VectorStore) -> None:
    """
    Validate a LangChain vector store.
    
    Args:
        vector_store: The vector store to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(vector_store, VectorStore):
        raise ValidationError(f"Expected VectorStore, got {type(vector_store).__name__}")

def validate_embeddings(embeddings: Embeddings) -> None:
    """
    Validate LangChain embeddings.
    
    Args:
        embeddings: The embeddings to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(embeddings, Embeddings):
        raise ValidationError(f"Expected Embeddings, got {type(embeddings).__name__}")

def validate_document(document: Document) -> None:
    """
    Validate a LangChain document.
    
    Args:
        document: The document to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(document, Document):
        raise ValidationError(f"Expected Document, got {type(document).__name__}")

def validate_messages(messages: List[BaseMessage]) -> None:
    """
    Validate a list of LangChain messages.
    
    Args:
        messages: The messages to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(messages, list):
        raise ValidationError(f"Expected list, got {type(messages).__name__}")
    if not all(isinstance(msg, BaseMessage) for msg in messages):
        raise ValidationError("All messages must be instances of BaseMessage")

def validate_llm_result(result: LLMResult) -> None:
    """
    Validate a LangChain LLM result.
    
    Args:
        result: The LLM result to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(result, LLMResult):
        raise ValidationError(f"Expected LLMResult, got {type(result).__name__}")

__all__ = [
    'LangGraphError',
    'ValidationError',
    'ProcessingError',
    'StorageError',
    'SearchError',
    'GraphError',
    'ErrorHandler',
    'error_handler',
    'validate_langchain_input',
    'validate_langchain_output',
    'validate_vector_store',
    'validate_embeddings',
    'validate_document',
    'validate_messages',
    'validate_llm_result'
] 