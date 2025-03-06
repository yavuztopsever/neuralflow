"""
Error handling utilities for the LangGraph application.
This module provides standardized error handling functions and context managers
to ensure consistent error handling across the application.
"""

import logging
import traceback
import sys
from typing import Any, Callable, TypeVar, Optional, Dict, Type, Union
from functools import wraps
import asyncio
import time

from .exceptions import LangGraphError, ValidationError, ExecutionError, BaseError

logger = logging.getLogger(__name__)

T = TypeVar('T')
ErrorHandler = Callable[[Exception], Any]

class ErrorHandler:
    """Base class for error handlers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_handlers: Dict[Type[Exception], callable] = {}
    
    def register_handler(
        self,
        exception_type: Type[Exception],
        handler: callable
    ) -> None:
        """Register an error handler for an exception type."""
        self.error_handlers[exception_type] = handler
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle an error using the appropriate handler."""
        # Find the most specific handler for the error type
        handler = None
        for error_type, h in self.error_handlers.items():
            if isinstance(error, error_type):
                if handler is None or issubclass(error_type, type(handler)):
                    handler = h
        
        if handler:
            return handler(error, context or {})
        else:
            return self._default_handler(error, context or {})
    
    def _default_handler(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Default error handler."""
        raise error

class ErrorConfig:
    """Configuration for error handling."""
    
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'langgraph')
        self.log_errors = kwargs.get('log_errors', True)
        self.parameters = kwargs.get('parameters', {})
        self.metadata = kwargs.get('metadata', {})

__all__ = ['ErrorHandler', 'ErrorConfig']

class ErrorContext:
    """
    Context manager for handling errors in a block of code.
    
    Example:
        with ErrorContext("Failed to process data", default_return=[], log_level=logging.WARNING):
            # code that might raise exceptions
    """
    
    def __init__(self, 
                 error_message: str = "An error occurred", 
                 default_return: Any = None,
                 log_level: int = logging.ERROR,
                 reraise: bool = False):
        """
        Initialize the error context.
        
        Args:
            error_message: Message to log if an error occurs
            default_return: Value to return if an error occurs
            log_level: Logging level for errors
            reraise: Whether to reraise the exception after logging
        """
        self.error_message = error_message
        self.default_return = default_return
        self.log_level = log_level
        self.reraise = reraise
        self.result = default_return
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            stack_trace = traceback.format_exc()
            logger.log(self.log_level, f"{self.error_message}: {str(exc_val)}\n{stack_trace}")
            return not self.reraise
        return True

class MockObjectFactory:
    """Factory for creating mock objects."""
    
    @staticmethod
    def create_mock_object(original_type: Type, 
                        mock_methods: Dict[str, Any] = None,
                        error_message: str = "Using mock object") -> Any:
        """
        Create a mock object with basic implementations of methods.
        
        Args:
            original_type: The type to mock
            mock_methods: Dictionary of method names to mock implementations
            error_message: Message to log when creating the mock
            
        Returns:
            A mock object of the original type
        """
        logger.warning(f"{error_message} for {original_type.__name__}")
        
        if mock_methods is None:
            mock_methods = {}
        
        class MockObject:
            def __getattr__(self, name):
                if name in mock_methods:
                    return mock_methods[name]
                
                # Default mock method that returns None or empty container
                def mock_method(*args, **kwargs):
                    return None
                    
                return mock_method
                
        return MockObject()

def handle_error(
    error_types: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    default_return: Any = None,
    log_error: bool = True
) -> Callable:
    """Decorator for handling errors in functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
            except Exception as e:
                if log_error:
                    logger.exception(f"Unexpected error in {func.__name__}")
                raise
        
        def sync_wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
            except Exception as e:
                if log_error:
                    logger.exception(f"Unexpected error in {func.__name__}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def validate_input(
    schema: Dict[str, Any],
    error_message: Optional[str] = None
) -> Callable:
    """Decorator for validating function inputs."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                # Validate inputs against schema
                # Implementation will be added later
                return await func(*args, **kwargs)
            except Exception as e:
                raise ValidationError(
                    error_message or f"Invalid input for {func.__name__}: {str(e)}"
                )
        
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                # Validate inputs against schema
                # Implementation will be added later
                return func(*args, **kwargs)
            except Exception as e:
                raise ValidationError(
                    error_message or f"Invalid input for {func.__name__}: {str(e)}"
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    error_types: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception
) -> Callable:
    """Decorator for retrying functions on error."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {current_delay} seconds..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise ExecutionError(
                f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
            )
        
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_error = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {current_delay} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise ExecutionError(
                f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
            )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator 