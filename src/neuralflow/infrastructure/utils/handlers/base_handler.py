"""
Base error handling utilities for the LangGraph project.
This module provides base error handling capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
import traceback
from ..models.base_model import BaseMetadataModel

logger = logging.getLogger(__name__)

class BaseError(BaseMetadataModel):
    """Base error model."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

class BaseErrorHandler:
    """Base handler for error management."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.errors: List[BaseError] = []
        self.error_counts: Dict[str, int] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def handle_error(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        include_stack_trace: bool = True
    ) -> BaseError:
        """
        Handle an error.
        
        Args:
            code: Error code
            message: Error message
            details: Optional error details
            include_stack_trace: Whether to include stack trace
            
        Returns:
            BaseError: Created error
        """
        try:
            # Create error
            error = BaseError(
                code=code,
                message=message,
                details=details,
                stack_trace=traceback.format_exc() if include_stack_trace else None
            )
            
            # Add to errors list
            self.errors.append(error)
            
            # Update error count
            self.error_counts[code] = self.error_counts.get(code, 0) + 1
            
            # Log error
            self.logger.error(
                f"Error {code}: {message}",
                extra={
                    "error_code": code,
                    "error_details": details,
                    "stack_trace": error.stack_trace
                }
            )
            
            return error
            
        except Exception as e:
            self.logger.error(f"Failed to handle error: {e}")
            raise
    
    def get_errors(
        self,
        code: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[BaseError]:
        """
        Get errors.
        
        Args:
            code: Optional error code to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            
        Returns:
            List[BaseError]: List of errors
        """
        errors = self.errors
        
        if code:
            errors = [error for error in errors if error.code == code]
        
        if start_time:
            errors = [error for error in errors if error.timestamp >= start_time]
        
        if end_time:
            errors = [error for error in errors if error.timestamp <= end_time]
        
        return errors
    
    def get_error_count(self, code: Optional[str] = None) -> int:
        """
        Get error count.
        
        Args:
            code: Optional error code to count
            
        Returns:
            int: Error count
        """
        if code:
            return self.error_counts.get(code, 0)
        return sum(self.error_counts.values())
    
    def clear_errors(
        self,
        code: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Clear errors.
        
        Args:
            code: Optional error code to clear
            start_time: Optional start time to clear from
            end_time: Optional end time to clear to
        """
        if code:
            self.errors = [error for error in self.errors if error.code != code]
            self.error_counts.pop(code, None)
        elif start_time and end_time:
            self.errors = [
                error for error in self.errors
                if error.timestamp < start_time or error.timestamp > end_time
            ]
        else:
            self.errors = []
            self.error_counts = {}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary.
        
        Returns:
            Dict[str, Any]: Error summary
        """
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "latest_error": self.errors[-1] if self.errors else None
        }
    
    def reset(self) -> None:
        """Reset error handler."""
        self.errors = []
        self.error_counts = {}
        self.logger.info("Error handler reset")

__all__ = ['BaseErrorHandler', 'BaseError'] 