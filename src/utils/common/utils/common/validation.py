"""Validation utilities for the LangGraph system."""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import re
import json


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate data against a schema."""
    try:
        for key, value in schema.items():
            if key not in data:
                return False
            
            if not validate_type(data[key], value):
                return False
        
        return True
    except Exception:
        return False


def validate_type(value: Any, expected_type: Any) -> bool:
    """Validate a value against an expected type."""
    if isinstance(expected_type, type):
        return isinstance(value, expected_type)
    
    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "number":
        return isinstance(value, (int, float))
    elif expected_type == "integer":
        return isinstance(value, int)
    elif expected_type == "boolean":
        return isinstance(value, bool)
    elif expected_type == "array":
        return isinstance(value, list)
    elif expected_type == "object":
        return isinstance(value, dict)
    elif expected_type == "null":
        return value is None
    elif expected_type == "date":
        try:
            datetime.fromisoformat(value)
            return True
        except:
            return False
    elif expected_type == "email":
        return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value))
    elif expected_type == "url":
        return bool(re.match(r"^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*$", value))
    elif expected_type == "json":
        try:
            json.loads(value)
            return True
        except:
            return False
    
    return False


def validate_required(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that all required fields are present."""
    return all(field in data for field in required_fields)


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None
) -> bool:
    """Validate that a numeric value is within a range."""
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def validate_length(
    value: Union[str, List[Any]],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> bool:
    """Validate the length of a string or list."""
    length = len(value)
    if min_length is not None and length < min_length:
        return False
    if max_length is not None and length > max_length:
        return False
    return True


def validate_pattern(value: str, pattern: str) -> bool:
    """Validate a string against a regex pattern."""
    try:
        return bool(re.match(pattern, value))
    except:
        return False


def validate_enum(value: Any, allowed_values: List[Any]) -> bool:
    """Validate that a value is in a list of allowed values."""
    return value in allowed_values


def validate_format(value: str, format_type: str) -> bool:
    """Validate a string against a specific format."""
    if format_type == "email":
        return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value))
    elif format_type == "url":
        return bool(re.match(r"^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*$", value))
    elif format_type == "date":
        try:
            datetime.fromisoformat(value)
            return True
        except:
            return False
    elif format_type == "json":
        try:
            json.loads(value)
            return True
        except:
            return False
    return False
