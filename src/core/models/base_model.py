"""
Base models for the LangGraph project.
This module provides base model capabilities.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class BaseTimestampedModel(BaseModel):
    """Base model with timestamp."""
    timestamp: datetime = Field(default_factory=datetime.now)

class BaseMetadataModel(BaseTimestampedModel):
    """Base model with metadata."""
    metadata: Optional[Dict[str, Any]] = None

class BaseNamedModel(BaseMetadataModel):
    """Base model with name."""
    name: str
    description: Optional[str] = None

class BaseDataModel(BaseNamedModel):
    """Base model with data."""
    data: Dict[str, Any]

class BaseTypeModel(BaseNamedModel):
    """Base model with type."""
    type: str

__all__ = [
    'BaseTimestampedModel',
    'BaseMetadataModel',
    'BaseNamedModel',
    'BaseDataModel',
    'BaseTypeModel'
] 