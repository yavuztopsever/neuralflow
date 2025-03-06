"""
Validation utilities specific to the LangGraph project.
These utilities handle validation that is not covered by LangChain's built-in validation.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import re
import json
from datetime import datetime

@dataclass
class ValidationError:
    field: str
    message: str
    value: Any = None

class ValidationResult:
    def __init__(self, is_valid: bool, errors: Optional[List[ValidationError]] = None):
        self.is_valid = is_valid
        self.errors = errors or []

    def add_error(self, field: str, message: str, value: Any = None):
        self.errors.append(ValidationError(field=field, message=message, value=value))
        self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [
                {
                    "field": error.field,
                    "message": error.message,
                    "value": error.value
                }
                for error in self.errors
            ]
        }

def validate_task_input(task: Dict[str, Any]) -> ValidationResult:
    """Validate task input data."""
    result = ValidationResult(True)
    
    # Required fields
    required_fields = ["input", "type"]
    for field in required_fields:
        if field not in task:
            result.add_error(field, f"Missing required field: {field}")
    
    # Validate task type
    if "type" in task:
        valid_types = ["query", "command", "knowledge_update", "system"]
        if task["type"] not in valid_types:
            result.add_error("type", f"Invalid task type. Must be one of: {valid_types}")
    
    # Validate input format
    if "input" in task and not isinstance(task["input"], str):
        result.add_error("input", "Input must be a string")
    
    # Validate metadata if present
    if "metadata" in task:
        if not isinstance(task["metadata"], dict):
            result.add_error("metadata", "Metadata must be a dictionary")
        else:
            # Validate timestamp if present
            if "timestamp" in task["metadata"]:
                try:
                    datetime.fromisoformat(task["metadata"]["timestamp"])
                except ValueError:
                    result.add_error("metadata.timestamp", "Invalid timestamp format")
    
    return result

def validate_graph_node(node: Dict[str, Any]) -> ValidationResult:
    """Validate graph node data."""
    result = ValidationResult(True)
    
    # Required fields
    required_fields = ["id", "content"]
    for field in required_fields:
        if field not in node:
            result.add_error(field, f"Missing required field: {field}")
    
    # Validate ID format
    if "id" in node:
        if not re.match(r"^[a-zA-Z0-9_-]+$", node["id"]):
            result.add_error("id", "ID must contain only alphanumeric characters, underscores, and hyphens")
    
    # Validate content
    if "content" in node and not isinstance(node["content"], str):
        result.add_error("content", "Content must be a string")
    
    # Validate metadata if present
    if "metadata" in node:
        if not isinstance(node["metadata"], dict):
            result.add_error("metadata", "Metadata must be a dictionary")
        else:
            # Validate required metadata fields
            if "timestamp" not in node["metadata"]:
                result.add_error("metadata.timestamp", "Missing required metadata field: timestamp")
            else:
                try:
                    datetime.fromisoformat(node["metadata"]["timestamp"])
                except ValueError:
                    result.add_error("metadata.timestamp", "Invalid timestamp format")
    
    return result

def validate_graph_edge(edge: Dict[str, Any]) -> ValidationResult:
    """Validate graph edge data."""
    result = ValidationResult(True)
    
    # Required fields
    required_fields = ["source", "target"]
    for field in required_fields:
        if field not in edge:
            result.add_error(field, f"Missing required field: {field}")
    
    # Validate source and target
    for field in ["source", "target"]:
        if field in edge and not re.match(r"^[a-zA-Z0-9_-]+$", edge[field]):
            result.add_error(field, f"{field} must contain only alphanumeric characters, underscores, and hyphens")
    
    # Validate weight if present
    if "weight" in edge:
        if not isinstance(edge["weight"], (int, float)):
            result.add_error("weight", "Weight must be a number")
        elif edge["weight"] < 0:
            result.add_error("weight", "Weight must be non-negative")
    
    # Validate metadata if present
    if "metadata" in edge:
        if not isinstance(edge["metadata"], dict):
            result.add_error("metadata", "Metadata must be a dictionary")
    
    return result

def validate_vector_store_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate vector store configuration."""
    result = ValidationResult(True)
    
    # Required fields
    required_fields = ["dimension", "index_type"]
    for field in required_fields:
        if field not in config:
            result.add_error(field, f"Missing required field: {field}")
    
    # Validate dimension
    if "dimension" in config:
        if not isinstance(config["dimension"], int):
            result.add_error("dimension", "Dimension must be an integer")
        elif config["dimension"] <= 0:
            result.add_error("dimension", "Dimension must be positive")
    
    # Validate index type
    if "index_type" in config:
        valid_types = ["faiss", "annoy", "hnsw"]
        if config["index_type"] not in valid_types:
            result.add_error("index_type", f"Invalid index type. Must be one of: {valid_types}")
    
    # Validate optional parameters
    if "metric" in config:
        valid_metrics = ["cosine", "l2", "dot"]
        if config["metric"] not in valid_metrics:
            result.add_error("metric", f"Invalid metric. Must be one of: {valid_metrics}")
    
    return result

def validate_state_data(state: Dict[str, Any]) -> ValidationResult:
    """Validate state data."""
    result = ValidationResult(True)
    
    # Required fields
    required_fields = ["status", "data"]
    for field in required_fields:
        if field not in state:
            result.add_error(field, f"Missing required field: {field}")
    
    # Validate status
    if "status" in state:
        valid_statuses = ["initial", "processing", "completed", "error"]
        if state["status"] not in valid_statuses:
            result.add_error("status", f"Invalid status. Must be one of: {valid_statuses}")
    
    # Validate data
    if "data" in state and not isinstance(state["data"], dict):
        result.add_error("data", "Data must be a dictionary")
    
    # Validate error field if present
    if "error" in state and state["error"] is not None:
        if not isinstance(state["error"], str):
            result.add_error("error", "Error must be a string")
    
    return result 