"""
Unit tests for the context management components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import BaseModel
import asyncio

from src.core.context.context_manager import ContextManager

@pytest.fixture
def mock_context_data():
    """Create mock context data for testing."""
    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "environment": "development",
        "variables": {
            "language": "python",
            "framework": "pytest"
        },
        "metadata": {
            "start_time": "2024-03-07T00:00:00Z",
            "source": "test"
        }
    }

def test_context_manager_initialization():
    """Test initialization of ContextManager."""
    manager = ContextManager()
    assert manager.contexts == {}
    assert manager.current_context is None
    assert manager.context_history == []

def test_context_manager_context_operations(mock_context_data):
    """Test context operations in ContextManager."""
    manager = ContextManager()
    
    # Create context
    context_id = manager.create_context(mock_context_data)
    assert context_id in manager.contexts
    assert manager.contexts[context_id] == mock_context_data
    
    # Get context
    context = manager.get_context(context_id)
    assert context == mock_context_data
    
    # Update context
    updated_data = mock_context_data.copy()
    updated_data["environment"] = "production"
    manager.update_context(context_id, updated_data)
    assert manager.contexts[context_id]["environment"] == "production"
    
    # Delete context
    manager.delete_context(context_id)
    assert context_id not in manager.contexts

def test_context_manager_current_context(mock_context_data):
    """Test current context management."""
    manager = ContextManager()
    
    # Set current context
    context_id = manager.create_context(mock_context_data)
    manager.set_current_context(context_id)
    assert manager.current_context == mock_context_data
    
    # Clear current context
    manager.clear_current_context()
    assert manager.current_context is None

def test_context_manager_context_history(mock_context_data):
    """Test context history management."""
    manager = ContextManager()
    
    # Add context to history
    context_id = manager.create_context(mock_context_data)
    manager.add_to_history(context_id)
    assert len(manager.context_history) == 1
    assert manager.context_history[0]["context_id"] == context_id
    
    # Get history
    history = manager.get_history()
    assert len(history) == 1
    assert history[0]["context_id"] == context_id
    
    # Clear history
    manager.clear_history()
    assert len(manager.context_history) == 0

def test_context_manager_variable_operations(mock_context_data):
    """Test variable operations in context."""
    manager = ContextManager()
    context_id = manager.create_context(mock_context_data)
    
    # Set variable
    manager.set_variable(context_id, "new_var", "value")
    assert manager.contexts[context_id]["variables"]["new_var"] == "value"
    
    # Get variable
    value = manager.get_variable(context_id, "new_var")
    assert value == "value"
    
    # Update variable
    manager.update_variable(context_id, "new_var", "updated_value")
    assert manager.contexts[context_id]["variables"]["new_var"] == "updated_value"
    
    # Delete variable
    manager.delete_variable(context_id, "new_var")
    assert "new_var" not in manager.contexts[context_id]["variables"]

def test_context_manager_metadata_operations(mock_context_data):
    """Test metadata operations in context."""
    manager = ContextManager()
    context_id = manager.create_context(mock_context_data)
    
    # Set metadata
    manager.set_metadata(context_id, "new_meta", "value")
    assert manager.contexts[context_id]["metadata"]["new_meta"] == "value"
    
    # Get metadata
    value = manager.get_metadata(context_id, "new_meta")
    assert value == "value"
    
    # Update metadata
    manager.update_metadata(context_id, "new_meta", "updated_value")
    assert manager.contexts[context_id]["metadata"]["new_meta"] == "updated_value"
    
    # Delete metadata
    manager.delete_metadata(context_id, "new_meta")
    assert "new_meta" not in manager.contexts[context_id]["metadata"]

def test_context_manager_context_merging():
    """Test context merging operations."""
    manager = ContextManager()
    
    # Create two contexts
    context1 = {
        "variables": {"var1": "value1"},
        "metadata": {"meta1": "value1"}
    }
    context2 = {
        "variables": {"var2": "value2"},
        "metadata": {"meta2": "value2"}
    }
    
    context1_id = manager.create_context(context1)
    context2_id = manager.create_context(context2)
    
    # Merge contexts
    merged_id = manager.merge_contexts(context1_id, context2_id)
    
    assert merged_id in manager.contexts
    merged_context = manager.contexts[merged_id]
    assert merged_context["variables"]["var1"] == "value1"
    assert merged_context["variables"]["var2"] == "value2"
    assert merged_context["metadata"]["meta1"] == "value1"
    assert merged_context["metadata"]["meta2"] == "value2"

def test_context_manager_context_cloning():
    """Test context cloning operations."""
    manager = ContextManager()
    context_id = manager.create_context(mock_context_data)
    
    # Clone context
    clone_id = manager.clone_context(context_id)
    
    assert clone_id in manager.contexts
    assert manager.contexts[clone_id] == manager.contexts[context_id]
    assert clone_id != context_id

def test_context_manager_error_handling():
    """Test error handling in ContextManager."""
    manager = ContextManager()
    
    # Test getting non-existent context
    with pytest.raises(KeyError):
        manager.get_context("non_existent")
    
    # Test updating non-existent context
    with pytest.raises(KeyError):
        manager.update_context("non_existent", {})
    
    # Test deleting non-existent context
    with pytest.raises(KeyError):
        manager.delete_context("non_existent")
    
    # Test getting non-existent variable
    context_id = manager.create_context({})
    with pytest.raises(KeyError):
        manager.get_variable(context_id, "non_existent")
    
    # Test getting non-existent metadata
    with pytest.raises(KeyError):
        manager.get_metadata(context_id, "non_existent")

def test_context_manager_serialization():
    """Test context serialization."""
    manager = ContextManager()
    context_id = manager.create_context(mock_context_data)
    
    # Serialize context
    serialized = manager.serialize_context(context_id)
    
    assert serialized["context_id"] == context_id
    assert serialized["data"] == mock_context_data
    assert "created_at" in serialized
    assert "updated_at" in serialized

def test_context_manager_deserialization():
    """Test context deserialization."""
    manager = ContextManager()
    
    # Create serialized data
    serialized_data = {
        "context_id": "test_context",
        "data": mock_context_data,
        "created_at": "2024-03-07T00:00:00Z",
        "updated_at": "2024-03-07T00:00:00Z"
    }
    
    # Deserialize context
    context_id = manager.deserialize_context(serialized_data)
    
    assert context_id in manager.contexts
    assert manager.contexts[context_id] == mock_context_data 