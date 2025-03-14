"""
Unit tests for the state management components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import BaseModel
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

from src.core.state.state_manager import StateManager
from src.core.state.state_persistence import StatePersistence
from src.core.state.state_validator import StateValidator
from src.core.state.workflow_state import WorkflowState
from src.core.state.persistence.state_persistence import StatePersistence as ConsolidatedStatePersistence

@pytest.fixture
def mock_state_data():
    """Create mock state data for testing."""
    return {
        "workflow_id": "test_workflow",
        "status": "running",
        "data": {
            "input": "test input",
            "output": "test output",
            "metadata": {
                "start_time": "2024-03-07T00:00:00Z",
                "end_time": None
            }
        }
    }

@pytest.fixture
def mock_validation_rules():
    """Create mock validation rules."""
    return {
        "required_fields": ["workflow_id", "status"],
        "field_types": {
            "workflow_id": "string",
            "status": "string",
            "data": "object"
        },
        "valid_status_values": ["pending", "running", "completed", "failed"]
    }

def test_state_manager_initialization():
    """Test initialization of StateManager."""
    manager = StateManager()
    assert manager.states == {}
    assert manager.history == []
    assert manager.current_state is None

def test_state_manager_state_operations(mock_state_data):
    """Test state operations in StateManager."""
    manager = StateManager()
    
    # Set state
    manager.set_state("test_workflow", mock_state_data)
    assert "test_workflow" in manager.states
    assert manager.states["test_workflow"] == mock_state_data
    
    # Get state
    state = manager.get_state("test_workflow")
    assert state == mock_state_data
    
    # Update state
    updated_data = mock_state_data.copy()
    updated_data["status"] = "completed"
    manager.update_state("test_workflow", updated_data)
    assert manager.states["test_workflow"]["status"] == "completed"
    
    # Delete state
    manager.delete_state("test_workflow")
    assert "test_workflow" not in manager.states

def test_state_manager_history(mock_state_data):
    """Test state history management."""
    manager = StateManager()
    
    # Add state to history
    manager.add_to_history("test_workflow", mock_state_data)
    assert len(manager.history) == 1
    assert manager.history[0]["workflow_id"] == "test_workflow"
    
    # Get history
    history = manager.get_history("test_workflow")
    assert len(history) == 1
    assert history[0] == mock_state_data
    
    # Clear history
    manager.clear_history()
    assert len(manager.history) == 0

def test_state_persistence_initialization():
    """Test initialization of StatePersistence."""
    persistence = StatePersistence()
    assert persistence.storage_path is not None
    assert os.path.exists(persistence.storage_path)

def test_state_persistence_save_load(mock_state_data):
    """Test state persistence operations."""
    persistence = StatePersistence()
    
    # Save state
    persistence.save_state("test_workflow", mock_state_data)
    assert os.path.exists(os.path.join(persistence.storage_path, "test_workflow.json"))
    
    # Load state
    loaded_state = persistence.load_state("test_workflow")
    assert loaded_state == mock_state_data
    
    # Delete state
    persistence.delete_state("test_workflow")
    assert not os.path.exists(os.path.join(persistence.storage_path, "test_workflow.json"))

def test_state_validator_initialization(mock_validation_rules):
    """Test initialization of StateValidator."""
    validator = StateValidator(rules=mock_validation_rules)
    assert validator.rules == mock_validation_rules

def test_state_validator_validation(mock_state_data, mock_validation_rules):
    """Test state validation."""
    validator = StateValidator(rules=mock_validation_rules)
    
    # Test valid state
    result = validator.validate(mock_state_data)
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    
    # Test invalid state
    invalid_state = {"workflow_id": "test_workflow"}  # Missing required field
    result = validator.validate(invalid_state)
    assert result["is_valid"] is False
    assert len(result["errors"]) > 0

def test_workflow_state_initialization():
    """Test initialization of WorkflowState."""
    state = WorkflowState()
    assert state.workflow_id is None
    assert state.status == "pending"
    assert state.data == {}
    assert state.metadata == {}
    assert state.created_at is not None
    assert state.updated_at is not None

def test_workflow_state_operations(mock_state_data):
    """Test WorkflowState operations."""
    state = WorkflowState(**mock_state_data)
    
    # Test state properties
    assert state.workflow_id == mock_state_data["workflow_id"]
    assert state.status == mock_state_data["status"]
    assert state.data == mock_state_data["data"]
    
    # Test state updates
    state.update_status("completed")
    assert state.status == "completed"
    assert state.updated_at > state.created_at
    
    # Test state data updates
    state.update_data({"output": "new output"})
    assert state.data["output"] == "new output"
    
    # Test state serialization
    serialized = state.serialize()
    assert serialized["workflow_id"] == state.workflow_id
    assert serialized["status"] == state.status
    assert serialized["data"] == state.data

def test_workflow_state_validation():
    """Test WorkflowState validation."""
    state = WorkflowState()
    
    # Test required fields
    assert not state.validate()
    
    state.workflow_id = "test_workflow"
    assert state.validate()
    
    # Test status validation
    state.status = "invalid_status"
    assert not state.validate()
    
    state.status = "running"
    assert state.validate()

def test_state_manager_error_handling():
    """Test error handling in StateManager."""
    manager = StateManager()
    
    # Test getting non-existent state
    with pytest.raises(KeyError):
        manager.get_state("non_existent")
    
    # Test updating non-existent state
    with pytest.raises(KeyError):
        manager.update_state("non_existent", {})
    
    # Test deleting non-existent state
    with pytest.raises(KeyError):
        manager.delete_state("non_existent")

def test_state_persistence_error_handling():
    """Test error handling in StatePersistence."""
    persistence = StatePersistence()
    
    # Test loading non-existent state
    with pytest.raises(FileNotFoundError):
        persistence.load_state("non_existent")
    
    # Test invalid JSON data
    invalid_json = "invalid json"
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = invalid_json
        with pytest.raises(json.JSONDecodeError):
            persistence.load_state("test_workflow")

def test_state_validator_error_handling():
    """Test error handling in StateValidator."""
    validator = StateValidator(rules={})
    
    # Test validation with invalid rules
    with pytest.raises(ValueError):
        validator.validate({})
    
    # Test validation with invalid data type
    validator.rules = {"field_types": {"test": "invalid_type"}}
    with pytest.raises(ValueError):
        validator.validate({"test": "value"})

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "states"
    storage_dir.mkdir()
    return storage_dir

@pytest.fixture
def state_persistence(temp_storage_dir):
    """Create a StatePersistence instance."""
    return ConsolidatedStatePersistence(storage_dir=temp_storage_dir)

@pytest.fixture
def sample_state():
    """Create a sample WorkflowState."""
    return WorkflowState(
        workflow_id="test_workflow",
        state_id="test_state",
        context={"input": "test"},
        metadata={"user_id": "user123"}
    )

class TestStatePersistence:
    """Test suite for state persistence functionality."""
    
    def test_initialization(self, state_persistence, temp_storage_dir):
        """Test initialization of StatePersistence."""
        assert state_persistence.storage_dir == temp_storage_dir
        assert temp_storage_dir.exists()
        assert len(state_persistence._cache) == 0
    
    def test_save_load_state(self, state_persistence, sample_state):
        """Test saving and loading a state."""
        # Save state
        state_persistence.save(sample_state)
        
        # Check file exists
        state_file = state_persistence.storage_dir / f"{sample_state.state_id}.json"
        assert state_file.exists()
        
        # Load state
        loaded_state = state_persistence.load(sample_state.state_id)
        assert loaded_state is not None
        assert loaded_state.state_id == sample_state.state_id
        assert loaded_state.workflow_id == sample_state.workflow_id
        assert loaded_state.context == sample_state.context
        assert loaded_state.metadata == sample_state.metadata
    
    def test_load_nonexistent_state(self, state_persistence):
        """Test loading a non-existent state."""
        state = state_persistence.load("nonexistent")
        assert state is None
    
    def test_delete_state(self, state_persistence, sample_state):
        """Test deleting a state."""
        # Save state
        state_persistence.save(sample_state)
        
        # Delete state
        success = state_persistence.delete(sample_state.state_id)
        assert success
        
        # Check file doesn't exist
        state_file = state_persistence.storage_dir / f"{sample_state.state_id}.json"
        assert not state_file.exists()
        
        # Check cache is cleared
        assert sample_state.state_id not in state_persistence._cache
    
    def test_load_all_states(self, state_persistence):
        """Test loading all states."""
        # Create multiple states
        states = [
            WorkflowState(workflow_id="w1", state_id="s1", context={"i": 1}),
            WorkflowState(workflow_id="w2", state_id="s2", context={"i": 2}),
            WorkflowState(workflow_id="w3", state_id="s3", context={"i": 3})
        ]
        
        # Save states
        for state in states:
            state_persistence.save(state)
        
        # Load all states
        loaded_states = state_persistence.load_all()
        assert len(loaded_states) == len(states)
        for state in states:
            assert state.state_id in loaded_states
            loaded = loaded_states[state.state_id]
            assert loaded.workflow_id == state.workflow_id
            assert loaded.context == state.context
    
    def test_cache_functionality(self, state_persistence, sample_state):
        """Test cache functionality."""
        # Save state
        state_persistence.save(sample_state)
        
        # First load should cache
        first_load = state_persistence.load(sample_state.state_id)
        assert sample_state.state_id in state_persistence._cache
        
        # Second load should use cache
        second_load = state_persistence.load(sample_state.state_id)
        assert first_load is second_load
    
    def test_cache_cleanup(self, state_persistence):
        """Test cache cleanup."""
        # Create states
        states = [
            WorkflowState(workflow_id=f"w{i}", state_id=f"s{i}", context={"i": i})
            for i in range(3)
        ]
        
        # Save states
        for state in states:
            state_persistence.save(state)
        
        # Modify access times
        now = datetime.now().timestamp()
        state_persistence._last_access["s0"] = now - 4000  # Expired by idle
        state_persistence._last_access["s1"] = now - 2000  # Not expired
        state_persistence._last_access["s2"] = now - 100000  # Expired by age
        
        # Run cleanup
        removed = state_persistence.cleanup_cache(max_idle=3600, max_age=50000)
        assert removed == 2
        assert "s1" in state_persistence._cache
        assert "s0" not in state_persistence._cache
        assert "s2" not in state_persistence._cache
    
    def test_get_cache_stats(self, state_persistence, sample_state):
        """Test getting cache statistics."""
        # Save state
        state_persistence.save(sample_state)
        
        # Get stats
        stats = state_persistence.get_cache_stats()
        assert stats['size'] == 1
        assert stats['creation_times'] == 1
        assert stats['last_access'] == 1
        assert stats['ttl'] == 0  # No TTL set 