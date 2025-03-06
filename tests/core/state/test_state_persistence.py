"""
Tests for the StatePersistence class.
"""

import pytest
from pathlib import Path
import json
import shutil
from langgraph.core.state import StatePersistence, WorkflowState

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Fixture providing a temporary storage directory."""
    storage_dir = tmp_path / "storage" / "states"
    storage_dir.mkdir(parents=True)
    yield storage_dir
    shutil.rmtree(storage_dir)

@pytest.fixture
def sample_context():
    """Fixture providing a sample context."""
    return {
        "input": "test input",
        "parameters": {"param1": "value1"}
    }

@pytest.fixture
def sample_state(sample_context):
    """Fixture providing a sample workflow state."""
    return WorkflowState(
        workflow_id="workflow1",
        state_id="state1",
        context=sample_context
    )

@pytest.fixture
def persistence(temp_storage_dir):
    """Fixture providing a StatePersistence instance."""
    return StatePersistence(storage_dir=temp_storage_dir)

class TestStatePersistence:
    """Test cases for state persistence."""
    
    def test_save_state(self, persistence, sample_state):
        """Test saving a state."""
        persistence.save(sample_state)
        
        # Check if file exists
        state_file = persistence.storage_dir / f"{sample_state.state_id}.json"
        assert state_file.exists()
        
        # Check file contents
        with open(state_file, "r") as f:
            data = json.load(f)
            assert data["workflow_id"] == sample_state.workflow_id
            assert data["state_id"] == sample_state.state_id
            assert data["context"] == sample_state.context
    
    def test_load_state(self, persistence, sample_state):
        """Test loading a state."""
        # Save state first
        persistence.save(sample_state)
        
        # Load state
        loaded = persistence.load(sample_state.state_id)
        assert loaded is not None
        assert loaded.workflow_id == sample_state.workflow_id
        assert loaded.state_id == sample_state.state_id
        assert loaded.context == sample_state.context
    
    def test_load_nonexistent_state(self, persistence):
        """Test loading a nonexistent state."""
        assert persistence.load("nonexistent") is None
    
    def test_load_all_states(self, persistence, sample_context):
        """Test loading all states."""
        # Create and save multiple states
        states = []
        for i in range(3):
            state = WorkflowState(
                workflow_id=f"workflow{i}",
                state_id=f"state{i}",
                context=sample_context
            )
            states.append(state)
            persistence.save(state)
        
        # Load all states
        loaded = persistence.load_all()
        
        # Check loaded states
        assert len(loaded) == 3
        for state in states:
            assert state.state_id in loaded
            assert loaded[state.state_id].workflow_id == state.workflow_id
    
    def test_delete_state(self, persistence, sample_state):
        """Test deleting a state."""
        # Save state first
        persistence.save(sample_state)
        
        # Delete state
        success = persistence.delete(sample_state.state_id)
        assert success
        
        # Check if file is deleted
        state_file = persistence.storage_dir / f"{sample_state.state_id}.json"
        assert not state_file.exists()
    
    def test_delete_nonexistent_state(self, persistence):
        """Test deleting a nonexistent state."""
        assert not persistence.delete("nonexistent")
    
    def test_load_corrupted_state(self, persistence, sample_state):
        """Test loading a corrupted state file."""
        # Create a corrupted state file
        state_file = persistence.storage_dir / f"{sample_state.state_id}.json"
        with open(state_file, "w") as f:
            f.write("invalid json")
        
        # Try to load the corrupted state
        loaded = persistence.load(sample_state.state_id)
        assert loaded is None
    
    def test_save_state_with_history(self, persistence, sample_state):
        """Test saving and loading state with history."""
        # Update state to create history
        sample_state.update({"output": "test"}, status="completed")
        
        # Save state
        persistence.save(sample_state)
        
        # Load state
        loaded = persistence.load(sample_state.state_id)
        assert loaded is not None
        assert len(loaded.history) == 1
        assert loaded.history[0]["status"] == "completed"
        assert loaded.history[0]["results"]["output"] == "test" 