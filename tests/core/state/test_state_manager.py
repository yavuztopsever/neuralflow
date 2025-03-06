"""
Tests for the StateManager class.
"""

import pytest
from pathlib import Path
import shutil
from langgraph.core.state import StateManager, WorkflowState

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
def sample_metadata():
    """Fixture providing sample metadata."""
    return {
        "user_id": "user123",
        "priority": "high"
    }

@pytest.fixture
def state_manager(temp_storage_dir):
    """Fixture providing a StateManager instance."""
    return StateManager(storage_dir=temp_storage_dir)

class TestStateManagerCreation:
    """Test cases for StateManager creation."""
    
    def test_create_state_manager(self, temp_storage_dir):
        """Test creating a StateManager instance."""
        manager = StateManager(storage_dir=temp_storage_dir)
        assert manager.storage_dir == temp_storage_dir
        assert len(manager._states) == 0

class TestStateManagerOperations:
    """Test cases for StateManager operations."""
    
    def test_create_state(self, state_manager, sample_context, sample_metadata):
        """Test creating a new state."""
        state = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context,
            metadata=sample_metadata
        )
        
        assert state.workflow_id == "workflow1"
        assert state.context == sample_context
        assert state.metadata == sample_metadata
        assert state.state_id in state_manager._states
        assert state_manager.get_state(state.state_id) == state
    
    def test_get_state(self, state_manager, sample_context):
        """Test getting a state."""
        state = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context
        )
        
        retrieved = state_manager.get_state(state.state_id)
        assert retrieved == state
    
    def test_get_nonexistent_state(self, state_manager):
        """Test getting a nonexistent state."""
        assert state_manager.get_state("nonexistent") is None
    
    def test_update_state(self, state_manager, sample_context):
        """Test updating a state."""
        state = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context
        )
        
        results = {"output": "test output"}
        success = state_manager.update_state(
            state.state_id,
            results=results,
            status="completed"
        )
        
        assert success
        updated = state_manager.get_state(state.state_id)
        assert updated.results == results
        assert updated.status == "completed"
    
    def test_update_nonexistent_state(self, state_manager):
        """Test updating a nonexistent state."""
        success = state_manager.update_state(
            "nonexistent",
            results={},
            status="completed"
        )
        assert not success
    
    def test_delete_state(self, state_manager, sample_context):
        """Test deleting a state."""
        state = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context
        )
        
        success = state_manager.delete_state(state.state_id)
        assert success
        assert state_manager.get_state(state.state_id) is None
    
    def test_delete_nonexistent_state(self, state_manager):
        """Test deleting a nonexistent state."""
        success = state_manager.delete_state("nonexistent")
        assert not success

class TestStateManagerWorkflow:
    """Test cases for workflow-related operations."""
    
    def test_get_workflow_states(self, state_manager, sample_context):
        """Test getting states for a workflow."""
        # Create states for different workflows
        state1 = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context
        )
        state2 = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context
        )
        state3 = state_manager.create_state(
            workflow_id="workflow2",
            context=sample_context
        )
        
        # Get states for workflow1
        workflow_states = state_manager.get_workflow_states("workflow1")
        assert len(workflow_states) == 2
        assert {s.state_id for s in workflow_states} == {state1.state_id, state2.state_id}
    
    def test_get_workflow_states_empty(self, state_manager):
        """Test getting states for a workflow with no states."""
        assert len(state_manager.get_workflow_states("nonexistent")) == 0

class TestStateManagerStats:
    """Test cases for state statistics."""
    
    def test_get_state_stats(self, state_manager, sample_context):
        """Test getting state statistics."""
        # Create states with different statuses
        state1 = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context
        )
        state2 = state_manager.create_state(
            workflow_id="workflow1",
            context=sample_context
        )
        state3 = state_manager.create_state(
            workflow_id="workflow2",
            context=sample_context
        )
        
        # Update some states
        state_manager.update_state(state1.state_id, {}, status="completed")
        state_manager.update_state(state2.state_id, {}, status="failed")
        
        # Get stats
        stats = state_manager.get_state_stats()
        
        assert stats["total_states"] == 3
        assert stats["workflows"]["workflow1"] == 2
        assert stats["workflows"]["workflow2"] == 1
        assert stats["status_counts"]["pending"] == 1
        assert stats["status_counts"]["completed"] == 1
        assert stats["status_counts"]["failed"] == 1 