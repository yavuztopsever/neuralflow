"""
Tests for the WorkflowState class.
"""

import pytest
from datetime import datetime
from langgraph.core.state import WorkflowState

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
def workflow_state(sample_context, sample_metadata):
    """Fixture providing a sample workflow state."""
    return WorkflowState(
        workflow_id="workflow1",
        state_id="state1",
        context=sample_context,
        metadata=sample_metadata
    )

class TestWorkflowStateCreation:
    """Test cases for WorkflowState creation."""
    
    def test_create_workflow_state(self, sample_context):
        """Test creating a basic workflow state."""
        state = WorkflowState(
            workflow_id="workflow1",
            state_id="state1",
            context=sample_context
        )
        
        assert state.workflow_id == "workflow1"
        assert state.state_id == "state1"
        assert state.context == sample_context
        assert state.metadata == {}
        assert state.status == "pending"
        assert state.results == {}
        assert state.error is None
        assert state.history == []
    
    def test_create_workflow_state_with_metadata(self, sample_context, sample_metadata):
        """Test creating a workflow state with metadata."""
        state = WorkflowState(
            workflow_id="workflow1",
            state_id="state1",
            context=sample_context,
            metadata=sample_metadata
        )
        
        assert state.metadata == sample_metadata

class TestWorkflowStateUpdate:
    """Test cases for WorkflowState updates."""
    
    def test_update_state(self, workflow_state):
        """Test updating state with results."""
        results = {"output": "test output"}
        workflow_state.update(results)
        
        assert workflow_state.results == results
        assert workflow_state.status == "completed"
        assert workflow_state.error is None
        assert len(workflow_state.history) == 1
        
        history_entry = workflow_state.history[0]
        assert history_entry["results"] == results
        assert history_entry["status"] == "completed"
        assert history_entry["error"] is None
    
    def test_update_state_with_error(self, workflow_state):
        """Test updating state with error."""
        results = {"output": "test output"}
        error = "Test error"
        workflow_state.update(results, status="failed", error=error)
        
        assert workflow_state.results == results
        assert workflow_state.status == "failed"
        assert workflow_state.error == error
        assert len(workflow_state.history) == 1
        
        history_entry = workflow_state.history[0]
        assert history_entry["results"] == results
        assert history_entry["status"] == "failed"
        assert history_entry["error"] == error

class TestWorkflowStateStatus:
    """Test cases for WorkflowState status checks."""
    
    def test_status_checks(self, workflow_state):
        """Test status check methods."""
        assert workflow_state.is_pending()
        assert not workflow_state.is_completed()
        assert not workflow_state.is_failed()
        
        workflow_state.update({}, status="completed")
        assert not workflow_state.is_pending()
        assert workflow_state.is_completed()
        assert not workflow_state.is_failed()
        
        workflow_state.update({}, status="failed")
        assert not workflow_state.is_pending()
        assert not workflow_state.is_completed()
        assert workflow_state.is_failed()

class TestWorkflowStateSerialization:
    """Test cases for WorkflowState serialization."""
    
    def test_to_dict(self, workflow_state):
        """Test converting state to dictionary."""
        state_dict = workflow_state.to_dict()
        
        assert state_dict["workflow_id"] == workflow_state.workflow_id
        assert state_dict["state_id"] == workflow_state.state_id
        assert state_dict["context"] == workflow_state.context
        assert state_dict["metadata"] == workflow_state.metadata
        assert state_dict["status"] == workflow_state.status
        assert state_dict["results"] == workflow_state.results
        assert state_dict["error"] == workflow_state.error
        assert state_dict["history"] == workflow_state.history
    
    def test_from_dict(self, workflow_state):
        """Test creating state from dictionary."""
        state_dict = workflow_state.to_dict()
        new_state = WorkflowState.from_dict(state_dict)
        
        assert new_state.workflow_id == workflow_state.workflow_id
        assert new_state.state_id == workflow_state.state_id
        assert new_state.context == workflow_state.context
        assert new_state.metadata == workflow_state.metadata
        assert new_state.status == workflow_state.status
        assert new_state.results == workflow_state.results
        assert new_state.error == workflow_state.error
        assert new_state.history == workflow_state.history

class TestWorkflowStateHistory:
    """Test cases for WorkflowState history."""
    
    def test_history_management(self, workflow_state):
        """Test history management."""
        # First update
        workflow_state.update({"output": "first"})
        assert len(workflow_state.history) == 1
        
        # Second update
        workflow_state.update({"output": "second"})
        assert len(workflow_state.history) == 2
        
        # Check latest entry
        latest = workflow_state.get_latest_history_entry()
        assert latest["results"]["output"] == "second"
    
    def test_empty_history(self, workflow_state):
        """Test history with no entries."""
        assert workflow_state.get_latest_history_entry() is None 