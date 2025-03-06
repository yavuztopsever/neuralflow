"""
Tests for the StateValidator class.
"""

import pytest
from langgraph.core.state import StateValidator, WorkflowState

@pytest.fixture
def sample_context():
    """Fixture providing a sample context."""
    return {
        "input": "test input",
        "parameters": {"param1": "value1"}
    }

@pytest.fixture
def valid_state(sample_context):
    """Fixture providing a valid workflow state."""
    return WorkflowState(
        workflow_id="workflow1",
        state_id="state1",
        context=sample_context
    )

@pytest.fixture
def validator():
    """Fixture providing a StateValidator instance."""
    return StateValidator()

class TestStateValidator:
    """Test cases for state validation."""
    
    def test_validate_valid_state(self, validator, valid_state):
        """Test validating a valid state."""
        # Should not raise any exceptions
        validator.validate_state(valid_state)
    
    def test_validate_missing_workflow_id(self, validator, sample_context):
        """Test validating state with missing workflow_id."""
        state = WorkflowState(
            workflow_id="",  # Empty workflow_id
            state_id="state1",
            context=sample_context
        )
        
        with pytest.raises(ValueError, match="workflow_id is required"):
            validator.validate_state(state)
    
    def test_validate_missing_state_id(self, validator, sample_context):
        """Test validating state with missing state_id."""
        state = WorkflowState(
            workflow_id="workflow1",
            state_id="",  # Empty state_id
            context=sample_context
        )
        
        with pytest.raises(ValueError, match="state_id is required"):
            validator.validate_state(state)
    
    def test_validate_missing_context(self, validator):
        """Test validating state with missing context."""
        state = WorkflowState(
            workflow_id="workflow1",
            state_id="state1",
            context=None  # Missing context
        )
        
        with pytest.raises(ValueError, match="context is required"):
            validator.validate_state(state)
    
    def test_validate_invalid_status(self, validator, valid_state):
        """Test validating state with invalid status."""
        valid_state.status = "invalid_status"
        
        with pytest.raises(ValueError, match="Invalid status"):
            validator.validate_state(valid_state)
    
    def test_validate_invalid_context_type(self, validator):
        """Test validating state with invalid context type."""
        state = WorkflowState(
            workflow_id="workflow1",
            state_id="state1",
            context="not_a_dict"  # Context should be a dict
        )
        
        with pytest.raises(ValueError, match="context must be a dictionary"):
            validator.validate_state(state)
    
    def test_validate_invalid_metadata_type(self, validator, sample_context):
        """Test validating state with invalid metadata type."""
        state = WorkflowState(
            workflow_id="workflow1",
            state_id="state1",
            context=sample_context,
            metadata="not_a_dict"  # Metadata should be a dict
        )
        
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            validator.validate_state(state) 