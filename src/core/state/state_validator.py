"""
State validation for workflow states.
"""

from typing import Dict, Any
from .workflow_state import WorkflowState

class StateValidator:
    """Validates workflow states."""
    
    def validate_state(self, state: WorkflowState) -> None:
        """Validate a workflow state.
        
        Args:
            state: WorkflowState to validate
            
        Raises:
            ValueError: If state is invalid
        """
        self._validate_required_fields(state)
        self._validate_status(state)
        self._validate_context(state)
        self._validate_metadata(state)
    
    def _validate_required_fields(self, state: WorkflowState) -> None:
        """Validate required fields are present."""
        if not state.workflow_id:
            raise ValueError("workflow_id is required")
        if not state.state_id:
            raise ValueError("state_id is required")
        if state.context is None:
            raise ValueError("context is required")
    
    def _validate_status(self, state: WorkflowState) -> None:
        """Validate status field."""
        valid_statuses = {"pending", "running", "completed", "failed"}
        if state.status not in valid_statuses:
            raise ValueError(f"Invalid status: {state.status}")
    
    def _validate_context(self, state: WorkflowState) -> None:
        """Validate context field."""
        if not isinstance(state.context, dict):
            raise ValueError("context must be a dictionary")
    
    def _validate_metadata(self, state: WorkflowState) -> None:
        """Validate metadata field."""
        if not isinstance(state.metadata, dict):
            raise ValueError("metadata must be a dictionary") 