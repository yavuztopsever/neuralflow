"""
State manager for handling workflow states.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json

from .workflow_state import WorkflowState
from .state_validator import StateValidator
from .state_persistence import StatePersistence

logger = logging.getLogger(__name__)

class StateManager:
    """Manages workflow states."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize the state manager.
        
        Args:
            storage_dir: Directory for state storage
        """
        self.storage_dir = storage_dir or Path("storage/states")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = StateValidator()
        self.persistence = StatePersistence(self.storage_dir)
        self._states: Dict[str, WorkflowState] = {}
        
        # Load existing states
        self._load_states()
    
    def _load_states(self) -> None:
        """Load states from storage."""
        try:
            self._states = self.persistence.load_all()
            logger.info(f"Loaded {len(self._states)} states")
        except Exception as e:
            logger.error(f"Failed to load states: {e}")
            raise
    
    def create_state(self, workflow_id: str,
                    context: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """Create a new workflow state.
        
        Args:
            workflow_id: ID of the workflow
            context: Execution context
            metadata: Optional metadata
            
        Returns:
            Created workflow state
        """
        try:
            # Generate state ID
            state_id = f"{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create state
            state = WorkflowState(
                workflow_id=workflow_id,
                state_id=state_id,
                context=context,
                metadata=metadata or {}
            )
            
            # Validate state
            self.validator.validate_state(state)
            
            # Save state
            self._states[state_id] = state
            self.persistence.save(state)
            
            logger.info(f"Created state {state_id}")
            return state
        except Exception as e:
            logger.error(f"Failed to create state: {e}")
            raise
    
    def get_state(self, state_id: str) -> Optional[WorkflowState]:
        """Get a state by ID.
        
        Args:
            state_id: State ID
            
        Returns:
            WorkflowState instance or None if not found
        """
        return self._states.get(state_id)
    
    def update_state(self, state_id: str,
                    results: Dict[str, Any],
                    status: str = "completed",
                    error: Optional[str] = None) -> bool:
        """Update a state with execution results.
        
        Args:
            state_id: State ID
            results: Execution results
            status: New status
            error: Optional error message
            
        Returns:
            True if state was updated, False otherwise
        """
        try:
            state = self.get_state(state_id)
            if not state:
                return False
            
            # Update state
            state.update(results, status, error)
            
            # Validate state
            self.validator.validate_state(state)
            
            # Save state
            self.persistence.save(state)
            
            logger.info(f"Updated state {state_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update state {state_id}: {e}")
            return False
    
    def delete_state(self, state_id: str) -> bool:
        """Delete a state.
        
        Args:
            state_id: State ID
            
        Returns:
            True if state was deleted, False otherwise
        """
        try:
            state = self.get_state(state_id)
            if not state:
                return False
            
            # Remove from memory
            del self._states[state_id]
            
            # Remove from storage
            self.persistence.delete(state_id)
            
            logger.info(f"Deleted state {state_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False
    
    def get_workflow_states(self, workflow_id: str) -> List[WorkflowState]:
        """Get all states for a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            List of workflow states
        """
        return [
            state for state in self._states.values()
            if state.workflow_id == workflow_id
        ]
    
    def get_state_stats(self) -> Dict[str, Any]:
        """Get statistics about states.
        
        Returns:
            Dictionary containing state statistics
        """
        stats = {
            "total_states": len(self._states),
            "workflows": {},
            "status_counts": {}
        }
        
        for state in self._states.values():
            # Count by workflow
            if state.workflow_id not in stats["workflows"]:
                stats["workflows"][state.workflow_id] = 0
            stats["workflows"][state.workflow_id] += 1
            
            # Count by status
            if state.status not in stats["status_counts"]:
                stats["status_counts"][state.status] = 0
            stats["status_counts"][state.status] += 1
        
        return stats 