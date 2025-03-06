"""
State persistence for workflow states.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)

class StatePersistence:
    """Handles persistence of workflow states."""
    
    def __init__(self, storage_dir: Path):
        """Initialize state persistence.
        
        Args:
            storage_dir: Directory for state storage
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, state: WorkflowState) -> None:
        """Save a state to storage.
        
        Args:
            state: WorkflowState to save
        """
        try:
            state_file = self.storage_dir / f"{state.state_id}.json"
            with open(state_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug(f"Saved state {state.state_id}")
        except Exception as e:
            logger.error(f"Failed to save state {state.state_id}: {e}")
            raise
    
    def load(self, state_id: str) -> Optional[WorkflowState]:
        """Load a state from storage.
        
        Args:
            state_id: ID of the state to load
            
        Returns:
            WorkflowState instance or None if not found
        """
        try:
            state_file = self.storage_dir / f"{state_id}.json"
            if not state_file.exists():
                return None
            
            with open(state_file, "r") as f:
                data = json.load(f)
                return WorkflowState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load state {state_id}: {e}")
            return None
    
    def load_all(self) -> Dict[str, WorkflowState]:
        """Load all states from storage.
        
        Returns:
            Dictionary mapping state IDs to WorkflowState instances
        """
        states = {}
        try:
            for state_file in self.storage_dir.glob("*.json"):
                try:
                    with open(state_file, "r") as f:
                        data = json.load(f)
                        state = WorkflowState.from_dict(data)
                        states[state.state_id] = state
                except Exception as e:
                    logger.error(f"Failed to load state from {state_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to load states: {e}")
        return states
    
    def delete(self, state_id: str) -> bool:
        """Delete a state from storage.
        
        Args:
            state_id: ID of the state to delete
            
        Returns:
            True if state was deleted, False otherwise
        """
        try:
            state_file = self.storage_dir / f"{state_id}.json"
            if state_file.exists():
                state_file.unlink()
                logger.debug(f"Deleted state {state_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False 