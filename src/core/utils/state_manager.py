"""
State management utilities for the LangGraph project.
These utilities provide state management capabilities for workflows.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class StateManager:
    """Manager for handling workflow state."""
    
    def __init__(self):
        """Initialize the state manager."""
        self.state: Dict[str, Any] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.
        
        Returns:
            Dict[str, Any]: Current state
        """
        return self.state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the current state.
        
        Args:
            state: New state
        """
        self.state = state
        self.history.append({
            'timestamp': datetime.now(),
            'state': state.copy()
        })
    
    def save_checkpoint(self, checkpoint_id: str) -> None:
        """
        Save the current state as a checkpoint.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
        """
        self.checkpoints[checkpoint_id] = self.state.copy()
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a state checkpoint.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint state if found, None otherwise
        """
        return self.checkpoints.get(checkpoint_id)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the state history.
        
        Returns:
            List[Dict[str, Any]]: State history
        """
        return self.history
    
    def clear_history(self) -> None:
        """Clear the state history."""
        self.history = []
    
    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        self.checkpoints = {}
    
    def reset(self) -> None:
        """Reset the state manager to its initial state."""
        self.state = {}
        self.checkpoints = {}
        self.history = []

__all__ = ['StateManager'] 