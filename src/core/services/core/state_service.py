"""
State service for the LangGraph project.
This service provides state management capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

from ..utils.state_manager import StateManager

class StateService:
    """Service for managing state in the LangGraph system."""
    
    def __init__(self):
        """Initialize the state service."""
        self.state_managers: Dict[str, StateManager] = {}
        self.history: List[Dict[str, Any]] = []
    
    def create_state_manager(self, name: str) -> StateManager:
        """
        Create a new state manager.
        
        Args:
            name: State manager name
            
        Returns:
            StateManager: Created state manager
        """
        manager = StateManager()
        self.state_managers[name] = manager
        return manager
    
    def get_state_manager(self, name: str) -> Optional[StateManager]:
        """
        Get a state manager by name.
        
        Args:
            name: State manager name
            
        Returns:
            Optional[StateManager]: State manager if found, None otherwise
        """
        return self.state_managers.get(name)
    
    def get_state(self, name: str) -> Dict[str, Any]:
        """
        Get the current state of a state manager.
        
        Args:
            name: State manager name
            
        Returns:
            Dict[str, Any]: Current state
        """
        manager = self.get_state_manager(name)
        if not manager:
            raise ValueError(f"State manager not found: {name}")
        return manager.get_state()
    
    def set_state(self, name: str, state: Dict[str, Any]) -> None:
        """
        Set the state of a state manager.
        
        Args:
            name: State manager name
            state: New state
        """
        manager = self.get_state_manager(name)
        if not manager:
            raise ValueError(f"State manager not found: {name}")
        manager.set_state(state)
        
        # Record in history
        self.history.append({
            'timestamp': datetime.now(),
            'manager': name,
            'state': state
        })
    
    def save_checkpoint(self, name: str, checkpoint_id: str) -> None:
        """
        Save a state checkpoint.
        
        Args:
            name: State manager name
            checkpoint_id: Checkpoint ID
        """
        manager = self.get_state_manager(name)
        if not manager:
            raise ValueError(f"State manager not found: {name}")
        manager.save_checkpoint(checkpoint_id)
        
        # Record in history
        self.history.append({
            'timestamp': datetime.now(),
            'manager': name,
            'checkpoint_id': checkpoint_id,
            'action': 'save_checkpoint'
        })
    
    def load_checkpoint(self, name: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a state checkpoint.
        
        Args:
            name: State manager name
            checkpoint_id: Checkpoint ID
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint state if found, None otherwise
        """
        manager = self.get_state_manager(name)
        if not manager:
            raise ValueError(f"State manager not found: {name}")
        
        state = manager.load_checkpoint(checkpoint_id)
        
        # Record in history
        self.history.append({
            'timestamp': datetime.now(),
            'manager': name,
            'checkpoint_id': checkpoint_id,
            'action': 'load_checkpoint',
            'state': state
        })
        
        return state
    
    def get_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the state history.
        
        Args:
            name: Optional state manager name to filter history
            
        Returns:
            List[Dict[str, Any]]: State history
        """
        if name:
            return [entry for entry in self.history if entry.get('manager') == name]
        return self.history
    
    def clear_history(self, name: Optional[str] = None) -> None:
        """
        Clear the state history.
        
        Args:
            name: Optional state manager name to clear history for
        """
        if name:
            self.history = [entry for entry in self.history if entry.get('manager') != name]
        else:
            self.history = []
    
    def clear_checkpoints(self, name: Optional[str] = None) -> None:
        """
        Clear state checkpoints.
        
        Args:
            name: Optional state manager name to clear checkpoints for
        """
        if name:
            manager = self.get_state_manager(name)
            if manager:
                manager.clear_checkpoints()
        else:
            for manager in self.state_managers.values():
                manager.clear_checkpoints()
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset state managers.
        
        Args:
            name: Optional state manager name to reset
        """
        if name:
            manager = self.get_state_manager(name)
            if manager:
                manager.reset()
        else:
            for manager in self.state_managers.values():
                manager.reset()
            self.state_managers = {}
            self.history = []

__all__ = ['StateService'] 