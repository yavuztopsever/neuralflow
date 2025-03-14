"""
State persistence for workflow states.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
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
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._last_access = {}
        self._creation_time = {}
        self._ttl = {}
        
    def save(self, state: WorkflowState) -> None:
        """Save a state to storage.
        
        Args:
            state: WorkflowState to save
        """
        try:
            state_file = self.storage_dir / f"{state.state_id}.json"
            state_data = state.to_dict()
            state_data['timestamp'] = datetime.now().isoformat()
            
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)
            
            # Update cache
            with self._cache_lock:
                self._cache[state.state_id] = state
                self._last_access[state.state_id] = datetime.now().timestamp()
                self._creation_time[state.state_id] = datetime.now().timestamp()
            
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
            # Check cache first
            with self._cache_lock:
                if state_id in self._cache:
                    self._last_access[state_id] = datetime.now().timestamp()
                    return self._cache[state_id]
            
            state_file = self.storage_dir / f"{state_id}.json"
            if not state_file.exists():
                return None
            
            with open(state_file, "r") as f:
                data = json.load(f)
                state = WorkflowState.from_dict(data)
                
                # Update cache
                with self._cache_lock:
                    self._cache[state_id] = state
                    self._last_access[state_id] = datetime.now().timestamp()
                    self._creation_time[state_id] = datetime.now().timestamp()
                
                return state
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
                        
                        # Update cache
                        with self._cache_lock:
                            self._cache[state.state_id] = state
                            self._last_access[state.state_id] = datetime.now().timestamp()
                            self._creation_time[state.state_id] = datetime.now().timestamp()
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
                
                # Clear from cache
                with self._cache_lock:
                    if state_id in self._cache:
                        del self._cache[state_id]
                    if state_id in self._last_access:
                        del self._last_access[state_id]
                    if state_id in self._creation_time:
                        del self._creation_time[state_id]
                    if state_id in self._ttl:
                        del self._ttl[state_id]
                
                logger.debug(f"Deleted state {state_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False
    
    def cleanup_cache(self, max_idle: int = 3600, max_age: int = 86400) -> int:
        """Remove expired or old cache entries.
        
        Args:
            max_idle: Maximum idle time in seconds
            max_age: Maximum age in seconds
            
        Returns:
            Number of entries removed
        """
        now = datetime.now().timestamp()
        to_remove = []
        
        with self._cache_lock:
            for state_id, last_access in self._last_access.items():
                if now - last_access > max_idle:
                    to_remove.append(state_id)
                elif state_id in self._creation_time and now - self._creation_time[state_id] > max_age:
                    to_remove.append(state_id)
                elif state_id in self._ttl and now - self._creation_time[state_id] > self._ttl[state_id]:
                    to_remove.append(state_id)
            
            for state_id in to_remove:
                if state_id in self._cache:
                    del self._cache[state_id]
                if state_id in self._last_access:
                    del self._last_access[state_id]
                if state_id in self._creation_time:
                    del self._creation_time[state_id]
                if state_id in self._ttl:
                    del self._ttl[state_id]
                    
        return len(to_remove)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self._cache_lock:
            return {
                'size': len(self._cache),
                'last_cleanup': getattr(self, '_last_cleanup', None),
                'creation_times': len(self._creation_time),
                'last_access': len(self._last_access),
                'ttl': len(self._ttl)
            } 