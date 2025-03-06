"""
State management utilities specific to the LangGraph project.
These utilities handle state management that is not covered by LangChain's built-in state management.
"""

import json
import pickle
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

T = TypeVar('T')

class StateType(Enum):
    """Types of state that can be stored."""
    DICT = "dict"
    LIST = "list"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CUSTOM = "custom"

@dataclass
class StateMetadata:
    """Metadata about a state."""
    created_at: str
    updated_at: str
    version: str
    type: StateType
    description: Optional[str] = None
    tags: List[str] = None

class StateManager(Generic[T]):
    """Manages state for the LangGraph project."""
    
    def __init__(
        self,
        state_dir: str = "state",
        state_version: str = "1.0.0",
        auto_save: bool = True
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_version = state_version
        self.auto_save = auto_save
        
        # Initialize state
        self.state: Dict[str, Any] = {}
        self.metadata: Dict[str, StateMetadata] = {}
        
        # Load existing state if available
        self._load_state()

    def _load_state(self):
        """Load state from file if it exists."""
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                self.state = data.get('state', {})
                self.metadata = {
                    k: StateMetadata(**v)
                    for k, v in data.get('metadata', {}).items()
                }

    def _save_state(self):
        """Save state to file if auto_save is enabled."""
        if not self.auto_save:
            return
            
        state_file = self.state_dir / "state.json"
        data = {
            'state': self.state,
            'metadata': {
                k: asdict(v)
                for k, v in self.metadata.items()
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def set_state(
        self,
        key: str,
        value: Any,
        state_type: Optional[StateType] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Set a state value with metadata."""
        self.state[key] = value
        
        # Determine state type if not provided
        if state_type is None:
            if isinstance(value, dict):
                state_type = StateType.DICT
            elif isinstance(value, list):
                state_type = StateType.LIST
            elif isinstance(value, str):
                state_type = StateType.STRING
            elif isinstance(value, (int, float)):
                state_type = StateType.NUMBER
            elif isinstance(value, bool):
                state_type = StateType.BOOLEAN
            else:
                state_type = StateType.CUSTOM
        
        # Update metadata
        now = datetime.now().isoformat()
        self.metadata[key] = StateMetadata(
            created_at=now,
            updated_at=now,
            version=self.state_version,
            type=state_type,
            description=description,
            tags=tags or []
        )
        
        self._save_state()

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)

    def delete_state(self, key: str):
        """Delete a state value and its metadata."""
        if key in self.state:
            del self.state[key]
            del self.metadata[key]
            self._save_state()

    def get_state_metadata(self, key: str) -> Optional[StateMetadata]:
        """Get metadata for a state value."""
        return self.metadata.get(key)

    def update_state(
        self,
        key: str,
        value: Any,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Update a state value and its metadata."""
        if key not in self.state:
            raise KeyError(f"State key not found: {key}")
            
        self.state[key] = value
        self.metadata[key].updated_at = datetime.now().isoformat()
        
        if description is not None:
            self.metadata[key].description = description
        if tags is not None:
            self.metadata[key].tags = tags
            
        self._save_state()

    def get_all_state(self) -> Dict[str, Any]:
        """Get all state values."""
        return self.state.copy()

    def get_all_metadata(self) -> Dict[str, StateMetadata]:
        """Get all state metadata."""
        return self.metadata.copy()

    def clear_state(self):
        """Clear all state and metadata."""
        self.state.clear()
        self.metadata.clear()
        self._save_state()

    def get_states_by_type(self, state_type: StateType) -> Dict[str, Any]:
        """Get all state values of a specific type."""
        return {
            k: v for k, v in self.state.items()
            if self.metadata[k].type == state_type
        }

    def get_states_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get all state values with a specific tag."""
        return {
            k: v for k, v in self.state.items()
            if tag in self.metadata[k].tags
        }

    def export_state(self, filepath: Union[str, Path]):
        """Export state to a file."""
        filepath = Path(filepath)
        data = {
            'state': self.state,
            'metadata': {
                k: asdict(v)
                for k, v in self.metadata.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_state(self, filepath: Union[str, Path]):
        """Import state from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.state = data.get('state', {})
            self.metadata = {
                k: StateMetadata(**v)
                for k, v in data.get('metadata', {}).items()
            }
            
        self._save_state()

    def get_state_history(self, key: str) -> List[Dict[str, Any]]:
        """Get history of state changes for a key."""
        history_file = self.state_dir / f"{key}_history.json"
        if not history_file.exists():
            return []
            
        with open(history_file, 'r') as f:
            return json.load(f)

    def save_state_history(self, key: str, value: Any):
        """Save state change to history."""
        history_file = self.state_dir / f"{key}_history.json"
        history = self.get_state_history(key)
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'version': self.state_version
        })
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def validate_state(self, key: str) -> List[str]:
        """Validate a state value."""
        errors = []
        
        if key not in self.state:
            errors.append(f"State key not found: {key}")
            return errors
            
        # Add validation rules based on state type
        state_type = self.metadata[key].type
        value = self.state[key]
        
        if state_type == StateType.DICT and not isinstance(value, dict):
            errors.append(f"Invalid type for dict state: {type(value)}")
        elif state_type == StateType.LIST and not isinstance(value, list):
            errors.append(f"Invalid type for list state: {type(value)}")
        elif state_type == StateType.STRING and not isinstance(value, str):
            errors.append(f"Invalid type for string state: {type(value)}")
        elif state_type == StateType.NUMBER and not isinstance(value, (int, float)):
            errors.append(f"Invalid type for number state: {type(value)}")
        elif state_type == StateType.BOOLEAN and not isinstance(value, bool):
            errors.append(f"Invalid type for boolean state: {type(value)}")
        
        return errors 