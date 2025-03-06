"""
State management utilities for the LangGraph application.
This module provides functionality for managing application state and caching.
"""

import os
import logging
import json
import threading
import time
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.config import Config

logger = logging.getLogger(__name__)

class StateManager:
    """Manages application state and caching operations."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the state manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.text_processor = TextProcessor()
        self._initialize_storage()
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._last_access = {}
        self._creation_time = {}
    
    def _initialize_storage(self):
        """Initialize the state storage directory."""
        try:
            self.storage_dir = Path(self.config.STATE_DIR)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized state storage at {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize state storage: {e}")
            raise
    
    def save_state(self, state_id: str, state_data: Dict[str, Any]) -> str:
        """Save application state to storage.
        
        Args:
            state_id: Unique identifier for the state
            state_data: State data to save
            
        Returns:
            Path to the saved state file
            
        Raises:
            ValueError: If state_id or state_data is invalid
            RuntimeError: If saving fails
        """
        if not state_id or not isinstance(state_data, dict):
            raise ValueError("State ID and data must be provided.")
            
        try:
            state_data['id'] = state_id
            state_data['timestamp'] = datetime.now().isoformat()
            
            file_path = self.storage_dir / f"{state_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved state {state_id} to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save state {state_id}: {e}")
            raise RuntimeError(f"Failed to save state: {e}")
    
    def load_state(self, state_id: str) -> Dict[str, Any]:
        """Load application state from storage.
        
        Args:
            state_id: Unique identifier for the state
            
        Returns:
            Dictionary containing state data
            
        Raises:
            FileNotFoundError: If the state doesn't exist
            ValueError: If the state data is invalid
        """
        try:
            file_path = self.storage_dir / f"{state_id}.json"
            if not file_path.exists():
                raise FileNotFoundError(f"State not found: {state_id}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                
            return state_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid state data in {state_id}: {e}")
            raise ValueError(f"Invalid state data: {e}")
        except Exception as e:
            logger.error(f"Failed to load state {state_id}: {e}")
            raise
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._cache_lock:
            value = self._cache.get(key)
            if value is not None:
                self._last_access[key] = time.time()
            return value
    
    def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        with self._cache_lock:
            self._cache[key] = value
            self._last_access[key] = time.time()
            self._creation_time[key] = time.time()
            if ttl is not None:
                self._ttl[key] = ttl
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear values from the cache.
        
        Args:
            key: Specific key to clear (if None, clears entire cache)
        """
        with self._cache_lock:
            if key is None:
                self._cache.clear()
                self._last_access.clear()
                self._creation_time.clear()
                self._ttl.clear()
            elif key in self._cache:
                del self._cache[key]
                if key in self._last_access:
                    del self._last_access[key]
                if key in self._creation_time:
                    del self._creation_time[key]
                if key in self._ttl:
                    del self._ttl[key]
    
    def cleanup_cache(self, max_idle: int = 3600, max_age: int = 86400) -> int:
        """Remove expired or old cache entries.
        
        Args:
            max_idle: Maximum idle time in seconds
            max_age: Maximum age in seconds
            
        Returns:
            Number of entries removed
        """
        now = time.time()
        to_remove = []
        
        with self._cache_lock:
            for key, last_access in self._last_access.items():
                if now - last_access > max_idle:
                    to_remove.append(key)
                elif key in self._creation_time and now - self._creation_time[key] > max_age:
                    to_remove.append(key)
                elif key in self._ttl and now - self._creation_time[key] > self._ttl[key]:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._cache[key]
                if key in self._last_access:
                    del self._last_access[key]
                if key in self._creation_time:
                    del self._creation_time[key]
                if key in self._ttl:
                    del self._ttl[key]
                    
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
    
    def list_states(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """List all states in storage.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of state data dictionaries
        """
        try:
            states = []
            for file_path in self.storage_dir.glob(pattern):
                if file_path.is_file():
                    try:
                        state_id = file_path.stem
                        state_data = self.load_state(state_id)
                        states.append(state_data)
                    except Exception as e:
                        logger.warning(f"Failed to load state from {file_path}: {e}")
                        continue
            return states
        except Exception as e:
            logger.error(f"Failed to list states: {e}")
            return []
    
    def search_states(self, query: str, 
                     search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search states by content or metadata.
        
        Args:
            query: Search query
            search_fields: Fields to search in (defaults to ['data'])
            
        Returns:
            List of matching state data dictionaries
        """
        if not query:
            return []
            
        search_fields = search_fields or ['data']
        query = query.lower()
        
        try:
            states = self.list_states()
            matches = []
            
            for state in states:
                for field in search_fields:
                    if field in state:
                        if isinstance(state[field], str):
                            if query in state[field].lower():
                                matches.append(state)
                                break
                        elif isinstance(state[field], dict):
                            if self._search_dict(state[field], query):
                                matches.append(state)
                                break
                                
            return matches
        except Exception as e:
            logger.error(f"Failed to search states: {e}")
            return []
    
    def _search_dict(self, d: Dict[str, Any], query: str) -> bool:
        """Recursively search a dictionary for a query string.
        
        Args:
            d: Dictionary to search
            query: Query string to search for
            
        Returns:
            True if the query is found, False otherwise
        """
        for value in d.values():
            if isinstance(value, str):
                if query in value.lower():
                    return True
            elif isinstance(value, dict):
                if self._search_dict(value, query):
                    return True
        return False 