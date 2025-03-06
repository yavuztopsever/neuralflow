"""
Note storage management utilities for the LangGraph application.
This module provides functionality for storing and retrieving notes.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.config import Config

logger = logging.getLogger(__name__)

class NoteStoreManager:
    """Manages note storage and retrieval operations."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the note store manager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.text_processor = TextProcessor()
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize the note storage directory."""
        try:
            self.storage_dir = Path(self.config.NOTES_DIR)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized note storage at {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize note storage: {e}")
            raise
    
    def save_note(self, note_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a note to storage.
        
        Args:
            note_id: Unique identifier for the note
            content: Note content
            metadata: Optional metadata for the note
            
        Returns:
            Path to the saved note file
            
        Raises:
            ValueError: If note_id or content is invalid
            RuntimeError: If saving fails
        """
        if not note_id or not content:
            raise ValueError("Note ID and content must be provided.")
            
        try:
            note_data = {
                'id': note_id,
                'content': content,
                'metadata': metadata or {},
                'created': datetime.now().isoformat(),
                'modified': datetime.now().isoformat()
            }
            
            file_path = self.storage_dir / f"{note_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(note_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved note {note_id} to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save note {note_id}: {e}")
            raise RuntimeError(f"Failed to save note: {e}")
    
    def load_note(self, note_id: str) -> Dict[str, Any]:
        """Load a note from storage.
        
        Args:
            note_id: Unique identifier for the note
            
        Returns:
            Dictionary containing note data
            
        Raises:
            FileNotFoundError: If the note doesn't exist
            ValueError: If the note data is invalid
        """
        try:
            file_path = self.storage_dir / f"{note_id}.json"
            if not file_path.exists():
                raise FileNotFoundError(f"Note not found: {note_id}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                note_data = json.load(f)
                
            return note_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid note data in {note_id}: {e}")
            raise ValueError(f"Invalid note data: {e}")
        except Exception as e:
            logger.error(f"Failed to load note {note_id}: {e}")
            raise
    
    def update_note(self, note_id: str, content: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update an existing note.
        
        Args:
            note_id: Unique identifier for the note
            content: New note content (optional)
            metadata: New metadata (optional)
            
        Returns:
            Updated note data
            
        Raises:
            FileNotFoundError: If the note doesn't exist
            ValueError: If the update data is invalid
        """
        try:
            note_data = self.load_note(note_id)
            
            if content is not None:
                note_data['content'] = content
            if metadata is not None:
                note_data['metadata'].update(metadata)
                
            note_data['modified'] = datetime.now().isoformat()
            
            file_path = self.storage_dir / f"{note_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(note_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Updated note {note_id}")
            return note_data
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {e}")
            raise
    
    def delete_note(self, note_id: str) -> bool:
        """Delete a note from storage.
        
        Args:
            note_id: Unique identifier for the note
            
        Returns:
            True if the note was deleted, False otherwise
        """
        try:
            file_path = self.storage_dir / f"{note_id}.json"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted note {note_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete note {note_id}: {e}")
            return False
    
    def list_notes(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """List all notes in storage.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of note data dictionaries
        """
        try:
            notes = []
            for file_path in self.storage_dir.glob(pattern):
                if file_path.is_file():
                    try:
                        note_id = file_path.stem
                        note_data = self.load_note(note_id)
                        notes.append(note_data)
                    except Exception as e:
                        logger.warning(f"Failed to load note from {file_path}: {e}")
                        continue
            return notes
        except Exception as e:
            logger.error(f"Failed to list notes: {e}")
            return []
    
    def search_notes(self, query: str, 
                    search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search notes by content or metadata.
        
        Args:
            query: Search query
            search_fields: Fields to search in (defaults to ['content'])
            
        Returns:
            List of matching note data dictionaries
        """
        if not query:
            return []
            
        search_fields = search_fields or ['content']
        query = query.lower()
        
        try:
            notes = self.list_notes()
            matches = []
            
            for note in notes:
                for field in search_fields:
                    if field in note:
                        if isinstance(note[field], str):
                            if query in note[field].lower():
                                matches.append(note)
                                break
                        elif isinstance(note[field], dict):
                            if self._search_dict(note[field], query):
                                matches.append(note)
                                break
                                
            return matches
        except Exception as e:
            logger.error(f"Failed to search notes: {e}")
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