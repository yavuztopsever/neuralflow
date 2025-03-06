"""
Core engine for the LangGraph application.
This module provides the main workflow and orchestration functionality.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager
from utils.logging.manager import LogManager
from storage.document.processor import DocumentProcessor
from storage.document.retriever import DocumentRetriever
from storage.note.manager import NoteStoreManager
from storage.state.manager import StateManager
from storage.vector.providers.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

class LangGraphEngine:
    """Main engine for the LangGraph application."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the LangGraph engine.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Initialize storage components
            self.document_processor = DocumentProcessor(self.config)
            self.document_retriever = DocumentRetriever(self.config)
            self.note_manager = NoteStoreManager(self.config)
            self.state_manager = StateManager(self.config)
            self.vector_store = ChromaVectorStore(self.config)
            
            logger.info("Initialized all components")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_document(self, file_path: Union[str, Path], 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a document and add it to the system.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
            
        Returns:
            Document ID
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
        """
        try:
            # Process and add document
            doc_id = self.document_retriever.add_document(file_path, metadata)
            
            # Create note from document
            doc = self.document_processor.load_document(file_path)
            note_id = f"doc_{doc_id}"
            self.note_manager.save_note(
                note_id,
                doc['content'],
                {**doc['metadata'], 'source': 'document', 'doc_id': doc_id}
            )
            
            logger.info(f"Processed document {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise
    
    def process_directory(self, directory: Union[str, Path],
                         file_pattern: str = "*.*",
                         recursive: bool = True,
                         metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Process all documents in a directory.
        
        Args:
            directory: Path to the directory
            file_pattern: Pattern to match files
            recursive: Whether to process subdirectories
            metadata: Optional metadata for all documents
            
        Returns:
            List of document IDs
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        try:
            # Process and add documents
            doc_ids = self.document_retriever.add_directory(
                directory, file_pattern, recursive, metadata
            )
            
            # Create notes from documents
            for doc_id in doc_ids:
                doc = self.document_retriever.get_document(doc_id)
                if doc:
                    note_id = f"doc_{doc_id}"
                    self.note_manager.save_note(
                        note_id,
                        doc['content'],
                        {**doc['metadata'], 'source': 'document', 'doc_id': doc_id}
                    )
            
            logger.info(f"Processed {len(doc_ids)} documents from {directory}")
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to process directory {directory}: {e}")
            raise
    
    def search(self, query: str, 
               top_k: Optional[int] = None,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search across documents and notes.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        try:
            # Search documents
            doc_results = self.document_retriever.search_documents(
                query, top_k, metadata_filter
            )
            
            # Search notes
            note_results = self.note_manager.search_notes(query)
            
            # Combine and sort results
            results = []
            for doc in doc_results:
                results.append({
                    'type': 'document',
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': doc.get('score', 0.0)
                })
            
            for note in note_results:
                results.append({
                    'type': 'note',
                    'id': note['id'],
                    'content': note['content'],
                    'metadata': note['metadata'],
                    'score': 1.0  # Notes don't have scores
                })
            
            # Sort by score if available
            results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            # Limit results
            if top_k:
                results = results[:top_k]
            
            return results
        except Exception as e:
            logger.error(f"Failed to perform search: {e}")
            return []
    
    def create_note(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new note.
        
        Args:
            content: Note content
            metadata: Optional metadata for the note
            
        Returns:
            Note ID
            
        Raises:
            ValueError: If content is invalid
        """
        try:
            note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.note_manager.save_note(note_id, content, metadata)
            logger.info(f"Created note {note_id}")
            return note_id
        except Exception as e:
            logger.error(f"Failed to create note: {e}")
            raise
    
    def update_note(self, note_id: str, 
                   content: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update an existing note.
        
        Args:
            note_id: Note ID
            content: New note content
            metadata: New metadata
            
        Returns:
            Updated note data
            
        Raises:
            FileNotFoundError: If the note doesn't exist
        """
        try:
            note_data = self.note_manager.update_note(note_id, content, metadata)
            logger.info(f"Updated note {note_id}")
            return note_data
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {e}")
            raise
    
    def delete_note(self, note_id: str) -> bool:
        """Delete a note.
        
        Args:
            note_id: Note ID
            
        Returns:
            True if the note was deleted, False otherwise
        """
        try:
            success = self.note_manager.delete_note(note_id)
            if success:
                logger.info(f"Deleted note {note_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete note {note_id}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        try:
            return {
                'documents': {
                    'count': len(self.document_retriever.get_collection_stats()),
                    'stats': self.document_retriever.get_collection_stats()
                },
                'notes': {
                    'count': len(self.note_manager.list_notes()),
                    'stats': {
                        'total_size': sum(
                            len(note['content']) 
                            for note in self.note_manager.list_notes()
                        )
                    }
                },
                'cache': self.state_manager.get_cache_stats(),
                'logging': self.log_manager.get_log_stats()
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def cleanup(self, days: int = 30) -> Dict[str, int]:
        """Clean up old data.
        
        Args:
            days: Number of days to keep data
            
        Returns:
            Dictionary containing cleanup statistics
        """
        try:
            stats = {
                'logs_cleared': self.log_manager.clear_logs(days),
                'cache_cleaned': self.state_manager.cleanup_cache()
            }
            logger.info(f"Cleaned up system data: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to cleanup system: {e}")
            return {} 