"""
Core engine for the LangGraph application.
This module provides the main workflow and orchestration functionality.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
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
from pydantic import BaseModel, Field
from transformers import pipeline
from models.model_manager import ModelManager
from tools.sentiment_analyzer import SentimentAnalyzer
from tools.memory_manager import MemoryManager
import json

logger = logging.getLogger(__name__)

class Document(BaseModel):
    """Document model."""
    id: str
    content: str
    type: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class Note(BaseModel):
    """Note model."""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

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

class EngineService(BaseService[Union[Document, Note]]):
    """Service for core engine capabilities including response generation and document processing."""
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        model_manager: Optional[ModelManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        sentiment_analyzer: Optional[Any] = None
    ):
        """
        Initialize the engine service.
        
        Args:
            storage_dir: Optional directory for storing data
            model_manager: Optional model manager instance
            memory_manager: Optional memory manager instance
            sentiment_analyzer: Optional sentiment analyzer instance
        """
        super().__init__()
        self.storage_dir = Path(storage_dir or os.path.join(os.getcwd(), "storage", "engine"))
        self.model_manager = model_manager or ModelManager()
        self.memory_manager = memory_manager or MemoryManager()
        self.sentiment_analyzer = sentiment_analyzer or self._initialize_sentiment_analyzer()
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.document_retriever = DocumentRetriever()
        self.note_manager = NoteStoreManager()
        self.state_manager = StateManager()
        self.vector_store = ChromaVectorStore()
        
        self._ensure_storage()
        self._load_data()
    
    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "documents").mkdir(exist_ok=True)
        (self.storage_dir / "notes").mkdir(exist_ok=True)
    
    def _load_data(self) -> None:
        """Load data from storage."""
        try:
            # Load documents
            docs_dir = self.storage_dir / "documents"
            for doc_file in docs_dir.glob("*.json"):
                try:
                    with open(doc_file, "r") as f:
                        data = json.load(f)
                        document = Document(**data)
                        self.document_retriever.add_document(document)
                except Exception as e:
                    self.logger.error(f"Failed to load document from {doc_file}: {e}")
            
            # Load notes
            notes_dir = self.storage_dir / "notes"
            for note_file in notes_dir.glob("*.json"):
                try:
                    with open(note_file, "r") as f:
                        data = json.load(f)
                        note = Note(**data)
                        self.note_manager.add_note(note)
                except Exception as e:
                    self.logger.error(f"Failed to load note from {note_file}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
    
    def _initialize_sentiment_analyzer(self) -> Any:
        """Initialize sentiment analyzer."""
        try:
            return pipeline("sentiment-analysis")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {e}")
            try:
                return pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except Exception as e2:
                self.logger.error(f"Failed to initialize fallback sentiment analyzer: {e2}")
                raise
    
    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        style: Optional[str] = None
    ) -> str:
        """
        Generate a response.
        
        Args:
            query: User query
            context: Context for response generation
            style: Optional response style
            
        Returns:
            str: Generated response
        """
        try:
            # Analyze sentiment
            sentiment = self.sentiment_analyzer(query)[0]
            sentiment_label = sentiment["label"]
            sentiment_score = sentiment["score"]
            
            # Get response style
            if not style and hasattr(self.model_manager, "classify_style"):
                style = self.model_manager.classify_style(query)
            
            # Create prompt
            prompt = self._create_prompt(query, context, sentiment_label, style)
            
            # Generate response
            response = await self.model_manager.generate(prompt)
            
            # Record in history
            self.record_history(
                "generate_response",
                details={
                    "query": query,
                    "sentiment": sentiment_label,
                    "style": style
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            raise
    
    async def stream_response(
        self,
        query: str,
        context: Dict[str, Any],
        style: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response.
        
        Args:
            query: User query
            context: Context for response generation
            style: Optional response style
            
        Returns:
            AsyncGenerator[str, None]: Response stream
        """
        try:
            # Analyze sentiment
            sentiment = self.sentiment_analyzer(query)[0]
            sentiment_label = sentiment["label"]
            
            # Get response style
            if not style and hasattr(self.model_manager, "classify_style"):
                style = self.model_manager.classify_style(query)
            
            # Create prompt
            prompt = self._create_prompt(query, context, sentiment_label, style)
            
            # Stream response
            async for token in self.model_manager.stream(prompt):
                yield token
            
            # Record in history
            self.record_history(
                "stream_response",
                details={
                    "query": query,
                    "sentiment": sentiment_label,
                    "style": style
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to stream response: {e}")
            raise
    
    def process_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Process a document.
        
        Args:
            file_path: Path to document file
            metadata: Optional metadata
            
        Returns:
            Document: Processed document
        """
        try:
            # Process document
            doc_data = self.document_processor.process_file(file_path)
            
            # Create document
            document = Document(
                id=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content=doc_data["content"],
                type=doc_data["type"],
                metadata={**(metadata or {}), **doc_data.get("metadata", {})}
            )
            
            # Store document
            self.document_retriever.add_document(document)
            
            # Save to storage
            self._save_document(document)
            
            # Record in history
            self.record_history(
                "process_document",
                details={
                    "file_path": str(file_path),
                    "document_id": document.id,
                    "document_type": document.type
                }
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process document: {e}")
            raise
    
    def create_note(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Note:
        """
        Create a note.
        
        Args:
            content: Note content
            metadata: Optional metadata
            
        Returns:
            Note: Created note
        """
        try:
            # Create note
            note = Note(
                id=f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content=content,
                metadata=metadata
            )
            
            # Store note
            self.note_manager.add_note(note)
            
            # Save to storage
            self._save_note(note)
            
            # Record in history
            self.record_history(
                "create_note",
                details={
                    "note_id": note.id,
                    "content_length": len(content)
                }
            )
            
            return note
            
        except Exception as e:
            self.logger.error(f"Failed to create note: {e}")
            raise
    
    def search(
        self,
        query: str,
        doc_types: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents and notes.
        
        Args:
            query: Search query
            doc_types: Optional list of document types to search
            metadata_filter: Optional metadata filter
            top_k: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        try:
            # Search documents
            doc_results = self.document_retriever.search(
                query,
                doc_types=doc_types,
                metadata_filter=metadata_filter,
                top_k=top_k
            )
            
            # Search notes
            note_results = self.note_manager.search(
                query,
                metadata_filter=metadata_filter,
                top_k=top_k
            )
            
            # Combine results
            results = []
            for doc in doc_results:
                results.append({
                    "type": "document",
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "timestamp": doc.timestamp
                })
            
            for note in note_results:
                results.append({
                    "type": "note",
                    "id": note.id,
                    "content": note.content,
                    "metadata": note.metadata,
                    "timestamp": note.timestamp
                })
            
            # Sort by timestamp
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Limit results
            results = results[:top_k]
            
            # Record in history
            self.record_history(
                "search",
                details={
                    "query": query,
                    "doc_types": doc_types,
                    "result_count": len(results)
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search: {e}")
            raise
    
    def _create_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        sentiment: str,
        style: Optional[str] = None
    ) -> str:
        """
        Create a prompt for response generation.
        
        Args:
            query: User query
            context: Context for response generation
            sentiment: Query sentiment
            style: Optional response style
            
        Returns:
            str: Generated prompt
        """
        # Create base prompt
        prompt = f"Query: {query}\n\nContext:\n"
        
        # Add context sections
        for key, value in context.items():
            prompt += f"\n{key}:\n{value}"
        
        # Add sentiment and style guidance
        prompt += f"\n\nSentiment: {sentiment}"
        if style:
            prompt += f"\nStyle: {style}"
        
        prompt += "\n\nResponse:"
        
        return prompt
    
    def _save_document(self, document: Document) -> None:
        """
        Save document to storage.
        
        Args:
            document: Document to save
        """
        try:
            file_path = self.storage_dir / "documents" / f"{document.id}.json"
            with open(file_path, "w") as f:
                json.dump(document.dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save document: {e}")
            raise
    
    def _save_note(self, note: Note) -> None:
        """
        Save note to storage.
        
        Args:
            note: Note to save
        """
        try:
            file_path = self.storage_dir / "notes" / f"{note.id}.json"
            with open(file_path, "w") as f:
                json.dump(note.dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save note: {e}")
            raise
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Dict[str, Any]: Engine statistics
        """
        return {
            "total_documents": len(self.document_retriever.documents),
            "total_notes": len(self.note_manager.notes),
            "document_types": {
                doc_type: len([d for d in self.document_retriever.documents.values() if d.type == doc_type])
                for doc_type in set(d.type for d in self.document_retriever.documents.values())
            },
            "latest_document": next(iter(self.document_retriever.documents.values())) if self.document_retriever.documents else None,
            "latest_note": next(iter(self.note_manager.notes.values())) if self.note_manager.notes else None
        }
    
    def reset(self) -> None:
        """Reset engine service."""
        super().reset()
        self.document_retriever.reset()
        self.note_manager.reset()
        self.state_manager.reset()
        self.vector_store.reset()

__all__ = ['EngineService', 'Document', 'Note'] 