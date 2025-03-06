"""
Document retrieval utilities for the LangGraph application.
This module provides functionality for retrieving and searching documents.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.config import Config
from storage.vector.providers.chroma import ChromaVectorStore
from storage.document.processor import DocumentProcessor

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Handles document retrieval and search operations."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the document retriever.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.text_processor = TextProcessor()
        self.vector_store = ChromaVectorStore(config)
        self.document_processor = DocumentProcessor(config)
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize the document storage directory."""
        try:
            self.storage_dir = Path(self.config.DOCUMENTS_DIR)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized document storage at {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize document storage: {e}")
            raise
    
    def add_document(self, file_path: Union[str, Path], 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the retrieval system.
        
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
            # Process the document
            doc = self.document_processor.load_document(file_path)
            if metadata:
                doc['metadata'].update(metadata)
            
            # Add to vector store
            self.vector_store.add_documents([doc])
            
            logger.info(f"Added document {doc['id']} to retrieval system")
            return doc['id']
        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {e}")
            raise
    
    def add_directory(self, directory: Union[str, Path],
                     file_pattern: str = "*.*",
                     recursive: bool = True,
                     metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add all documents in a directory to the retrieval system.
        
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
            documents = self.document_processor.process_directory(
                directory, file_pattern, recursive
            )
            
            if metadata:
                for doc in documents:
                    doc['metadata'].update(metadata)
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            doc_ids = [doc['id'] for doc in documents]
            logger.info(f"Added {len(doc_ids)} documents to retrieval system")
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to add directory {directory}: {e}")
            raise
    
    def search_documents(self, query: str, 
                        top_k: Optional[int] = None,
                        metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of matching documents with scores
        """
        try:
            # Get results from vector store
            results = self.vector_store.search_documents(query, n_results=top_k or self.config.SEARCH_TOP_K)
            
            # Apply metadata filter if provided
            if metadata_filter:
                results = self._filter_by_metadata(results, metadata_filter)
            
            return results
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            results = self.vector_store.search_documents(
                query="",  # Empty query to get all documents
                n_results=1,
                where={"id": doc_id}
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the retrieval system.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if the document was deleted, False otherwise
        """
        try:
            # Delete from vector store
            self.vector_store.delete_documents([doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def update_document(self, doc_id: str, 
                       content: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a document in the retrieval system.
        
        Args:
            doc_id: Document ID
            content: New document content
            metadata: New metadata
            
        Returns:
            True if the document was updated, False otherwise
        """
        try:
            # Get existing document
            doc = self.get_document(doc_id)
            if not doc:
                return False
            
            # Update content if provided
            if content is not None:
                doc['content'] = content
            
            # Update metadata if provided
            if metadata is not None:
                doc['metadata'].update(metadata)
            
            # Update timestamp
            doc['modified'] = datetime.now().isoformat()
            
            # Delete old version and add updated version
            self.delete_document(doc_id)
            self.vector_store.add_documents([doc])
            
            logger.info(f"Updated document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            return self.vector_store.get_collection_statistics()
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def _filter_by_metadata(self, results: List[Dict[str, Any]], 
                          metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results by metadata criteria.
        
        Args:
            results: List of search results
            metadata_filter: Metadata filter criteria
            
        Returns:
            Filtered list of results
        """
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            matches = True
            
            for key, value in metadata_filter.items():
                if key not in metadata or metadata[key] != value:
                    matches = False
                    break
            
            if matches:
                filtered_results.append(result)
                
        return filtered_results 