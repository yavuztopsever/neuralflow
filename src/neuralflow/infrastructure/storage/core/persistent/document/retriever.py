"""
Document retrieval utilities for NeuralFlow.
This module provides functionality for retrieving and searching documents.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from ...vector.base import BaseVectorStore, VectorStoreConfig
from ...vector.providers.chroma import ChromaVectorStore
import json

logger = logging.getLogger(__name__)

class Document:
    """Document model class."""
    
    def __init__(self,
                 doc_id: str,
                 content: str,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a document.
        
        Args:
            doc_id: Document ID
            content: Document content
            metadata: Optional document metadata
        """
        self.id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.created = datetime.now().isoformat()
        self.modified = self.created

class DocumentRetriever:
    """Handles document storage, retrieval and search operations."""
    
    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        """Initialize the document retriever.
        
        Args:
            storage_dir: Directory for document storage
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
    def _initialize_vector_store(self) -> BaseVectorStore:
        """Initialize the vector store.
        
        Returns:
            Initialized vector store
        """
        try:
            config = VectorStoreConfig(
                store_id="documents",
                store_type="chroma",
                dimension=1536,  # Default for text embeddings
                metric="cosine",
                index_type="hnsw",
                embedder_model="all-MiniLM-L6-v2",
                persist_directory=str(self.storage_dir / "vectors") if self.storage_dir else None
            )
            store = ChromaVectorStore(config)
            store.initialize()
            return store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
            
    def add_document(self, document: Document) -> str:
        """Add a document to storage.
        
        Args:
            document: Document to add
            
        Returns:
            Document ID
            
        Raises:
            ValueError: If document is invalid
        """
        if not document.id or not document.content:
            raise ValueError("Document ID and content must be provided")
            
        try:
            # Save document to file system
            if self.storage_dir:
                doc_file = self.storage_dir / f"{document.id}.json"
                doc_data = {
                    'id': document.id,
                    'content': document.content,
                    'metadata': document.metadata,
                    'created': document.created,
                    'modified': document.modified
                }
                with open(doc_file, 'w') as f:
                    json.dump(doc_data, f, indent=2)
            
            # Add to vector store
            self.vector_store.add_texts(
                texts=[document.content],
                metadata=[document.metadata],
                ids=[document.id]
            )
            
            logger.info(f"Added document {document.id}")
            return document.id
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            raise
            
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        try:
            # Try file system first
            if self.storage_dir:
                doc_file = self.storage_dir / f"{doc_id}.json"
                if doc_file.exists():
                    with open(doc_file, 'r') as f:
                        doc_data = json.load(f)
                        return Document(
                            doc_id=doc_data['id'],
                            content=doc_data['content'],
                            metadata=doc_data['metadata']
                        )
            
            # Try vector store
            vector = self.vector_store.get_vector(doc_id)
            metadata = self.vector_store.get_metadata(doc_id)
            if vector and metadata:
                return Document(
                    doc_id=doc_id,
                    content=metadata.get('text', ''),
                    metadata=metadata
                )
            
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
            
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = True
            
            # Delete from file system
            if self.storage_dir:
                doc_file = self.storage_dir / f"{doc_id}.json"
                if doc_file.exists():
                    doc_file.unlink()
            
            # Delete from vector store
            if not self.vector_store.delete_vector(doc_id):
                success = False
            
            if success:
                logger.info(f"Deleted document {doc_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
            
    def search_documents(self,
                        query: str,
                        top_k: Optional[int] = None,
                        metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of matching documents
        """
        try:
            results = self.vector_store.search_texts(
                query=query,
                k=top_k or 5,
                filter_metadata=metadata_filter
            )
            
            documents = []
            for result in results:
                doc = Document(
                    doc_id=result['id'],
                    content=result['metadata'].get('text', ''),
                    metadata=result['metadata']
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
            
    def list_documents(self, pattern: str = "*.json") -> List[Document]:
        """List all documents.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of documents
        """
        try:
            documents = []
            
            # List from file system
            if self.storage_dir:
                for doc_file in self.storage_dir.glob(pattern):
                    try:
                        with open(doc_file, 'r') as f:
                            doc_data = json.load(f)
                            document = Document(
                                doc_id=doc_data['id'],
                                content=doc_data['content'],
                                metadata=doc_data['metadata']
                            )
                            documents.append(document)
                    except Exception as e:
                        logger.warning(f"Failed to load document from {doc_file}: {e}")
            
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        try:
            stats = {
                'vector_store': self.vector_store.get_store_info(),
                'documents': {
                    'total': 0,
                    'size': 0
                }
            }
            
            if self.storage_dir:
                doc_files = list(self.storage_dir.glob("*.json"))
                stats['documents']['total'] = len(doc_files)
                stats['documents']['size'] = sum(f.stat().st_size for f in doc_files)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {} 