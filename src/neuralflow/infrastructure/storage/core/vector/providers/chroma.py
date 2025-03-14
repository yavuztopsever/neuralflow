"""
ChromaDB vector store provider for NeuralFlow.
This module provides vector storage functionality using ChromaDB.
"""

import os
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

from ..base import BaseVectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """Manages vector search operations using ChromaDB."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the vector store.
        
        Args:
            config: Vector store configuration
        """
        super().__init__(config)
        self._initialized = False
        self.client = None
        self.collection = None

    def initialize(self) -> None:
        """Initialize ChromaDB components."""
        try:
            self.client = Client(Settings(
                persist_directory=str(self.config.parameters.get('persist_directory', 'chroma_db')),
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_or_create_collection(
                name=self.config.id or "default",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedder_model
                ),
                metadata={"hnsw:space": self.config.metric}
            )
            self._initialized = True
            logger.info(f"Initialized ChromaDB store {self.config.id}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB store: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up ChromaDB resources."""
        try:
            if self.client:
                self.client = None
            if self.collection:
                self.collection = None
            self._initialized = False
            logger.info(f"Cleaned up ChromaDB store {self.config.id}")
        except Exception as e:
            logger.error(f"Error cleaning up ChromaDB store: {e}")
            raise

    def add_vectors(self,
                   vectors: List[List[float]],
                   metadata: Optional[List[Dict[str, Any]]] = None,
                   ids: Optional[List[str]] = None) -> List[str]:
        """Add vectors to the store.
        
        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector
            ids: Optional vector IDs
            
        Returns:
            List of vector IDs
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB store not initialized")
            
        try:
            if ids is None:
                ids = [f"v{i}" for i in range(len(vectors))]
            if metadata is None:
                metadata = [{} for _ in vectors]
                
            self.collection.add(
                embeddings=vectors,
                metadatas=metadata,
                ids=ids
            )
            return ids
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            raise

    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Get a vector by ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Vector or None if not found
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB store not initialized")
            
        try:
            result = self.collection.get(ids=[vector_id])
            if result and result['embeddings']:
                return result['embeddings'][0]
            return None
        except Exception as e:
            logger.error(f"Error getting vector {vector_id}: {e}")
            return None

    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Metadata dictionary or None if not found
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB store not initialized")
            
        try:
            result = self.collection.get(ids=[vector_id])
            if result and result['metadatas']:
                return result['metadatas'][0]
            return None
        except Exception as e:
            logger.error(f"Error getting metadata for {vector_id}: {e}")
            return None

    def search(self,
              query_vector: List[float],
              k: int = 10,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with scores and metadata
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB store not initialized")
            
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=filter_metadata
            )
            
            if not results['ids']:
                return []
                
            return [
                {
                    'id': id_,
                    'score': float(distance),
                    'vector': embedding,
                    'metadata': metadata
                }
                for id_, distance, embedding, metadata in zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['embeddings'][0],
                    results['metadatas'][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            True if vector was deleted, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB store not initialized")
            
        try:
            self.collection.delete(ids=[vector_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting vector {vector_id}: {e}")
            return False

    def save_to_disk(self, path: Union[str, Path]) -> None:
        """Save vector store to disk.
        
        Args:
            path: Path to save to
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB store not initialized")
            
        try:
            # ChromaDB automatically persists data if persist_directory is set
            logger.info(f"ChromaDB store {self.config.id} saved to {self.config.parameters.get('persist_directory')}")
        except Exception as e:
            logger.error(f"Error saving ChromaDB store: {e}")
            raise

    def load_from_disk(self, path: Union[str, Path]) -> None:
        """Load vector store from disk.
        
        Args:
            path: Path to load from
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB store not initialized")
            
        try:
            # ChromaDB automatically loads data from persist_directory
            logger.info(f"ChromaDB store {self.config.id} loaded from {self.config.parameters.get('persist_directory')}")
        except Exception as e:
            logger.error(f"Error loading ChromaDB store: {e}")
            raise 