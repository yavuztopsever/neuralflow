"""
Base vector storage interfaces for NeuralFlow.
"""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStoreConfig:
    """Configuration for a vector store."""
    
    def __init__(self,
                 store_id: str = '',
                 store_type: str = '',
                 dimension: int = 1536,
                 metric: str = 'cosine',
                 index_type: str = 'flat',
                 embedder_model: str = 'all-MiniLM-L6-v2',
                 **kwargs):
        """Initialize vector store configuration.
        
        Args:
            store_id: Unique identifier for the store
            store_type: Store type
            dimension: Vector dimensions
            metric: Distance metric (cosine, euclidean, etc.)
            index_type: Index type (flat, ivf, etc.)
            embedder_model: Name of the sentence transformer model
            **kwargs: Additional configuration parameters
        """
        self.id = store_id
        self.type = store_type
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.embedder_model = embedder_model
        self.parameters = kwargs
        self.metadata = {}

class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the vector store.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self.embedding_model = SentenceTransformer(config.embedder_model)
        
    def _generate_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(text)
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up vector store resources."""
        pass
    
    @abstractmethod
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
        pass
    
    def add_texts(self,
                 texts: List[str],
                 metadata: Optional[List[Dict[str, Any]]] = None,
                 ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to the store.
        
        Args:
            texts: List of texts to add
            metadata: Optional metadata for each text
            ids: Optional text IDs
            
        Returns:
            List of vector IDs
        """
        try:
            embeddings = self._generate_embedding(texts)
            return self.add_vectors(embeddings, metadata, ids)
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            raise
    
    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Get a vector by ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Vector or None if not found
        """
        pass
    
    @abstractmethod
    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Metadata dictionary or None if not found
        """
        pass
    
    @abstractmethod
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
        pass
    
    def search_texts(self,
                    query: str,
                    k: int = 10,
                    filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar texts.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with scores and metadata
        """
        try:
            query_vector = self._generate_embedding(query)
            return self.search(query_vector, k, filter_metadata)
        except Exception as e:
            logger.error(f"Error searching texts: {e}")
            raise
    
    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            True if vector was deleted, False otherwise
        """
        pass
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get store information.
        
        Returns:
            Dictionary containing store information
        """
        return {
            'id': self.config.id,
            'type': self.config.type,
            'dimension': self.config.dimension,
            'metric': self.config.metric,
            'index_type': self.config.index_type,
            'embedder_model': self.config.embedder_model,
            'created': self.created,
            'modified': self.modified,
            'parameters': self.config.parameters,
            'metadata': self.config.metadata
        }
    
    @abstractmethod
    def save_to_disk(self, path: Union[str, Path]) -> None:
        """Save vector store to disk.
        
        Args:
            path: Path to save to
        """
        pass
    
    @abstractmethod
    def load_from_disk(self, path: Union[str, Path]) -> None:
        """Load vector store from disk.
        
        Args:
            path: Path to load from
        """
        pass

__all__ = ['BaseVectorStore', 'VectorStoreConfig']
