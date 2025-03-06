"""
ChromaDB vector store provider for the LangGraph application.
This module provides vector storage functionality using ChromaDB.
"""

import os
import redis
import threading
import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
from config.config import Config
from utils.error.handlers import ErrorHandler

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Manages vector search operations using ChromaDB and Sentence Transformers."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the vector store.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self._initialize_components()

    def _initialize_components(self):
        """Initialize required components."""
        try:
            self.client = Client(Settings(
                persist_directory=str(self.config.VECTOR_DB_DIR),
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.EMBEDDER_MODEL
                ),
                metadata={"hnsw:space": "cosine"}
            )
            self.embedding_function = SentenceTransformer(self.config.EMBEDDER_MODEL)
        except Exception as e:
            logger.warning(f"Failed to initialize components: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        try:
            embeddings = [self._generate_embedding(doc['content']) for doc in documents]
            self.collection.add(
                embeddings=embeddings,
                documents=[doc['content'] for doc in documents],
                metadatas=[doc.get('metadata', {}) for doc in documents],
                ids=[doc.get('id', str(i)) for i, doc in enumerate(documents)]
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            query_embedding = self._generate_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def clear_collection(self):
        """Clear the vector store collection."""
        try:
            self.collection.delete()
            logger.info("Cleared vector store collection")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.EMBEDDER_MODEL
                ),
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def get_collection_statistics(self) -> Dict:
        """Get statistics about the vector store collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            return {
                'count': self.collection.count(),
                'name': self.collection.name,
                'metadata': self.collection.get()
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def _get_cached_embedding(self, texts):
        """Get embeddings from cache or create them if not present.
        
        Args:
            texts: Text or list of texts to embed
            
        Returns:
            List of embeddings
        """
        if not isinstance(texts, list):
            texts = [texts]
        embeddings = []
        cache_keys = [f"embedding:{text}" for text in texts]
        with self.cache_lock:
            for cache_key, text in zip(cache_keys, texts):
                embedding = self.memory_manager.get_cache(cache_key)
                if embedding is None:
                    embedding = self._generate_embedding([text])[0]
                    if not isinstance(embedding, list):
                        raise TypeError("Embedding function must return a list")
                    self.memory_manager.set_cache(cache_key, embedding)
                embeddings.append(embedding)
        return embeddings

    def _generate_embedding(self, text):
        """Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            return self.embedding_function.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def _create_mock_embedding(self, texts):
        """Create mock embeddings for testing purposes.
        
        Args:
            texts: Text or list of texts to create mock embeddings for
            
        Returns:
            List of mock embeddings
        """
        if isinstance(texts, str):
            return [self._mock_embedding(texts)]
        return [self._mock_embedding(text) for text in texts]

    def _mock_embedding(self, text):
        """Create a mock embedding for testing.
        
        Args:
            text: Text to create mock embedding for
            
        Returns:
            Mock embedding vector
        """
        try:
            # Try to determine vector dimensions from existing DB
            if hasattr(self.vector_db, '_collection'):
                if hasattr(self.vector_db._collection, 'dimension'):
                    dim = self.vector_db._collection.dimension
                    return [0.0] * dim
                elif hasattr(self.vector_db._collection, 'metadata') and self.vector_db._collection.metadata.get('dimension'):
                    dim = self.vector_db._collection.metadata.get('dimension')
                    return [0.0] * dim
            # Fixed dimensions for different embedding models
            if hasattr(Config, 'EMBEDDING_DIMENSION') and Config.EMBEDDING_DIMENSION:
                return [0.0] * Config.EMBEDDING_DIMENSION
            # Default to 1536 (OpenAI embedding dimension)
            return [0.0] * 1536
        except Exception as e:
            print(f"Error determining embedding dimension: {e}, using default 1536")
            return [0.0] * 1536
    
    def add_document(self, doc_id: str, text: str) -> str:
        """Add a document to the vector database and cache its embedding.
        
        Args:
            doc_id: Unique identifier for the document
            text: Text content of the document
        
        Returns:
            Confirmation message that the document has been indexed.
        
        Raises:
            ValueError: If the document ID or text is invalid
            RuntimeError: If there is an error adding the document to the database
        """
        if not doc_id or not text:
            raise ValueError("Document ID and text must be provided.")
        
        try:
            embedding = self._get_cached_embedding(text)[0]
            self.vector_db.add_texts([text], ids=[doc_id], embeddings=[embedding])
            return f"Document {doc_id} indexed in vector DB."
        except Exception as e:
            raise RuntimeError(f"Failed to add document {doc_id} to vector DB: {e}")
    
    def search_similar_documents(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Find similar documents using vector search.
        
        Args:
            query: The query text to search for similar documents
            top_k: The number of top similar documents to retrieve
        
        Returns:
            List of similar document contents
        
        Raises:
            ValueError: If the query is invalid
            RuntimeError: If there is an error performing the similarity search
        """
        if not query:
            raise ValueError("Query must be provided.")
        
        top_k = top_k or Config.VECTOR_SEARCH_TOP_K
        
        try:
            # Try multiple methods to handle different Chroma API versions
            try:
                # Try the newer method with query embedding
                query_embedding = self._get_cached_embedding(query)[0]
                results = self.vector_db.similarity_search_by_vector(query_embedding, k=top_k)
                return [result.page_content for result in results]
            except (AttributeError, TypeError):
                # Try the direct text search as fallback
                results = self.vector_db.similarity_search(query, k=top_k)
                return [result.page_content for result in results]
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}") 