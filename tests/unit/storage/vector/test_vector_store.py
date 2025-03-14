"""
Unit tests for vector store functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.storage.vector.base import BaseVectorStore, VectorStoreConfig

class MockVectorStore(BaseVectorStore):
    """Mock vector store implementation for testing."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        self._initialized = True
    
    def cleanup(self) -> None:
        self._vectors.clear()
        self._metadata.clear()
        self._initialized = False
    
    def add_vectors(self,
                   vectors: List[List[float]],
                   metadata: List[Dict[str, Any]] = None,
                   ids: List[str] = None) -> List[str]:
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
            
        if ids is None:
            ids = [f"v{i}" for i in range(len(vectors))]
        if metadata is None:
            metadata = [{} for _ in vectors]
            
        for vid, vec, meta in zip(ids, vectors, metadata):
            self._vectors[vid] = vec
            self._metadata[vid] = meta
            
        return ids
    
    def get_vector(self, vector_id: str) -> List[float]:
        return self._vectors.get(vector_id)
    
    def get_metadata(self, vector_id: str) -> Dict[str, Any]:
        return self._metadata.get(vector_id)
    
    def search(self,
              query_vector: List[float],
              k: int = 10,
              filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self._vectors:
            return []
            
        # Convert to numpy for cosine similarity
        query = np.array(query_vector)
        vectors = {
            vid: np.array(vec)
            for vid, vec in self._vectors.items()
        }
        
        # Calculate similarities
        similarities = {
            vid: np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            for vid, vec in vectors.items()
        }
        
        # Sort by similarity
        results = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply metadata filter
        if filter_metadata:
            results = [
                (vid, score) for vid, score in results
                if all(
                    self._metadata[vid].get(k) == v
                    for k, v in filter_metadata.items()
                )
            ]
        
        # Format results
        return [
            {
                'id': vid,
                'score': score,
                'vector': self._vectors[vid],
                'metadata': self._metadata[vid]
            }
            for vid, score in results[:k]
        ]
    
    def delete_vector(self, vector_id: str) -> bool:
        if vector_id not in self._vectors:
            return False
            
        del self._vectors[vector_id]
        if vector_id in self._metadata:
            del self._metadata[vector_id]
            
        return True

@pytest.fixture
def vector_store():
    """Create a mock vector store."""
    config = VectorStoreConfig(
        name="test_store",
        store_type="mock",
        dimensions=4,
        embedder_model="mock"
    )
    store = MockVectorStore(config)
    store.initialize()
    return store

@pytest.fixture
def sample_vectors():
    """Create sample vectors."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]

@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return [
        {'type': 'a', 'value': 1},
        {'type': 'b', 'value': 2},
        {'type': 'a', 'value': 3}
    ]

class TestVectorStore:
    """Test suite for vector store functionality."""
    
    def test_initialization(self, vector_store):
        """Test vector store initialization."""
        assert vector_store.config.name == "test_store"
        assert vector_store.config.dimensions == 4
        assert vector_store._initialized
    
    def test_add_vectors(self, vector_store, sample_vectors, sample_metadata):
        """Test adding vectors."""
        ids = vector_store.add_vectors(
            vectors=sample_vectors,
            metadata=sample_metadata
        )
        assert len(ids) == len(sample_vectors)
        
        for i, vid in enumerate(ids):
            stored_vector = vector_store.get_vector(vid)
            stored_metadata = vector_store.get_metadata(vid)
            assert stored_vector == sample_vectors[i]
            assert stored_metadata == sample_metadata[i]
    
    def test_add_texts(self, vector_store):
        """Test adding texts."""
        texts = ["hello world", "test text", "another text"]
        metadata = [{'type': 'text'} for _ in texts]
        
        ids = vector_store.add_texts(
            texts=texts,
            metadata=metadata
        )
        assert len(ids) == len(texts)
        
        for vid in ids:
            vector = vector_store.get_vector(vid)
            assert len(vector) == vector_store.config.dimensions
    
    def test_search(self, vector_store, sample_vectors, sample_metadata):
        """Test vector search."""
        # Add vectors
        ids = vector_store.add_vectors(
            vectors=sample_vectors,
            metadata=sample_metadata
        )
        
        # Search similar to first vector
        results = vector_store.search(
            query_vector=sample_vectors[0],
            k=2
        )
        assert len(results) == 2
        assert results[0]['id'] == ids[0]  # Most similar should be itself
        
        # Test with metadata filter
        filtered_results = vector_store.search(
            query_vector=sample_vectors[0],
            k=2,
            filter_metadata={'type': 'a'}
        )
        assert len(filtered_results) == 2
        for result in filtered_results:
            assert result['metadata']['type'] == 'a'
    
    def test_search_texts(self, vector_store):
        """Test text search."""
        texts = [
            "hello world",
            "test document",
            "another test"
        ]
        vector_store.add_texts(texts=texts)
        
        results = vector_store.search_texts(
            query="hello",
            k=1
        )
        assert len(results) == 1
    
    def test_delete_vector(self, vector_store, sample_vectors):
        """Test vector deletion."""
        ids = vector_store.add_vectors(vectors=sample_vectors)
        
        # Delete first vector
        success = vector_store.delete_vector(ids[0])
        assert success
        assert vector_store.get_vector(ids[0]) is None
        
        # Try to delete non-existent vector
        success = vector_store.delete_vector("nonexistent")
        assert not success
    
    def test_store_info(self, vector_store):
        """Test getting store information."""
        info = vector_store.get_store_info()
        assert info['name'] == "test_store"
        assert info['type'] == "mock"
        assert info['dimensions'] == 4
        assert 'created' in info
        assert 'modified' in info 