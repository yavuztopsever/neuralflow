"""
Unit tests for vector store functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.storage.vector import VectorStore
from src.storage.vector.index import VectorIndex
from src.storage.vector.search import VectorSearch

class TestVectorStore:
    """Test suite for vector store functionality."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store for testing."""
        return VectorStore()
    
    @pytest.fixture
    def vector_index(self):
        """Create a vector index for testing."""
        return VectorIndex()
    
    @pytest.fixture
    def vector_search(self):
        """Create a vector search for testing."""
        return VectorSearch()
    
    @pytest.mark.asyncio
    async def test_vector_store_operations(self, vector_store):
        """Test vector store operations."""
        # Test storing vectors
        vector_data = {
            "id": "test_vector",
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"key": "value"}
        }
        
        await vector_store.store(vector_data)
        
        # Test retrieving vectors
        retrieved_vector = await vector_store.retrieve("test_vector")
        assert retrieved_vector is not None
        assert retrieved_vector["id"] == "test_vector"
        assert retrieved_vector["vector"] == [0.1, 0.2, 0.3]
        
        # Test updating vectors
        updated_data = {
            "id": "test_vector",
            "vector": [0.4, 0.5, 0.6],
            "metadata": {"key": "updated_value"}
        }
        
        await vector_store.update(updated_data)
        updated_vector = await vector_store.retrieve("test_vector")
        assert updated_vector["vector"] == [0.4, 0.5, 0.6]
        assert updated_vector["metadata"]["key"] == "updated_value"
        
        # Test deleting vectors
        await vector_store.delete("test_vector")
        with pytest.raises(KeyError):
            await vector_store.retrieve("test_vector")
    
    @pytest.mark.asyncio
    async def test_vector_index_operations(self, vector_index):
        """Test vector index operations."""
        # Test index creation
        vectors = [
            {"id": "vec1", "vector": [0.1, 0.2, 0.3]},
            {"id": "vec2", "vector": [0.4, 0.5, 0.6]},
            {"id": "vec3", "vector": [0.7, 0.8, 0.9]}
        ]
        
        await vector_index.build(vectors)
        assert vector_index.size == len(vectors)
        
        # Test index search
        query_vector = [0.1, 0.2, 0.3]
        results = await vector_index.search(query_vector, k=2)
        assert len(results) == 2
        assert all("id" in r and "distance" in r for r in results)
        
        # Test index update
        new_vector = {"id": "vec4", "vector": [0.2, 0.3, 0.4]}
        await vector_index.add(new_vector)
        assert vector_index.size == len(vectors) + 1
        
        # Test index deletion
        await vector_index.delete("vec1")
        assert vector_index.size == len(vectors)
    
    @pytest.mark.asyncio
    async def test_vector_search_operations(self, vector_search):
        """Test vector search operations."""
        # Test similarity search
        query_vector = [0.1, 0.2, 0.3]
        target_vectors = [
            {"id": "vec1", "vector": [0.1, 0.2, 0.3]},
            {"id": "vec2", "vector": [0.4, 0.5, 0.6]},
            {"id": "vec3", "vector": [0.7, 0.8, 0.9]}
        ]
        
        results = await vector_search.similarity_search(query_vector, target_vectors, k=2)
        assert len(results) == 2
        assert all("id" in r and "similarity" in r for r in results)
        
        # Test range search
        range_results = await vector_search.range_search(query_vector, target_vectors, radius=0.5)
        assert isinstance(range_results, list)
        assert all("id" in r and "distance" in r for r in range_results)
        
        # Test batch search
        query_vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        batch_results = await vector_search.batch_search(query_vectors, target_vectors)
        assert len(batch_results) == len(query_vectors)
        assert all(len(r) > 0 for r in batch_results)
    
    @pytest.mark.asyncio
    async def test_vector_store_error_handling(self, vector_store, vector_index, vector_search):
        """Test vector store error handling."""
        # Test vector store errors
        with pytest.raises(ValueError):
            await vector_store.store(None)
        
        # Test vector index errors
        with pytest.raises(ValueError):
            await vector_index.build(None)
        
        # Test vector search errors
        with pytest.raises(ValueError):
            await vector_search.similarity_search(None, [])
    
    @pytest.mark.asyncio
    async def test_vector_store_metrics(self, vector_store, vector_index, vector_search):
        """Test vector store metrics collection."""
        # Test vector store metrics
        store_metrics = await vector_store.collect_metrics()
        assert store_metrics is not None
        assert isinstance(store_metrics, dict)
        assert "total_vectors" in store_metrics
        assert "storage_size" in store_metrics
        
        # Test vector index metrics
        index_metrics = await vector_index.collect_metrics()
        assert index_metrics is not None
        assert isinstance(index_metrics, dict)
        assert "index_size" in index_metrics
        assert "average_search_time" in index_metrics
        
        # Test vector search metrics
        search_metrics = await vector_search.collect_metrics()
        assert search_metrics is not None
        assert isinstance(search_metrics, dict)
        assert "total_searches" in search_metrics
        assert "average_similarity" in search_metrics
    
    @pytest.mark.asyncio
    async def test_vector_store_optimization(self, vector_store, vector_index, vector_search):
        """Test vector store optimization operations."""
        # Test vector store optimization
        store_params = {
            "index_type": "hnsw",
            "dimension": 128,
            "max_elements": 1000
        }
        
        optimized_store = await vector_store.optimize(store_params)
        assert optimized_store is not None
        assert optimized_store.is_optimized
        
        # Test vector index optimization
        index_params = {
            "index_type": "ivf",
            "nlist": 100,
            "nprobe": 10
        }
        
        optimized_index = await vector_index.optimize(index_params)
        assert optimized_index is not None
        assert optimized_index.is_optimized
        
        # Test vector search optimization
        search_params = {
            "batch_size": 32,
            "cache_size": 1000,
            "timeout": 5
        }
        
        optimized_search = await vector_search.optimize(search_params)
        assert optimized_search is not None
        assert optimized_search.is_optimized 