"""
Unit tests for vector store functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.storage.vector import VectorStore

class TestVectorStore:
    """Test suite for vector store functionality."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store for testing."""
        return VectorStore()
    
    @pytest.mark.asyncio
    async def test_vector_store_operations(self, vector_store):
        """Test vector store operations."""
        vector_data = {
            "id": "test_vector",
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"key": "value"}
        }
        
        await vector_store.store(vector_data)
        
        retrieved_vector = await vector_store.retrieve("test_vector")
        assert retrieved_vector is not None
        assert retrieved_vector["id"] == "test_vector"
        assert retrieved_vector["vector"] == [0.1, 0.2, 0.3]
        
        search_results = await vector_store.search([0.1, 0.2, 0.3], limit=1)
        assert len(search_results) > 0
        assert search_results[0]["id"] == "test_vector"
    
    @pytest.mark.asyncio
    async def test_vector_store_error_handling(self, vector_store):
        """Test vector store error handling."""
        with pytest.raises(ValueError):
            await vector_store.store(None)
        
        with pytest.raises(KeyError):
            await vector_store.retrieve("non_existent_vector")
    
    @pytest.mark.asyncio
    async def test_vector_store_metrics(self, vector_store):
        """Test vector store metrics collection."""
        vector_metrics = await vector_store.collect_metrics()
        assert vector_metrics is not None
        assert isinstance(vector_metrics, dict)
        assert "total_vectors" in vector_metrics
        assert "storage_size" in vector_metrics
    
    @pytest.mark.asyncio
    async def test_vector_store_optimization(self, vector_store):
        """Test vector store optimization operations."""
        vector_params = {
            "index_type": "hnsw",
            "dimension": 128,
            "max_elements": 1000
        }
        
        optimized_vector_store = await vector_store.optimize(vector_params)
        assert optimized_vector_store is not None
        assert optimized_vector_store.is_optimized 