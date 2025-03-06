"""
Unit tests for storage system functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.storage.vector import VectorStore
from src.storage.cache import Cache
from src.storage.database import Database

class TestStorage:
    """Test suite for storage system functionality."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store for testing."""
        return VectorStore()
    
    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        return Cache()
    
    @pytest.fixture
    def database(self):
        """Create a database for testing."""
        return Database()
    
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
        
        # Test similarity search
        search_results = await vector_store.search([0.1, 0.2, 0.3], limit=1)
        assert len(search_results) > 0
        assert search_results[0]["id"] == "test_vector"
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, cache):
        """Test cache operations."""
        # Test storing in cache
        cache_data = {
            "key": "test_key",
            "value": "test_value",
            "ttl": 3600
        }
        
        await cache.store(cache_data)
        
        # Test retrieving from cache
        retrieved_value = await cache.retrieve("test_key")
        assert retrieved_value == "test_value"
        
        # Test cache expiration
        await asyncio.sleep(0.1)  # Simulate time passing
        expired_value = await cache.retrieve("test_key")
        assert expired_value is None
    
    @pytest.mark.asyncio
    async def test_database_operations(self, database):
        """Test database operations."""
        # Test storing data
        db_data = {
            "collection": "test_collection",
            "document": {
                "id": "test_doc",
                "data": "test_data"
            }
        }
        
        await database.store(db_data)
        
        # Test retrieving data
        retrieved_doc = await database.retrieve(
            "test_collection",
            "test_doc"
        )
        assert retrieved_doc is not None
        assert retrieved_doc["data"] == "test_data"
        
        # Test updating data
        update_data = {
            "collection": "test_collection",
            "document": {
                "id": "test_doc",
                "data": "updated_data"
            }
        }
        
        await database.update(update_data)
        updated_doc = await database.retrieve(
            "test_collection",
            "test_doc"
        )
        assert updated_doc["data"] == "updated_data"
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self, vector_store, cache, database):
        """Test storage error handling."""
        # Test vector store errors
        with pytest.raises(ValueError):
            await vector_store.store(None)
        
        # Test cache errors
        with pytest.raises(ValueError):
            await cache.store(None)
        
        # Test database errors
        with pytest.raises(ValueError):
            await database.store(None)
    
    @pytest.mark.asyncio
    async def test_storage_metrics(self, vector_store, cache, database):
        """Test storage metrics collection."""
        # Test vector store metrics
        vector_metrics = await vector_store.collect_metrics()
        assert vector_metrics is not None
        assert isinstance(vector_metrics, dict)
        assert "total_vectors" in vector_metrics
        assert "storage_size" in vector_metrics
        
        # Test cache metrics
        cache_metrics = await cache.collect_metrics()
        assert cache_metrics is not None
        assert isinstance(cache_metrics, dict)
        assert "hit_rate" in cache_metrics
        assert "miss_rate" in cache_metrics
        
        # Test database metrics
        db_metrics = await database.collect_metrics()
        assert db_metrics is not None
        assert isinstance(db_metrics, dict)
        assert "total_documents" in db_metrics
        assert "storage_size" in db_metrics
    
    @pytest.mark.asyncio
    async def test_storage_optimization(self, vector_store, cache, database):
        """Test storage optimization operations."""
        # Test vector store optimization
        vector_params = {
            "index_type": "hnsw",
            "dimension": 128,
            "max_elements": 1000
        }
        
        optimized_vector_store = await vector_store.optimize(vector_params)
        assert optimized_vector_store is not None
        assert optimized_vector_store.is_optimized
        
        # Test cache optimization
        cache_params = {
            "max_size": 1000,
            "eviction_policy": "lru",
            "ttl": 3600
        }
        
        optimized_cache = await cache.optimize(cache_params)
        assert optimized_cache is not None
        assert optimized_cache.is_optimized
        
        # Test database optimization
        db_params = {
            "index_fields": ["id", "data"],
            "max_connections": 10,
            "pool_size": 5
        }
        
        optimized_database = await database.optimize(db_params)
        assert optimized_database is not None
        assert optimized_database.is_optimized 