"""
Unit tests for cache system functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.storage.cache import Cache
from src.storage.cache.memory import MemoryCache
from src.storage.cache.redis import RedisCache

class TestCache:
    """Test suite for cache system functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        return Cache()
    
    @pytest.fixture
    def memory_cache(self):
        """Create a memory cache for testing."""
        return MemoryCache()
    
    @pytest.fixture
    def redis_cache(self):
        """Create a Redis cache for testing."""
        return RedisCache()
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, cache):
        """Test cache operations."""
        # Test storing data
        cache_data = {
            "key": "test_key",
            "value": "test_value",
            "ttl": 3600
        }
        
        await cache.store(cache_data)
        
        # Test retrieving data
        retrieved_value = await cache.retrieve("test_key")
        assert retrieved_value == "test_value"
        
        # Test updating data
        updated_data = {
            "key": "test_key",
            "value": "updated_value",
            "ttl": 3600
        }
        
        await cache.update(updated_data)
        updated_value = await cache.retrieve("test_key")
        assert updated_value == "updated_value"
        
        # Test deleting data
        await cache.delete("test_key")
        with pytest.raises(KeyError):
            await cache.retrieve("test_key")
    
    @pytest.mark.asyncio
    async def test_memory_cache_operations(self, memory_cache):
        """Test memory cache operations."""
        # Test storing data
        cache_data = {
            "key": "test_key",
            "value": "test_value",
            "ttl": 3600
        }
        
        await memory_cache.store(cache_data)
        
        # Test retrieving data
        retrieved_value = await memory_cache.retrieve("test_key")
        assert retrieved_value == "test_value"
        
        # Test cache expiration
        await asyncio.sleep(0.1)  # Simulate time passing
        expired_value = await memory_cache.retrieve("test_key")
        assert expired_value is None
        
        # Test cache size limit
        for i in range(100):
            await memory_cache.store({
                "key": f"key_{i}",
                "value": f"value_{i}",
                "ttl": 3600
            })
        assert memory_cache.size <= 100
    
    @pytest.mark.asyncio
    async def test_redis_cache_operations(self, redis_cache):
        """Test Redis cache operations."""
        # Test storing data
        cache_data = {
            "key": "test_key",
            "value": "test_value",
            "ttl": 3600
        }
        
        await redis_cache.store(cache_data)
        
        # Test retrieving data
        retrieved_value = await redis_cache.retrieve("test_key")
        assert retrieved_value == "test_value"
        
        # Test cache expiration
        await asyncio.sleep(0.1)  # Simulate time passing
        expired_value = await redis_cache.retrieve("test_key")
        assert expired_value is None
        
        # Test batch operations
        batch_data = [
            {"key": f"key_{i}", "value": f"value_{i}", "ttl": 3600}
            for i in range(5)
        ]
        
        await redis_cache.store_batch(batch_data)
        batch_values = await redis_cache.retrieve_batch([f"key_{i}" for i in range(5)])
        assert len(batch_values) == 5
        assert all(v is not None for v in batch_values)
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, cache, memory_cache, redis_cache):
        """Test cache error handling."""
        # Test cache errors
        with pytest.raises(ValueError):
            await cache.store(None)
        
        # Test memory cache errors
        with pytest.raises(ValueError):
            await memory_cache.store(None)
        
        # Test Redis cache errors
        with pytest.raises(ValueError):
            await redis_cache.store(None)
    
    @pytest.mark.asyncio
    async def test_cache_metrics(self, cache, memory_cache, redis_cache):
        """Test cache metrics collection."""
        # Test cache metrics
        cache_metrics = await cache.collect_metrics()
        assert cache_metrics is not None
        assert isinstance(cache_metrics, dict)
        assert "hit_rate" in cache_metrics
        assert "miss_rate" in cache_metrics
        
        # Test memory cache metrics
        memory_metrics = await memory_cache.collect_metrics()
        assert memory_metrics is not None
        assert isinstance(memory_metrics, dict)
        assert "size" in memory_metrics
        assert "eviction_count" in memory_metrics
        
        # Test Redis cache metrics
        redis_metrics = await redis_cache.collect_metrics()
        assert redis_metrics is not None
        assert isinstance(redis_metrics, dict)
        assert "connected" in redis_metrics
        assert "memory_usage" in redis_metrics
    
    @pytest.mark.asyncio
    async def test_cache_optimization(self, cache, memory_cache, redis_cache):
        """Test cache optimization operations."""
        # Test cache optimization
        cache_params = {
            "max_size": 1000,
            "eviction_policy": "lru",
            "ttl": 3600
        }
        
        optimized_cache = await cache.optimize(cache_params)
        assert optimized_cache is not None
        assert optimized_cache.is_optimized
        
        # Test memory cache optimization
        memory_params = {
            "max_size": 100,
            "eviction_policy": "fifo",
            "cleanup_interval": 60
        }
        
        optimized_memory = await memory_cache.optimize(memory_params)
        assert optimized_memory is not None
        assert optimized_memory.is_optimized
        
        # Test Redis cache optimization
        redis_params = {
            "max_memory": "1gb",
            "eviction_policy": "allkeys-lru",
            "compression": True
        }
        
        optimized_redis = await redis_cache.optimize(redis_params)
        assert optimized_redis is not None
        assert optimized_redis.is_optimized
    
    @pytest.mark.asyncio
    async def test_cache_persistence(self, cache, memory_cache, redis_cache):
        """Test cache persistence operations."""
        # Test cache persistence
        await cache.persist()
        assert await cache.check_persistence()
        
        # Test memory cache persistence
        await memory_cache.persist()
        assert await memory_cache.check_persistence()
        
        # Test Redis cache persistence
        await redis_cache.persist()
        assert await redis_cache.check_persistence()
        
        # Test cache recovery
        await cache.recover()
        assert await cache.check_recovery()
        
        # Test memory cache recovery
        await memory_cache.recover()
        assert await memory_cache.check_recovery()
        
        # Test Redis cache recovery
        await redis_cache.recover()
        assert await redis_cache.check_recovery() 