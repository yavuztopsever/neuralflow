"""
Unit tests for cache functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.storage.cache import Cache

class TestCache:
    """Test suite for cache functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        return Cache()
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, cache):
        """Test cache operations."""
        cache_data = {
            "key": "test_key",
            "value": "test_value",
            "ttl": 3600
        }
        
        await cache.store(cache_data)
        
        retrieved_value = await cache.retrieve("test_key")
        assert retrieved_value == "test_value"
        
        await asyncio.sleep(0.1)  # Simulate time passing
        expired_value = await cache.retrieve("test_key")
        assert expired_value is None
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, cache):
        """Test cache error handling."""
        with pytest.raises(ValueError):
            await cache.store(None)
        
        with pytest.raises(KeyError):
            await cache.retrieve("non_existent_key")
    
    @pytest.mark.asyncio
    async def test_cache_metrics(self, cache):
        """Test cache metrics collection."""
        cache_metrics = await cache.collect_metrics()
        assert cache_metrics is not None
        assert isinstance(cache_metrics, dict)
        assert "hit_rate" in cache_metrics
        assert "miss_rate" in cache_metrics
    
    @pytest.mark.asyncio
    async def test_cache_optimization(self, cache):
        """Test cache optimization operations."""
        cache_params = {
            "max_size": 1000,
            "eviction_policy": "lru",
            "ttl": 3600
        }
        
        optimized_cache = await cache.optimize(cache_params)
        assert optimized_cache is not None
        assert optimized_cache.is_optimized 