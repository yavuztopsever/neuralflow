"""
Unit tests for cache functionality.
"""

import pytest
import time
from typing import Optional, Dict, Any
from src.storage.storage.memory.cache.base import BaseCache, CacheConfig, CacheEntry

class MockCache(BaseCache[str]):
    """Mock cache implementation for testing."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        super().__init__(config)
        self._initialized = False
    
    def initialize(self) -> None:
        self._initialized = True

@pytest.fixture
def cache():
    """Create a mock cache."""
    config = CacheConfig(
        max_size=5,
        ttl=3600  # 1 hour
    )
    cache = MockCache(config)
    cache.initialize()
    return cache

class TestCache:
    """Test suite for cache functionality."""
    
    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.config.max_size == 5
        assert cache.config.ttl == 3600
        assert cache._initialized
        assert len(cache._entries) == 0
    
    def test_set_get(self, cache):
        """Test setting and getting values."""
        # Set value
        success = cache.set("key1", "value1")
        assert success
        assert len(cache._entries) == 1
        
        # Get value
        value = cache.get("key1")
        assert value == "value1"
        
        # Get non-existent value
        value = cache.get("nonexistent")
        assert value is None
    
    def test_ttl(self, cache):
        """Test time-to-live functionality."""
        # Set value with short TTL
        cache.set("key1", "value1", ttl=1)  # 1 second TTL
        
        # Value should exist initially
        assert cache.get("key1") == "value1"
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Value should be gone
        assert cache.get("key1") is None
    
    def test_max_size(self, cache):
        """Test maximum size functionality."""
        # Fill cache to max size
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")
        assert len(cache._entries) == 5
        
        # Add one more item
        cache.set("key5", "value5")
        assert len(cache._entries) == 5  # Size should stay at max
        assert "key5" in cache._entries  # New item should be there
    
    def test_exists(self, cache):
        """Test key existence checking."""
        cache.set("key1", "value1")
        
        assert cache.exists("key1")
        assert not cache.exists("nonexistent")
        
        # Test with TTL
        cache.set("key2", "value2", ttl=1)
        assert cache.exists("key2")
        time.sleep(1.1)
        assert not cache.exists("key2")
    
    def test_delete(self, cache):
        """Test value deletion."""
        cache.set("key1", "value1")
        
        # Delete existing key
        success = cache.delete("key1")
        assert success
        assert not cache.exists("key1")
        
        # Delete non-existent key
        success = cache.delete("nonexistent")
        assert not success
    
    def test_clear(self, cache):
        """Test cache clearing."""
        # Add some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Clear cache
        success = cache.clear()
        assert success
        assert len(cache._entries) == 0
    
    def test_cleanup(self, cache):
        """Test cache cleanup."""
        # Add some values
        cache.set("key1", "value1")  # No TTL
        cache.set("key2", "value2", ttl=1)  # Short TTL
        cache.set("key3", "value3")  # No TTL
        
        # Access key3 to update last access time
        cache.get("key3")
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Modify last access time of key1 to simulate idle
        entry = cache._entries["key1"]
        entry.last_access = time.time() - 3700  # More than 1 hour old
        
        # Run cleanup
        removed = cache.cleanup(max_idle=3600)  # 1 hour max idle
        assert removed == 2  # key1 (idle) and key2 (TTL) should be removed
        assert len(cache._entries) == 1
        assert "key3" in cache._entries
    
    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        # Add some values
        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=1)
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Get stats
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['entries']['total'] == 2
        assert stats['entries']['expired'] == 1  # key2 is expired
        assert stats['config']['max_size'] == 5
        assert stats['config']['ttl'] == 3600 