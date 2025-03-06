"""
Unit tests for memory management functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.core.nodes.memory import MemoryNode
from src.storage.vector import VectorStore
from src.storage.cache import Cache

class TestMemoryManagement:
    """Test suite for memory management functionality."""
    
    @pytest.fixture
    def memory_node(self):
        """Create a memory node for testing."""
        return MemoryNode("test_memory", "memory", {})
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store for testing."""
        return VectorStore()
    
    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        return Cache()
    
    @pytest.mark.asyncio
    async def test_memory_storage(self, memory_node):
        """Test memory storage operations."""
        # Test storing data
        await memory_node.store("test_key", "test_value")
        
        # Test retrieving data
        value = await memory_node.retrieve("test_key")
        assert value == "test_value"
        
        # Test storing multiple values
        await memory_node.store("test_key2", {"key": "value"})
        value2 = await memory_node.retrieve("test_key2")
        assert value2 == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_memory_retrieval(self, memory_node):
        """Test memory retrieval operations."""
        # Store test data
        test_data = {"key": "value", "list": [1, 2, 3]}
        await memory_node.store("test_data", test_data)
        
        # Test retrieving data
        retrieved_data = await memory_node.retrieve("test_data")
        assert retrieved_data == test_data
        
        # Test retrieving non-existent data
        with pytest.raises(KeyError):
            await memory_node.retrieve("non_existent_key")
    
    @pytest.mark.asyncio
    async def test_memory_update(self, memory_node):
        """Test memory update operations."""
        # Store initial data
        await memory_node.store("test_key", "initial_value")
        
        # Update data
        await memory_node.update("test_key", "updated_value")
        
        # Verify update
        value = await memory_node.retrieve("test_key")
        assert value == "updated_value"
        
        # Test updating non-existent key
        with pytest.raises(KeyError):
            await memory_node.update("non_existent_key", "value")
    
    @pytest.mark.asyncio
    async def test_memory_deletion(self, memory_node):
        """Test memory deletion operations."""
        # Store test data
        await memory_node.store("test_key", "test_value")
        
        # Delete data
        await memory_node.delete("test_key")
        
        # Verify deletion
        with pytest.raises(KeyError):
            await memory_node.retrieve("test_key")
        
        # Test deleting non-existent key
        with pytest.raises(KeyError):
            await memory_node.delete("non_existent_key")
    
    @pytest.mark.asyncio
    async def test_memory_search(self, memory_node):
        """Test memory search operations."""
        # Store test data
        await memory_node.store("key1", "value1")
        await memory_node.store("key2", "value2")
        await memory_node.store("key3", "value3")
        
        # Test searching data
        results = await memory_node.search("value")
        assert len(results) == 3
        assert all("value" in result for result in results)
        
        # Test searching with no matches
        results = await memory_node.search("nonexistent")
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_node):
        """Test memory cleanup operations."""
        # Store test data
        await memory_node.store("key1", "value1")
        await memory_node.store("key2", "value2")
        
        # Cleanup memory
        await memory_node.cleanup()
        
        # Verify cleanup
        with pytest.raises(KeyError):
            await memory_node.retrieve("key1")
        with pytest.raises(KeyError):
            await memory_node.retrieve("key2") 