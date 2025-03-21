"""
Unit tests for context management functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.core.nodes.context import ContextNode

class TestContextManagement:
    """Test suite for context management functionality."""
    
    @pytest.fixture
    def context_node(self):
        """Create a context node for testing."""
        return ContextNode("test_context", "context", {})
    
    @pytest.mark.asyncio
    async def test_context_aggregation(self, context_node):
        """Test context aggregation operations."""
        context_data = await context_node.aggregate_context({
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        })
        
        assert context_data is not None
        assert "memory_data" in context_data
        assert "current_input" in context_data
        assert "session_data" in context_data
    
    @pytest.mark.asyncio
    async def test_context_summarization(self, context_node):
        """Test context summarization operations."""
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        summary = await context_node.summarize_context(context_data)
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_emotion_context_generation(self, context_node):
        """Test emotion context generation."""
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        emotion_context = await context_node.generate_emotion_context(context_data)
        assert emotion_context is not None
        assert isinstance(emotion_context, dict)
        assert "emotion" in emotion_context
        assert "confidence" in emotion_context
    
    @pytest.mark.asyncio
    async def test_context_pooling(self, context_node):
        """Test context pooling operations."""
        context_pool = await context_node.pool_context([
            {"source1": "data1"},
            {"source2": "data2"},
            {"source3": "data3"}
        ])
        
        assert context_pool is not None
        assert isinstance(context_pool, dict)
        assert len(context_pool) > 0
    
    @pytest.mark.asyncio
    async def test_dynamic_context_filtering(self, context_node):
        """Test dynamic context filtering."""
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"},
            "irrelevant_data": "should be filtered"
        }
        
        filtered_context = await context_node.filter_context(context_data, "test")
        assert filtered_context is not None
        assert "irrelevant_data" not in filtered_context
        assert "current_input" in filtered_context
    
    @pytest.mark.asyncio
    async def test_context_sufficiency_check(self, context_node):
        """Test context sufficiency check."""
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        is_sufficient = await context_node.check_context_sufficiency(context_data)
        assert isinstance(is_sufficient, bool)
        
        insufficient_context = {"current_input": "test input"}
        is_sufficient = await context_node.check_context_sufficiency(insufficient_context)
        assert not is_sufficient
    
    @pytest.mark.asyncio
    async def test_context_update(self, context_node):
        """Test context update operations."""
        initial_context = {
            "memory_data": {"key": "value"},
            "current_input": "test input"
        }
        
        updated_context = await context_node.update_context(
            initial_context,
            {"new_data": "new value"}
        )
        
        assert updated_context is not None
        assert "new_data" in updated_context
        assert updated_context["new_data"] == "new value"
    
    @pytest.mark.asyncio
    async def test_context_validation(self, context_node):
        """Test context validation operations."""
        valid_context = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        is_valid = await context_node.validate_context(valid_context)
        assert is_valid
        
        invalid_context = {
            "current_input": ""  # Empty input
        }
        
        is_valid = await context_node.validate_context(invalid_context)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_context_serialization(self, context_node):
        """Test context serialization operations."""
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        serialized = await context_node.serialize_context(context_data)
        assert serialized is not None
        assert isinstance(serialized, str)
        
        deserialized = await context_node.deserialize_context(serialized)
        assert deserialized is not None
        assert deserialized == context_data 