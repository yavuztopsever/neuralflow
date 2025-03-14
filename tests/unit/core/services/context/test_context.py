"""
Unit tests for context service functionality.
"""
import pytest
from typing import Dict, Any, List
from datetime import datetime, timedelta

from src.core.services.context.context_service import ContextService
from src.core.services.context.context_handler import ContextHandler

class TestContextService:
    """Test suite for context service functionality."""
    
    @pytest.fixture
    def context_service(self):
        """Create a context service for testing."""
        return ContextService(
            max_context_size=1000,
            context_ttl=3600,  # 1 hour
            cleanup_interval=300  # 5 minutes
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context data for testing."""
        return {
            "session_id": "test_session",
            "user_id": "test_user",
            "timestamp": datetime.now(),
            "data": {
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ],
                "metadata": {
                    "language": "en",
                    "platform": "web"
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_context_initialization(self, context_service):
        """Test context service initialization."""
        assert context_service.max_context_size == 1000
        assert context_service.context_ttl == 3600
        assert context_service.cleanup_interval == 300
        assert context_service._contexts == {}
        assert context_service._last_cleanup is not None
    
    @pytest.mark.asyncio
    async def test_context_management(self, context_service, sample_context):
        """Test context management operations."""
        # Add context
        await context_service.add_context(
            sample_context["session_id"],
            sample_context
        )
        
        # Verify context was added
        assert sample_context["session_id"] in context_service._contexts
        stored_context = context_service._contexts[sample_context["session_id"]]
        assert stored_context["data"] == sample_context["data"]
        assert stored_context["timestamp"] is not None
        
        # Get context
        retrieved_context = await context_service.get_context(
            sample_context["session_id"]
        )
        assert retrieved_context is not None
        assert retrieved_context["data"] == sample_context["data"]
        
        # Update context
        updated_data = {
            "conversation_history": [
                {"role": "user", "content": "Updated message"}
            ]
        }
        await context_service.update_context(
            sample_context["session_id"],
            updated_data
        )
        updated_context = await context_service.get_context(
            sample_context["session_id"]
        )
        assert updated_context["data"]["conversation_history"] == updated_data["conversation_history"]
        
        # Delete context
        await context_service.delete_context(sample_context["session_id"])
        assert sample_context["session_id"] not in context_service._contexts
    
    @pytest.mark.asyncio
    async def test_context_expiration(self, context_service, sample_context):
        """Test context expiration handling."""
        # Add context with short TTL
        context_service.context_ttl = 1  # 1 second
        await context_service.add_context(
            sample_context["session_id"],
            sample_context
        )
        
        # Wait for context to expire
        await asyncio.sleep(2)
        
        # Verify context is expired
        retrieved_context = await context_service.get_context(
            sample_context["session_id"]
        )
        assert retrieved_context is None
    
    @pytest.mark.asyncio
    async def test_context_cleanup(self, context_service, sample_context):
        """Test context cleanup functionality."""
        # Add multiple contexts
        for i in range(3):
            session_id = f"test_session_{i}"
            context = sample_context.copy()
            context["session_id"] = session_id
            await context_service.add_context(session_id, context)
        
        # Force cleanup
        await context_service._cleanup_expired_contexts()
        
        # Verify all contexts are still present (not expired)
        assert len(context_service._contexts) == 3
        
        # Expire some contexts
        for session_id in list(context_service._contexts.keys())[:2]:
            context_service._contexts[session_id]["timestamp"] = (
                datetime.now() - timedelta(seconds=context_service.context_ttl + 1)
            )
        
        # Run cleanup
        await context_service._cleanup_expired_contexts()
        
        # Verify only one context remains
        assert len(context_service._contexts) == 1
    
    @pytest.mark.asyncio
    async def test_context_size_limits(self, context_service):
        """Test context size limits."""
        # Create large context
        large_context = {
            "session_id": "large_session",
            "data": {
                "large_array": ["x" * 1000 for _ in range(1000)]
            }
        }
        
        # Try to add context exceeding size limit
        with pytest.raises(ValueError):
            await context_service.add_context(
                large_context["session_id"],
                large_context
            )
    
    @pytest.mark.asyncio
    async def test_context_metrics(self, context_service, sample_context):
        """Test context metrics collection."""
        # Add some contexts
        for i in range(3):
            session_id = f"test_session_{i}"
            context = sample_context.copy()
            context["session_id"] = session_id
            await context_service.add_context(session_id, context)
        
        # Get metrics
        metrics = await context_service.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_contexts" in metrics
        assert "active_contexts" in metrics
        assert "average_context_size" in metrics
    
    @pytest.mark.asyncio
    async def test_context_error_handling(self, context_service):
        """Test context error handling."""
        # Test invalid session ID
        with pytest.raises(ValueError):
            await context_service.add_context(None, {})
        
        # Test invalid context data
        with pytest.raises(ValueError):
            await context_service.add_context("test_session", None)
        
        # Test getting non-existent context
        context = await context_service.get_context("non_existent")
        assert context is None
        
        # Test updating non-existent context
        with pytest.raises(KeyError):
            await context_service.update_context("non_existent", {})
        
        # Test deleting non-existent context
        with pytest.raises(KeyError):
            await context_service.delete_context("non_existent") 