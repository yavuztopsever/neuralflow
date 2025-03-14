"""
Unit tests for chat component functionality.
"""
import pytest
from datetime import datetime
from typing import Dict, Any, List

from src.ui.chat import ChatInterface, Message, ChatSession

class TestChatInterface:
    """Test suite for chat interface functionality."""
    
    @pytest.fixture
    def chat_interface(self):
        """Create a chat interface for testing."""
        return ChatInterface(
            max_history=100,
            message_ttl=3600,  # 1 hour
            auto_cleanup=True
        )
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message for testing."""
        return Message(
            content="Hello, World!",
            role="user",
            timestamp=datetime.now(),
            metadata={
                "user_id": "test_user",
                "session_id": "test_session"
            }
        )
    
    @pytest.mark.asyncio
    async def test_chat_initialization(self, chat_interface):
        """Test chat interface initialization."""
        assert chat_interface.max_history == 100
        assert chat_interface.message_ttl == 3600
        assert chat_interface.auto_cleanup is True
        assert chat_interface._sessions == {}
        assert chat_interface._last_cleanup is not None
    
    @pytest.mark.asyncio
    async def test_session_management(self, chat_interface, sample_message):
        """Test chat session management."""
        session_id = "test_session"
        
        # Create session
        session = await chat_interface.create_session(session_id)
        assert session is not None
        assert session_id in chat_interface._sessions
        assert session.id == session_id
        
        # Add message to session
        await chat_interface.add_message(session_id, sample_message)
        session = chat_interface._sessions[session_id]
        assert len(session.messages) == 1
        assert session.messages[0].content == sample_message.content
        
        # Get session
        retrieved_session = await chat_interface.get_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session.id == session_id
        
        # Delete session
        await chat_interface.delete_session(session_id)
        assert session_id not in chat_interface._sessions
    
    @pytest.mark.asyncio
    async def test_message_management(self, chat_interface, sample_message):
        """Test message management operations."""
        session_id = "test_session"
        await chat_interface.create_session(session_id)
        
        # Add message
        await chat_interface.add_message(session_id, sample_message)
        session = chat_interface._sessions[session_id]
        assert len(session.messages) == 1
        
        # Get message
        message = await chat_interface.get_message(session_id, 0)
        assert message is not None
        assert message.content == sample_message.content
        
        # Update message
        updated_content = "Updated message"
        await chat_interface.update_message(session_id, 0, updated_content)
        message = await chat_interface.get_message(session_id, 0)
        assert message.content == updated_content
        
        # Delete message
        await chat_interface.delete_message(session_id, 0)
        session = chat_interface._sessions[session_id]
        assert len(session.messages) == 0
    
    @pytest.mark.asyncio
    async def test_message_history(self, chat_interface):
        """Test message history functionality."""
        session_id = "test_session"
        await chat_interface.create_session(session_id)
        
        # Add multiple messages
        messages = [
            Message(
                content=f"Message {i}",
                role="user" if i % 2 == 0 else "assistant",
                timestamp=datetime.now(),
                metadata={"user_id": "test_user"}
            )
            for i in range(5)
        ]
        
        for message in messages:
            await chat_interface.add_message(session_id, message)
        
        # Get history
        history = await chat_interface.get_history(session_id)
        assert len(history) == 5
        assert all(msg.content == f"Message {i}" for i, msg in enumerate(history))
        
        # Test history limit
        chat_interface.max_history = 3
        history = await chat_interface.get_history(session_id)
        assert len(history) == 3
        assert history[-1].content == "Message 4"
    
    @pytest.mark.asyncio
    async def test_message_expiration(self, chat_interface, sample_message):
        """Test message expiration handling."""
        session_id = "test_session"
        await chat_interface.create_session(session_id)
        
        # Add message with short TTL
        chat_interface.message_ttl = 1  # 1 second
        await chat_interface.add_message(session_id, sample_message)
        
        # Wait for message to expire
        await asyncio.sleep(2)
        
        # Verify message is expired
        history = await chat_interface.get_history(session_id)
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_chat_cleanup(self, chat_interface, sample_message):
        """Test chat cleanup functionality."""
        session_id = "test_session"
        await chat_interface.create_session(session_id)
        await chat_interface.add_message(session_id, sample_message)
        
        # Force cleanup
        await chat_interface._cleanup_expired_messages()
        
        # Verify session still exists (message not expired)
        assert session_id in chat_interface._sessions
        
        # Expire message
        chat_interface._sessions[session_id].messages[0].timestamp = (
            datetime.now() - timedelta(seconds=chat_interface.message_ttl + 1)
        )
        
        # Run cleanup
        await chat_interface._cleanup_expired_messages()
        
        # Verify message is removed
        session = chat_interface._sessions[session_id]
        assert len(session.messages) == 0
    
    @pytest.mark.asyncio
    async def test_chat_metrics(self, chat_interface, sample_message):
        """Test chat metrics collection."""
        session_id = "test_session"
        await chat_interface.create_session(session_id)
        await chat_interface.add_message(session_id, sample_message)
        
        # Get metrics
        metrics = await chat_interface.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_sessions" in metrics
        assert "total_messages" in metrics
        assert "active_sessions" in metrics
    
    @pytest.mark.asyncio
    async def test_chat_error_handling(self, chat_interface):
        """Test chat error handling."""
        # Test invalid session ID
        with pytest.raises(ValueError):
            await chat_interface.create_session(None)
        
        # Test invalid message
        with pytest.raises(ValueError):
            await chat_interface.add_message("test_session", None)
        
        # Test getting non-existent session
        session = await chat_interface.get_session("non_existent")
        assert session is None
        
        # Test getting non-existent message
        message = await chat_interface.get_message("test_session", 0)
        assert message is None
        
        # Test updating non-existent message
        with pytest.raises(IndexError):
            await chat_interface.update_message("test_session", 0, "test")
        
        # Test deleting non-existent message
        with pytest.raises(IndexError):
            await chat_interface.delete_message("test_session", 0) 