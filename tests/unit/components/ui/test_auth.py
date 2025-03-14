"""
Unit tests for authentication component functionality.
"""
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.ui.auth import UserManager, User, Session, AuthToken

class TestUserManager:
    """Test suite for user management functionality."""
    
    @pytest.fixture
    def user_manager(self):
        """Create a user manager for testing."""
        return UserManager(
            token_expiry=3600,  # 1 hour
            session_ttl=86400,  # 24 hours
            max_sessions=5
        )
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            role="user",
            created_at=datetime.now(),
            last_login=datetime.now(),
            metadata={
                "department": "engineering",
                "location": "remote"
            }
        )
    
    @pytest.mark.asyncio
    async def test_user_manager_initialization(self, user_manager):
        """Test user manager initialization."""
        assert user_manager.token_expiry == 3600
        assert user_manager.session_ttl == 86400
        assert user_manager.max_sessions == 5
        assert user_manager._users == {}
        assert user_manager._sessions == {}
        assert user_manager._tokens == {}
    
    @pytest.mark.asyncio
    async def test_user_management(self, user_manager, sample_user):
        """Test user management operations."""
        # Create user
        await user_manager.create_user(sample_user)
        assert sample_user.id in user_manager._users
        stored_user = user_manager._users[sample_user.id]
        assert stored_user.username == sample_user.username
        assert stored_user.email == sample_user.email
        
        # Get user
        retrieved_user = await user_manager.get_user(sample_user.id)
        assert retrieved_user is not None
        assert retrieved_user.id == sample_user.id
        
        # Update user
        updated_user = User(
            id=sample_user.id,
            username="updated_user",
            email="updated@example.com",
            password_hash=sample_user.password_hash,
            role=sample_user.role,
            created_at=sample_user.created_at,
            last_login=sample_user.last_login,
            metadata=sample_user.metadata
        )
        await user_manager.update_user(updated_user)
        retrieved_user = await user_manager.get_user(sample_user.id)
        assert retrieved_user.username == "updated_user"
        assert retrieved_user.email == "updated@example.com"
        
        # Delete user
        await user_manager.delete_user(sample_user.id)
        assert sample_user.id not in user_manager._users
    
    @pytest.mark.asyncio
    async def test_session_management(self, user_manager, sample_user):
        """Test session management operations."""
        await user_manager.create_user(sample_user)
        
        # Create session
        session = await user_manager.create_session(sample_user.id)
        assert session is not None
        assert session.user_id == sample_user.id
        assert session.id in user_manager._sessions
        
        # Get session
        retrieved_session = await user_manager.get_session(session.id)
        assert retrieved_session is not None
        assert retrieved_session.id == session.id
        
        # Update session
        updated_session = Session(
            id=session.id,
            user_id=session.user_id,
            created_at=session.created_at,
            last_activity=datetime.now(),
            metadata={"ip": "127.0.0.1"}
        )
        await user_manager.update_session(updated_session)
        retrieved_session = await user_manager.get_session(session.id)
        assert retrieved_session.metadata["ip"] == "127.0.0.1"
        
        # Delete session
        await user_manager.delete_session(session.id)
        assert session.id not in user_manager._sessions
    
    @pytest.mark.asyncio
    async def test_token_management(self, user_manager, sample_user):
        """Test token management operations."""
        await user_manager.create_user(sample_user)
        
        # Create token
        token = await user_manager.create_token(sample_user.id)
        assert token is not None
        assert token.user_id == sample_user.id
        assert token.id in user_manager._tokens
        
        # Validate token
        is_valid = await user_manager.validate_token(token.id)
        assert is_valid is True
        
        # Refresh token
        refreshed_token = await user_manager.refresh_token(token.id)
        assert refreshed_token is not None
        assert refreshed_token.id == token.id
        assert refreshed_token.expires_at > token.expires_at
        
        # Revoke token
        await user_manager.revoke_token(token.id)
        is_valid = await user_manager.validate_token(token.id)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_session_limits(self, user_manager, sample_user):
        """Test session limit enforcement."""
        await user_manager.create_user(sample_user)
        
        # Create maximum allowed sessions
        sessions = []
        for _ in range(user_manager.max_sessions):
            session = await user_manager.create_session(sample_user.id)
            sessions.append(session)
        
        # Try to create one more session
        with pytest.raises(ValueError):
            await user_manager.create_session(sample_user.id)
        
        # Delete one session and create a new one
        await user_manager.delete_session(sessions[0].id)
        new_session = await user_manager.create_session(sample_user.id)
        assert new_session is not None
        assert new_session.id in user_manager._sessions
    
    @pytest.mark.asyncio
    async def test_session_expiration(self, user_manager, sample_user):
        """Test session expiration handling."""
        await user_manager.create_user(sample_user)
        
        # Create session with short TTL
        user_manager.session_ttl = 1  # 1 second
        session = await user_manager.create_session(sample_user.id)
        
        # Wait for session to expire
        await asyncio.sleep(2)
        
        # Verify session is expired
        retrieved_session = await user_manager.get_session(session.id)
        assert retrieved_session is None
    
    @pytest.mark.asyncio
    async def test_token_expiration(self, user_manager, sample_user):
        """Test token expiration handling."""
        await user_manager.create_user(sample_user)
        
        # Create token with short expiry
        user_manager.token_expiry = 1  # 1 second
        token = await user_manager.create_token(sample_user.id)
        
        # Wait for token to expire
        await asyncio.sleep(2)
        
        # Verify token is expired
        is_valid = await user_manager.validate_token(token.id)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_user_metrics(self, user_manager, sample_user):
        """Test user metrics collection."""
        await user_manager.create_user(sample_user)
        
        # Create some sessions and tokens
        for _ in range(3):
            await user_manager.create_session(sample_user.id)
            await user_manager.create_token(sample_user.id)
        
        # Get metrics
        metrics = await user_manager.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_users" in metrics
        assert "active_sessions" in metrics
        assert "active_tokens" in metrics
    
    @pytest.mark.asyncio
    async def test_user_error_handling(self, user_manager):
        """Test user error handling."""
        # Test invalid user
        with pytest.raises(ValueError):
            await user_manager.create_user(None)
        
        # Test invalid user ID
        with pytest.raises(ValueError):
            await user_manager.get_user(None)
        
        # Test getting non-existent user
        user = await user_manager.get_user("non_existent")
        assert user is None
        
        # Test updating non-existent user
        with pytest.raises(KeyError):
            await user_manager.update_user(User(
                id="non_existent",
                username="test",
                email="test@example.com",
                password_hash="hash",
                role="user",
                created_at=datetime.now(),
                last_login=datetime.now(),
                metadata={}
            ))
        
        # Test deleting non-existent user
        with pytest.raises(KeyError):
            await user_manager.delete_user("non_existent")
        
        # Test invalid session
        with pytest.raises(ValueError):
            await user_manager.create_session(None)
        
        # Test invalid token
        with pytest.raises(ValueError):
            await user_manager.create_token(None) 