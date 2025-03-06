"""
Base provider interfaces for authentication providers.
This module provides base classes for authentication provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

T = TypeVar('T')

class AuthType(Enum):
    """Authentication types."""
    PASSWORD = "password"
    TOKEN = "token"
    OAUTH = "oauth"
    API_KEY = "api_key"

@dataclass
class User:
    """User with metadata."""
    
    id: str
    username: str
    email: str
    auth_type: AuthType
    created: datetime
    modified: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary.
        
        Returns:
            Dictionary representation of the user
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'auth_type': self.auth_type.value,
            'created': self.created.isoformat(),
            'modified': self.modified.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'metadata': self.metadata or {}
        }

@dataclass
class Token:
    """Authentication token with metadata."""
    
    id: str
    user_id: str
    token: str
    type: str
    created: datetime
    expires: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary.
        
        Returns:
            Dictionary representation of the token
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'token': self.token,
            'type': self.type,
            'created': self.created.isoformat(),
            'expires': self.expires.isoformat(),
            'metadata': self.metadata or {}
        }

class AuthConfig:
    """Configuration for authentication providers."""
    
    def __init__(self,
                 token_expiry: Optional[int] = None,
                 refresh_token_expiry: Optional[int] = None,
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            token_expiry: Token expiry time in seconds
            refresh_token_expiry: Refresh token expiry time in seconds
            **kwargs: Additional configuration parameters
        """
        self.token_expiry = token_expiry or 3600  # 1 hour
        self.refresh_token_expiry = refresh_token_expiry or 2592000  # 30 days
        self.extra_params = kwargs

class BaseAuthProvider(ABC):
    """Base class for authentication providers."""
    
    def __init__(self, provider_id: str,
                 config: AuthConfig,
                 **kwargs):
        """Initialize the provider.
        
        Args:
            provider_id: Unique identifier for the provider
            config: Authentication provider configuration
            **kwargs: Additional initialization parameters
        """
        self.id = provider_id
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
        self._users: Dict[str, User] = {}
        self._tokens: Dict[str, Token] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    def create_user(self,
                   username: str,
                   email: str,
                   auth_type: AuthType,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[User]:
        """Create a new user.
        
        Args:
            username: Username
            email: Email address
            auth_type: Authentication type
            metadata: Optional user metadata
            
        Returns:
            Created user or None if failed
        """
        try:
            if any(u.username == username for u in self._users.values()):
                return None
            
            if any(u.email == email for u in self._users.values()):
                return None
            
            user = User(
                id=str(uuid.uuid4()),
                username=username,
                email=email,
                auth_type=auth_type,
                created=datetime.now(),
                modified=datetime.now(),
                metadata=metadata
            )
            
            self._users[user.id] = user
            self.modified = datetime.now().isoformat()
            return user
        except Exception as e:
            logger.error(f"Failed to create user in provider {self.id}: {e}")
            return None
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User or None if not found
        """
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username.
        
        Args:
            username: Username
            
        Returns:
            User or None if not found
        """
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email.
        
        Args:
            email: Email address
            
        Returns:
            User or None if not found
        """
        for user in self._users.values():
            if user.email == email:
                return user
        return None
    
    def update_user(self, user: User) -> bool:
        """Update a user.
        
        Args:
            user: User to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if user.id not in self._users:
                return False
            
            user.modified = datetime.now()
            self._users[user.id] = user
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to update user in provider {self.id}: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if user_id not in self._users:
                return False
            
            # Delete user's tokens
            self._tokens = {
                t.id: t for t in self._tokens.values()
                if t.user_id != user_id
            }
            
            del self._users[user_id]
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to delete user from provider {self.id}: {e}")
            return False
    
    def create_token(self,
                    user_id: str,
                    token_type: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[Token]:
        """Create a new token.
        
        Args:
            user_id: User ID
            token_type: Token type
            metadata: Optional token metadata
            
        Returns:
            Created token or None if failed
        """
        try:
            if user_id not in self._users:
                return None
            
            expiry = (
                datetime.now() + timedelta(seconds=self.config.refresh_token_expiry)
                if token_type == 'refresh'
                else datetime.now() + timedelta(seconds=self.config.token_expiry)
            )
            
            token = Token(
                id=str(uuid.uuid4()),
                user_id=user_id,
                token=str(uuid.uuid4()),
                type=token_type,
                created=datetime.now(),
                expires=expiry,
                metadata=metadata
            )
            
            self._tokens[token.id] = token
            self.modified = datetime.now().isoformat()
            return token
        except Exception as e:
            logger.error(f"Failed to create token in provider {self.id}: {e}")
            return None
    
    def get_token(self, token_id: str) -> Optional[Token]:
        """Get a token by ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token or None if not found/expired
        """
        token = self._tokens.get(token_id)
        if token and token.expires > datetime.now():
            return token
        return None
    
    def delete_token(self, token_id: str) -> bool:
        """Delete a token.
        
        Args:
            token_id: Token ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if token_id not in self._tokens:
                return False
            
            del self._tokens[token_id]
            self.modified = datetime.now().isoformat()
            return True
        except Exception as e:
            logger.error(f"Failed to delete token from provider {self.id}: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information.
        
        Returns:
            Dictionary containing provider information
        """
        return {
            'id': self.id,
            'type': type(self).__name__,
            'created': self.created,
            'modified': self.modified,
            'config': {
                'token_expiry': self.config.token_expiry,
                'refresh_token_expiry': self.config.refresh_token_expiry,
                'extra_params': self.config.extra_params
            },
            'stats': {
                'total_users': len(self._users),
                'total_tokens': len(self._tokens),
                'active_users': len([u for u in self._users.values() if u.is_active])
            }
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            return {
                'total_users': len(self._users),
                'auth_types': {
                    at.value: len([u for u in self._users.values() if u.auth_type == at])
                    for at in AuthType
                },
                'user_status': {
                    'active': len([u for u in self._users.values() if u.is_active]),
                    'inactive': len([u for u in self._users.values() if not u.is_active]),
                    'verified': len([u for u in self._users.values() if u.is_verified]),
                    'unverified': len([u for u in self._users.values() if not u.is_verified])
                },
                'tokens': {
                    'total': len(self._tokens),
                    'expired': len([t for t in self._tokens.values() if t.expires <= datetime.now()]),
                    'types': {
                        t.type: len([t for t in self._tokens.values() if t.type == t])
                        for t in set(t.type for t in self._tokens.values())
                    }
                }
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 