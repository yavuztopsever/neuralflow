"""
Security utilities for the LangGraph application.
This module provides functionality for authentication and authorization.
"""

import os
import logging
import hashlib
import secrets
import jwt
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime, timedelta
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager
from utils.logging.manager import LogManager

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Exception raised for security errors."""
    pass

class User:
    """Represents a user in the system."""
    
    def __init__(self, user_id: str,
                 username: str,
                 password_hash: str,
                 roles: List[str],
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a user.
        
        Args:
            user_id: Unique identifier for the user
            username: Username
            password_hash: Hashed password
            roles: List of user roles
            metadata: Optional metadata for the user
        """
        self.id = user_id
        self.username = username
        self.password_hash = password_hash
        self.roles = roles
        self.metadata = metadata or {}
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self.last_login = None
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary.
        
        Returns:
            Dictionary representation of the user
        """
        return {
            'id': self.id,
            'username': self.username,
            'password_hash': self.password_hash,
            'roles': self.roles,
            'metadata': self.metadata,
            'created': self.created,
            'modified': self.modified,
            'last_login': self.last_login,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary.
        
        Args:
            data: Dictionary representation of the user
            
        Returns:
            User instance
        """
        user = cls(
            user_id=data['id'],
            username=data['username'],
            password_hash=data['password_hash'],
            roles=data['roles'],
            metadata=data.get('metadata')
        )
        user.created = data['created']
        user.modified = data['modified']
        user.last_login = data.get('last_login')
        user.is_active = data.get('is_active', True)
        return user
    
    def update_password(self, password_hash: str) -> None:
        """Update user's password.
        
        Args:
            password_hash: New password hash
        """
        self.password_hash = password_hash
        self.modified = datetime.now().isoformat()
    
    def update_roles(self, roles: List[str]) -> None:
        """Update user's roles.
        
        Args:
            roles: New list of roles
        """
        self.roles = roles
        self.modified = datetime.now().isoformat()
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update user's metadata.
        
        Args:
            metadata: New metadata
        """
        self.metadata.update(metadata)
        self.modified = datetime.now().isoformat()
    
    def record_login(self) -> None:
        """Record user's last login."""
        self.last_login = datetime.now().isoformat()
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.is_active = False
        self.modified = datetime.now().isoformat()
    
    def activate(self) -> None:
        """Activate user account."""
        self.is_active = True
        self.modified = datetime.now().isoformat()

class SecurityManager:
    """Manages security operations."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the security manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self._users = {}
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize security storage."""
        try:
            # Create storage directory
            storage_dir = Path(self.config.get('storage_dir', 'storage'))
            security_dir = storage_dir / 'security'
            security_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing users
            self._load_users(security_dir)
            
            # Create default admin user if no users exist
            if not self._users:
                self.create_user(
                    'admin',
                    'admin',
                    ['admin'],
                    {'description': 'Default administrator'}
                )
            
            logger.info("Initialized security storage")
        except Exception as e:
            logger.error(f"Failed to initialize security storage: {e}")
            raise
    
    def _load_users(self, security_dir: Path) -> None:
        """Load users from storage.
        
        Args:
            security_dir: Directory containing user files
        """
        try:
            for user_file in security_dir.glob('*.json'):
                try:
                    with open(user_file, 'r') as f:
                        data = json.load(f)
                        user = User.from_dict(data)
                        self._users[user.id] = user
                except Exception as e:
                    logger.error(f"Failed to load user from {user_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
    
    def create_user(self, username: str,
                   password: str,
                   roles: List[str],
                   metadata: Optional[Dict[str, Any]] = None) -> User:
        """Create a new user.
        
        Args:
            username: Username
            password: Password
            roles: List of roles
            metadata: Optional metadata
            
        Returns:
            Created user
            
        Raises:
            ValueError: If username already exists
        """
        try:
            # Check if username exists
            if any(user.username == username for user in self._users.values()):
                raise ValueError(f"Username {username} already exists")
            
            # Generate user ID
            user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Hash password
            password_hash = self._hash_password(password)
            
            # Create user
            user = User(user_id, username, password_hash, roles, metadata)
            self._users[user.id] = user
            
            # Save to storage
            self._save_user(user)
            
            logger.info(f"Created user {user_id}")
            return user
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User instance or None if not found
        """
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username.
        
        Args:
            username: Username
            
        Returns:
            User instance or None if not found
        """
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def update_user(self, user_id: str,
                   password: Optional[str] = None,
                   roles: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a user.
        
        Args:
            user_id: User ID
            password: Optional new password
            roles: Optional new roles
            metadata: Optional new metadata
            
        Returns:
            True if user was updated, False otherwise
        """
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            
            if password is not None:
                user.update_password(self._hash_password(password))
            
            if roles is not None:
                user.update_roles(roles)
            
            if metadata is not None:
                user.update_metadata(metadata)
            
            # Save to storage
            self._save_user(user)
            
            logger.info(f"Updated user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user was deleted, False otherwise
        """
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            
            # Remove from memory
            del self._users[user_id]
            
            # Remove from storage
            storage_dir = Path(self.config.get('storage_dir', 'storage'))
            user_file = storage_dir / 'security' / f"{user_id}.json"
            if user_file.exists():
                user_file.unlink()
            
            logger.info(f"Deleted user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token if authentication successful, None otherwise
        """
        try:
            user = self.get_user_by_username(username)
            if not user or not user.is_active:
                return None
            
            if not self._verify_password(password, user.password_hash):
                return None
            
            # Record login
            user.record_login()
            self._save_user(user)
            
            # Generate token
            return self._generate_token(user)
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            User instance if token is valid, None otherwise
        """
        try:
            # Verify token
            payload = self._verify_token(token)
            if not payload:
                return None
            
            # Get user
            user = self.get_user(payload['user_id'])
            if not user or not user.is_active:
                return None
            
            return user
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    def authorize(self, user: User, required_roles: List[str]) -> bool:
        """Authorize a user.
        
        Args:
            user: User to authorize
            required_roles: List of required roles
            
        Returns:
            True if user is authorized, False otherwise
        """
        try:
            return any(role in user.roles for role in required_roles)
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            return False
    
    def _save_user(self, user: User) -> None:
        """Save user to storage.
        
        Args:
            user: User instance to save
        """
        try:
            storage_dir = Path(self.config.get('storage_dir', 'storage'))
            user_file = storage_dir / 'security' / f"{user.id}.json"
            
            with open(user_file, 'w') as f:
                json.dump(user.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user {user.id}: {e}")
            raise
    
    def _hash_password(self, password: str) -> str:
        """Hash a password.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        salt = secrets.token_hex(16)
        hash_obj = hashlib.sha256()
        hash_obj.update(f"{password}{salt}".encode())
        return f"{salt}:{hash_obj.hexdigest()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password.
        
        Args:
            password: Password to verify
            password_hash: Hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            salt, hash_value = password_hash.split(':')
            hash_obj = hashlib.sha256()
            hash_obj.update(f"{password}{salt}".encode())
            return hash_obj.hexdigest() == hash_value
        except Exception:
            return False
    
    def _generate_token(self, user: User) -> str:
        """Generate a JWT token.
        
        Args:
            user: User to generate token for
            
        Returns:
            JWT token
        """
        try:
            payload = {
                'user_id': user.id,
                'username': user.username,
                'roles': user.roles,
                'exp': datetime.utcnow() + timedelta(days=1)
            }
            
            secret = self.config.get('jwt_secret', 'your-secret-key')
            return jwt.encode(payload, secret, algorithm='HS256')
        except Exception as e:
            logger.error(f"Failed to generate token: {e}")
            raise
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            secret = self.config.get('jwt_secret', 'your-secret-key')
            return jwt.decode(token, secret, algorithms=['HS256'])
        except Exception:
            return None
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get statistics about users.
        
        Returns:
            Dictionary containing user statistics
        """
        try:
            return {
                'total_users': len(self._users),
                'active_users': len([u for u in self._users.values() if u.is_active]),
                'roles': {
                    role: len([u for u in self._users.values() if role in u.roles])
                    for role in set(role for u in self._users.values() for role in u.roles)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {} 