"""
Authentication service for the LangGraph project.
This service provides authentication and authorization capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import jwt
import bcrypt
import secrets
import hashlib
import os
from pathlib import Path
from fastapi import HTTPException, status
from ...services.base_service import BaseService

class User(BaseModel):
    """User model."""
    username: str
    email: str
    hashed_password: str
    roles: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

class AuthService(BaseService[User]):
    """Service for managing authentication in the LangGraph system."""
    
    def __init__(self, secret_key: str, storage_dir: Optional[str] = None):
        """
        Initialize the authentication service.
        
        Args:
            secret_key: Secret key for JWT token generation
            storage_dir: Optional directory for storing user data
        """
        super().__init__()
        self.secret_key = secret_key
        self.users: Dict[str, User] = {}
        self.storage_dir = Path(storage_dir or os.path.join(os.getcwd(), "storage", "auth"))
        self._ensure_storage()
        self._load_users()
    
    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_users(self) -> None:
        """Load users from storage."""
        try:
            for user_file in self.storage_dir.glob("*.json"):
                try:
                    with open(user_file, "r") as f:
                        data = json.load(f)
                        user = User(**data)
                        self.users[user.username] = user
                except Exception as e:
                    self.logger.error(f"Failed to load user from {user_file}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load users: {e}")
    
    def _save_user(self, user: User) -> None:
        """
        Save user to storage.
        
        Args:
            user: User to save
        """
        try:
            file_path = self.storage_dir / f"{user.username}.json"
            with open(file_path, "w") as f:
                json.dump(user.dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save user: {e}")
            raise
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None,
        is_superuser: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            roles: Optional list of roles
            is_superuser: Whether the user is a superuser
            metadata: Optional metadata
            
        Returns:
            User: Created user
        """
        if username in self.users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if email in [user.email for user in self.users.values()]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        try:
            hashed_password = bcrypt.hashpw(
                password.encode(),
                bcrypt.gensalt()
            ).decode()
            
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                roles=roles or [],
                is_superuser=is_superuser,
                metadata=metadata
            )
            self.users[username] = user
            
            # Save to storage
            self._save_user(user)
            
            # Record in history
            self.record_history(
                "create_user",
                details={
                    "username": username,
                    "email": email,
                    "roles": roles,
                    "is_superuser": is_superuser
                }
            )
            
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            raise
    
    def get_user(self, username: str) -> Optional[User]:
        """
        Get a user by username.
        
        Args:
            username: Username
            
        Returns:
            Optional[User]: User if found, None otherwise
        """
        return self.users.get(username)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email.
        
        Args:
            email: Email address
            
        Returns:
            Optional[User]: User if found, None otherwise
        """
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def update_user(
        self,
        username: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        roles: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        is_superuser: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[User]:
        """
        Update a user.
        
        Args:
            username: Username
            email: Optional new email
            password: Optional new password
            roles: Optional new roles
            is_active: Optional new active status
            is_superuser: Optional new superuser status
            metadata: Optional new metadata
            
        Returns:
            Optional[User]: Updated user if found
        """
        user = self.get_user(username)
        if not user:
            return None
        
        try:
            if email is not None:
                user.email = email
            if password is not None:
                user.hashed_password = bcrypt.hashpw(
                    password.encode(),
                    bcrypt.gensalt()
                ).decode()
            if roles is not None:
                user.roles = roles
            if is_active is not None:
                user.is_active = is_active
            if is_superuser is not None:
                user.is_superuser = is_superuser
            if metadata is not None:
                user.metadata = metadata
            
            # Save to storage
            self._save_user(user)
            
            # Record in history
            self.record_history(
                "update_user",
                details={
                    "username": username,
                    "email": email,
                    "roles": roles,
                    "is_active": is_active,
                    "is_superuser": is_superuser
                }
            )
            
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to update user: {e}")
            raise
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user.
        
        Args:
            username: Username
            
        Returns:
            bool: True if user was deleted
        """
        if username not in self.users:
            return False
        
        try:
            # Delete from storage
            file_path = self.storage_dir / f"{username}.json"
            if file_path.exists():
                file_path.unlink()
            
            # Remove from users
            del self.users[username]
            
            # Record in history
            self.record_history(
                "delete_user",
                details={"username": username}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete user: {e}")
            raise
    
    def authenticate_user(self, username: str, password: str) -> User:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User: Authenticated user
            
        Raises:
            HTTPException: If authentication fails
        """
        user = self.get_user(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User is inactive"
            )
        
        if not bcrypt.checkpw(
            password.encode(),
            user.hashed_password.encode()
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        try:
            user.last_login = datetime.now()
            
            # Save to storage
            self._save_user(user)
            
            # Record in history
            self.record_history(
                "authenticate_user",
                details={"username": username}
            )
            
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to authenticate user: {e}")
            raise
    
    def create_access_token(
        self,
        username: str,
        expires_delta: Optional[timedelta] = None
    ) -> Token:
        """
        Create an access token.
        
        Args:
            username: Username
            expires_delta: Optional expiration time delta
            
        Returns:
            Token: Created access token
        """
        user = self.get_user(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User is inactive"
            )
        
        try:
            if expires_delta is None:
                expires_delta = timedelta(minutes=15)
            
            expires_at = datetime.now() + expires_delta
            
            to_encode = {
                "sub": username,
                "exp": expires_at,
                "roles": user.roles,
                "is_superuser": user.is_superuser
            }
            
            access_token = jwt.encode(
                to_encode,
                self.secret_key,
                algorithm="HS256"
            )
            
            token = Token(
                access_token=access_token,
                expires_at=expires_at
            )
            
            # Record in history
            self.record_history(
                "create_access_token",
                details={
                    "username": username,
                    "expires_at": expires_at
                }
            )
            
            return token
            
        except Exception as e:
            self.logger.error(f"Failed to create access token: {e}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify an access token.
        
        Args:
            token: Access token
            
        Returns:
            Dict[str, Any]: Token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
            
            # Record in history
            self.record_history(
                "verify_token",
                details={"username": payload.get("sub")}
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            # Record error in history
            self.record_history(
                "verify_token_error",
                details={"error": "Token has expired"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            # Record error in history
            self.record_history(
                "verify_token_error",
                details={"error": "Invalid token"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authorize(self, username: str, required_roles: List[str]) -> bool:
        """
        Check if a user has required roles.
        
        Args:
            username: Username
            required_roles: List of required roles
            
        Returns:
            bool: True if user has required roles
        """
        user = self.get_user(username)
        if not user:
            return False
        
        if user.is_superuser:
            return True
        
        return all(role in user.roles for role in required_roles)
    
    def get_user_stats(self) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Returns:
            Dict[str, Any]: User statistics
        """
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "superusers": len([u for u in self.users.values() if u.is_superuser]),
            "roles": list(set(role for u in self.users.values() for role in u.roles))
        }
    
    def reset(self) -> None:
        """Reset authentication service."""
        super().reset()
        self.users = {}

__all__ = ['AuthService', 'User', 'Token'] 