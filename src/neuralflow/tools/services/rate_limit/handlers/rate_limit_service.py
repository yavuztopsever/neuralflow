"""
Rate limit service for the LangGraph project.
This service provides rate limiting capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import time
from collections import defaultdict

class RateLimit(BaseModel):
    """Rate limit model."""
    requests: int
    window: timedelta
    last_reset: datetime = Field(default_factory=datetime.now)
    current_requests: int = 0

class RateLimitService:
    """Service for managing rate limits in the LangGraph system."""
    
    def __init__(self):
        """Initialize the rate limit service."""
        self.rate_limits: Dict[str, RateLimit] = {}
        self.history: List[Dict[str, Any]] = []
    
    def create_rate_limit(
        self,
        name: str,
        requests: int,
        window: Union[int, timedelta]
    ) -> RateLimit:
        """
        Create a new rate limit.
        
        Args:
            name: Rate limit name
            requests: Number of requests allowed
            window: Time window in seconds or timedelta
            
        Returns:
            RateLimit: Created rate limit
        """
        if isinstance(window, int):
            window = timedelta(seconds=window)
        
        rate_limit = RateLimit(
            requests=requests,
            window=window
        )
        self.rate_limits[name] = rate_limit
        
        # Record in history
        self.history.append({
            'timestamp': datetime.now(),
            'action': 'create_rate_limit',
            'name': name,
            'requests': requests,
            'window': str(window)
        })
        
        return rate_limit
    
    def get_rate_limit(self, name: str) -> Optional[RateLimit]:
        """
        Get a rate limit by name.
        
        Args:
            name: Rate limit name
            
        Returns:
            Optional[RateLimit]: Rate limit if found, None otherwise
        """
        return self.rate_limits.get(name)
    
    def check_rate_limit(self, name: str) -> bool:
        """
        Check if a rate limit has been exceeded.
        
        Args:
            name: Rate limit name
            
        Returns:
            bool: True if rate limit is not exceeded, False otherwise
        """
        rate_limit = self.get_rate_limit(name)
        if not rate_limit:
            return True
        
        now = datetime.now()
        
        # Reset if window has passed
        if now - rate_limit.last_reset > rate_limit.window:
            rate_limit.current_requests = 0
            rate_limit.last_reset = now
            
            # Record in history
            self.history.append({
                'timestamp': now,
                'action': 'reset_rate_limit',
                'name': name
            })
        
        # Check if limit exceeded
        if rate_limit.current_requests >= rate_limit.requests:
            # Record in history
            self.history.append({
                'timestamp': now,
                'action': 'rate_limit_exceeded',
                'name': name,
                'current_requests': rate_limit.current_requests,
                'max_requests': rate_limit.requests
            })
            return False
        
        # Increment request count
        rate_limit.current_requests += 1
        
        # Record in history
        self.history.append({
            'timestamp': now,
            'action': 'increment_rate_limit',
            'name': name,
            'current_requests': rate_limit.current_requests
        })
        
        return True
    
    def get_remaining_requests(self, name: str) -> Optional[int]:
        """
        Get remaining requests for a rate limit.
        
        Args:
            name: Rate limit name
            
        Returns:
            Optional[int]: Remaining requests if found, None otherwise
        """
        rate_limit = self.get_rate_limit(name)
        if not rate_limit:
            return None
        
        now = datetime.now()
        
        # Reset if window has passed
        if now - rate_limit.last_reset > rate_limit.window:
            rate_limit.current_requests = 0
            rate_limit.last_reset = now
            
            # Record in history
            self.history.append({
                'timestamp': now,
                'action': 'reset_rate_limit',
                'name': name
            })
        
        return rate_limit.requests - rate_limit.current_requests
    
    def reset_rate_limit(self, name: str) -> None:
        """
        Reset a rate limit.
        
        Args:
            name: Rate limit name
        """
        rate_limit = self.get_rate_limit(name)
        if not rate_limit:
            return
        
        rate_limit.current_requests = 0
        rate_limit.last_reset = datetime.now()
        
        # Record in history
        self.history.append({
            'timestamp': datetime.now(),
            'action': 'reset_rate_limit',
            'name': name
        })
    
    def get_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the rate limit service usage history.
        
        Args:
            name: Optional rate limit name to filter history
            
        Returns:
            List[Dict[str, Any]]: Rate limit service usage history
        """
        if name:
            return [entry for entry in self.history if entry.get('name') == name]
        return self.history
    
    def clear_history(self, name: Optional[str] = None) -> None:
        """
        Clear the rate limit service usage history.
        
        Args:
            name: Optional rate limit name to clear history for
        """
        if name:
            self.history = [entry for entry in self.history if entry.get('name') != name]
        else:
            self.history = []
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset rate limits and history.
        
        Args:
            name: Optional rate limit name to reset
        """
        if name:
            self.rate_limits.pop(name, None)
        else:
            self.rate_limits = {}
            self.history = []

__all__ = ['RateLimitService', 'RateLimit'] 