"""
Cache management utilities specific to the LangGraph project.
These utilities handle caching that is not covered by LangChain's built-in caching.
"""

import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

T = TypeVar('T')

class CacheType(Enum):
    """Types of cache that can be used."""
    MEMORY = "memory"
    FILE = "file"
    CUSTOM = "custom"

@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    value: Any
    created_at: str
    expires_at: Optional[str]
    metadata: Dict[str, Any]

class CacheManager(Generic[T]):
    """Manages caching for the LangGraph project."""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        default_ttl: int = 3600,  # 1 hour
        max_size: int = 1000,
        cache_type: CacheType = CacheType.MEMORY
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cache_type = cache_type
        
        # Initialize cache
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.Lock()
        
        # Load existing cache if using file cache
        if cache_type == CacheType.FILE:
            self._load_cache()

    def _load_cache(self):
        """Load cache from file if using file cache."""
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                self.cache = {
                    k: CacheEntry(**v)
                    for k, v in data.items()
                }

    def _save_cache(self):
        """Save cache to file if using file cache."""
        if self.cache_type != CacheType.FILE:
            return
            
        cache_file = self.cache_dir / "cache.json"
        data = {
            k: asdict(v)
            for k, v in self.cache.items()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_key(self, key: Union[str, Any]) -> str:
        """Generate a cache key from input."""
        if isinstance(key, str):
            return key
            
        # Convert to string and hash
        key_str = str(key)
        return hashlib.md5(key_str.encode()).hexdigest()

    def set(
        self,
        key: Union[str, Any],
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set a cache value with optional TTL and metadata."""
        with self.lock:
            # Generate cache key
            cache_key = self._generate_key(key)
            
            # Calculate expiration
            now = datetime.now()
            expires_at = None
            if ttl is not None:
                expires_at = (now + timedelta(seconds=ttl)).isoformat()
            elif self.default_ttl > 0:
                expires_at = (now + timedelta(seconds=self.default_ttl)).isoformat()
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=now.isoformat(),
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            # Update cache
            self.cache[cache_key] = entry
            
            # Enforce max size
            if len(self.cache) > self.max_size:
                # Remove oldest entry
                oldest_key = min(
                    self.cache.items(),
                    key=lambda x: x[1].created_at
                )[0]
                del self.cache[oldest_key]
            
            # Save if using file cache
            self._save_cache()

    def get(self, key: Union[str, Any], default: Any = None) -> Any:
        """Get a cache value if it exists and is not expired."""
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key not in self.cache:
                return default
                
            entry = self.cache[cache_key]
            
            # Check expiration
            if entry.expires_at:
                expires_at = datetime.fromisoformat(entry.expires_at)
                if datetime.now() > expires_at:
                    del self.cache[cache_key]
                    self._save_cache()
                    return default
            
            return entry.value

    def delete(self, key: Union[str, Any]):
        """Delete a cache value."""
        with self.lock:
            cache_key = self._generate_key(key)
            if cache_key in self.cache:
                del self.cache[cache_key]
                self._save_cache()

    def clear(self):
        """Clear all cache values."""
        with self.lock:
            self.cache.clear()
            self._save_cache()

    def get_metadata(self, key: Union[str, Any]) -> Optional[Dict[str, Any]]:
        """Get metadata for a cache value."""
        with self.lock:
            cache_key = self._generate_key(key)
            if cache_key in self.cache:
                return self.cache[cache_key].metadata
            return None

    def update_metadata(
        self,
        key: Union[str, Any],
        metadata: Dict[str, Any]
    ):
        """Update metadata for a cache value."""
        with self.lock:
            cache_key = self._generate_key(key)
            if cache_key in self.cache:
                self.cache[cache_key].metadata.update(metadata)
                self._save_cache()

    def get_expiration(self, key: Union[str, Any]) -> Optional[datetime]:
        """Get expiration time for a cache value."""
        with self.lock:
            cache_key = self._generate_key(key)
            if cache_key in self.cache:
                expires_at = self.cache[cache_key].expires_at
                if expires_at:
                    return datetime.fromisoformat(expires_at)
            return None

    def set_expiration(
        self,
        key: Union[str, Any],
        ttl: int
    ):
        """Set expiration time for a cache value."""
        with self.lock:
            cache_key = self._generate_key(key)
            if cache_key in self.cache:
                now = datetime.now()
                self.cache[cache_key].expires_at = (
                    now + timedelta(seconds=ttl)
                ).isoformat()
                self._save_cache()

    def get_size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        expired = 0
        total_size = 0
        
        for entry in self.cache.values():
            if entry.expires_at:
                expires_at = datetime.fromisoformat(entry.expires_at)
                if now > expires_at:
                    expired += 1
            total_size += 1
        
        return {
            'total_entries': len(self.cache),
            'expired_entries': expired,
            'active_entries': len(self.cache) - expired,
            'max_size': self.max_size
        }

    def cleanup(self):
        """Remove expired entries from cache."""
        with self.lock:
            now = datetime.now()
            expired_keys = [
                k for k, v in self.cache.items()
                if v.expires_at and datetime.fromisoformat(v.expires_at) <= now
            ]
            
            for k in expired_keys:
                del self.cache[k]
            
            if expired_keys:
                self._save_cache()

    def cached(
        self,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Decorator for caching function results."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs) -> T:
                # Generate cache key from function and arguments
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Try to get from cache
                cached_value = self.get(key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, ttl, metadata)
                return result
                
            return wrapper
        return decorator

    def export_cache(self, filepath: Union[str, Path]):
        """Export cache to a file."""
        filepath = Path(filepath)
        data = {
            k: asdict(v)
            for k, v in self.cache.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_cache(self, filepath: Union[str, Path]):
        """Import cache from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Cache file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.cache = {
                k: CacheEntry(**v)
                for k, v in data.items()
            }
            
        self._save_cache() 