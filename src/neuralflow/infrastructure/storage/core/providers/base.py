"""
Base provider interfaces for storage providers.
This module provides base classes for storage provider implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union, BinaryIO
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class StorageConfig:
    """Configuration for storage providers."""
    
    def __init__(self,
                 root_dir: Union[str, Path],
                 **kwargs):
        """Initialize the configuration.
        
        Args:
            root_dir: Root directory for storage
            **kwargs: Additional configuration parameters
        """
        self.root_dir = Path(root_dir)
        self.extra_params = kwargs

class StorageEntry:
    """Storage entry with metadata."""
    
    def __init__(self,
                 path: Union[str, Path],
                 size: int,
                 content_type: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize the storage entry.
        
        Args:
            path: Entry path
            size: Entry size in bytes
            content_type: Optional content type
            metadata: Optional metadata dictionary
        """
        self.path = Path(path)
        self.size = size
        self.content_type = content_type
        self.metadata = metadata or {}
        self.created = datetime.now()
        self.modified = self.created
        self.accessed = self.created
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update entry metadata.
        
        Args:
            metadata: New metadata dictionary
        """
        self.metadata.update(metadata)
        self.modified = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary.
        
        Returns:
            Dictionary representation of the entry
        """
        return {
            'path': str(self.path),
            'size': self.size,
            'content_type': self.content_type,
            'metadata': self.metadata,
            'created': self.created.isoformat(),
            'modified': self.modified.isoformat(),
            'accessed': self.accessed.isoformat()
        }

class BaseStorageProvider(ABC):
    """Base class for storage providers."""
    
    def __init__(self, provider_id: str,
                 config: StorageConfig,
                 **kwargs):
        """Initialize the provider.
        
        Args:
            provider_id: Unique identifier for the provider
            config: Storage provider configuration
            **kwargs: Additional initialization parameters
        """
        self.id = provider_id
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._kwargs = kwargs
        self._entries: Dict[str, StorageEntry] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass
    
    @abstractmethod
    def put_file(self,
                path: Union[str, Path],
                file_obj: BinaryIO,
                content_type: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put a file into storage.
        
        Args:
            path: Storage path
            file_obj: File object to store
            content_type: Optional content type
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_file(self,
                path: Union[str, Path]) -> Optional[BinaryIO]:
        """Get a file from infrastructure.storage.
        
        Args:
            path: Storage path
            
        Returns:
            File object or None if not found
        """
        pass
    
    @abstractmethod
    def delete_file(self,
                   path: Union[str, Path]) -> bool:
        """Delete a file from infrastructure.storage.
        
        Args:
            path: Storage path
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_files(self,
                  prefix: Optional[Union[str, Path]] = None) -> List[StorageEntry]:
        """List files in storage.
        
        Args:
            prefix: Optional path prefix to filter by
            
        Returns:
            List of storage entries
        """
        pass
    
    @abstractmethod
    def get_file_info(self,
                     path: Union[str, Path]) -> Optional[StorageEntry]:
        """Get information about a file.
        
        Args:
            path: Storage path
            
        Returns:
            Storage entry or None if not found
        """
        pass
    
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
                'root_dir': str(self.config.root_dir),
                'extra_params': self.config.extra_params
            },
            'stats': {
                'entries': len(self._entries),
                'total_size': sum(e.size for e in self._entries.values())
            }
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary containing provider statistics
        """
        try:
            return {
                'total_entries': len(self._entries),
                'total_size': sum(e.size for e in self._entries.values()),
                'content_types': {
                    ct: len([e for e in self._entries.values() if e.content_type == ct])
                    for ct in set(e.content_type for e in self._entries.values() if e.content_type)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {} 