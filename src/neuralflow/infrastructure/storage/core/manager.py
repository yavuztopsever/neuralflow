"""
Storage manager for coordinating all storage operations in NeuralFlow.
"""

import logging
from typing import Dict, Any, Optional, Union, Type, TypeVar, Generic
from pathlib import Path
from datetime import datetime

from .base import BaseStorage, StorageConfig
from .providers.base import BaseStorageProvider
from .database.providers.base import BaseDatabaseProvider
from .vector.base import BaseVectorStore
from .persistent.document.retriever import DocumentRetriever
from .persistent.note.manager import NoteStoreManager
from .persistent.state.manager import StateManager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class StorageManager(Generic[T]):
    """Manages all storage operations across different storage types."""
    
    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        """Initialize the storage manager.
        
        Args:
            root_dir: Optional root directory for all storage operations
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd() / "storage"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage providers
        self._providers: Dict[str, BaseStorage] = {}
        self._initialize_storage_providers()
    
    def _initialize_storage_providers(self) -> None:
        """Initialize all storage providers."""
        try:
            # Initialize file storage
            file_config = StorageConfig(
                storage_id="file_storage",
                storage_type="file",
                root_dir=self.root_dir / "files"
            )
            self._providers["file"] = BaseStorageProvider("file", file_config)
            
            # Initialize database storage
            db_config = StorageConfig(
                storage_id="db_storage",
                storage_type="database",
                root_dir=self.root_dir / "db"
            )
            self._providers["database"] = BaseDatabaseProvider("database", db_config)
            
            # Initialize vector storage
            vector_config = StorageConfig(
                storage_id="vector_storage",
                storage_type="vector",
                root_dir=self.root_dir / "vectors"
            )
            self._providers["vector"] = BaseVectorStore(vector_config)
            
            # Initialize document storage
            doc_config = StorageConfig(
                storage_id="document_storage",
                storage_type="document",
                root_dir=self.root_dir / "documents"
            )
            self._providers["document"] = DocumentRetriever(str(doc_config.root_dir))
            
            # Initialize note storage
            note_config = StorageConfig(
                storage_id="note_storage",
                storage_type="note",
                root_dir=self.root_dir / "notes"
            )
            self._providers["note"] = NoteStoreManager(note_config)
            
            # Initialize state storage
            state_config = StorageConfig(
                storage_id="state_storage",
                storage_type="state",
                root_dir=self.root_dir / "states"
            )
            self._providers["state"] = StateManager(state_config)
            
            logger.info("Initialized all storage providers")
        except Exception as e:
            logger.error(f"Failed to initialize storage providers: {e}")
            raise
    
    def get_provider(self, provider_type: str) -> Optional[BaseStorage]:
        """Get a storage provider by type.
        
        Args:
            provider_type: Type of provider to get
            
        Returns:
            Storage provider or None if not found
        """
        return self._providers.get(provider_type)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about all storage providers.
        
        Returns:
            Dictionary containing storage information
        """
        return {
            'root_dir': str(self.root_dir),
            'providers': {
                name: provider.get_storage_info()
                for name, provider in self._providers.items()
            }
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics for all storage providers.
        
        Returns:
            Dictionary containing storage statistics
        """
        try:
            stats = {
                'total_providers': len(self._providers),
                'providers': {
                    name: provider.get_storage_stats()
                    for name, provider in self._providers.items()
                }
            }
            
            # Calculate total disk usage
            total_size = 0
            for provider_stats in stats['providers'].values():
                if 'disk_usage' in provider_stats:
                    total_size += provider_stats['disk_usage']
            stats['total_disk_usage'] = total_size
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def cleanup(self) -> None:
        """Clean up all storage providers."""
        for provider in self._providers.values():
            try:
                provider.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup provider {provider.config.id}: {e}")
    
    def __del__(self):
        """Clean up resources on deletion."""
        self.cleanup() 