"""
Storage utilities for the LangGraph project.
These utilities provide storage capabilities integrated with LangChain.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import yaml
import pickle
import sqlite3
from datetime import datetime
from langchain.schema import Document
from langchain.vectorstores import VectorStore, FAISS
from langchain.embeddings.base import Embeddings

class StorageManager:
    """Manager for handling storage operations in the LangGraph project."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        embeddings: Optional[Embeddings] = None
    ):
        """
        Initialize the storage manager.
        
        Args:
            base_dir: Base directory for storage
            embeddings: Optional embeddings model
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        
        # Create subdirectories
        self.doc_dir = self.base_dir / "documents"
        self.vector_dir = self.base_dir / "vectors"
        self.meta_dir = self.base_dir / "metadata"
        
        for dir_path in [self.doc_dir, self.vector_dir, self.meta_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_document(self, document: Document, doc_id: str) -> None:
        """
        Save a document to storage.
        
        Args:
            document: Document to save
            doc_id: Unique identifier for the document
        """
        # Save document content
        doc_path = self.doc_dir / f"{doc_id}.json"
        with open(doc_path, "w") as f:
            json.dump({
                "content": document.page_content,
                "metadata": document.metadata
            }, f, indent=2)
        
        # Save metadata
        meta_path = self.meta_dir / f"{doc_id}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }, f, indent=2)
    
    def load_document(self, doc_id: str) -> Document:
        """
        Load a document from storage.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            Document: Loaded document
        """
        doc_path = self.doc_dir / f"{doc_id}.json"
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_id}")
        
        with open(doc_path) as f:
            data = json.load(f)
            return Document(
                page_content=data["content"],
                metadata=data["metadata"]
            )
    
    def save_vector_store(
        self,
        vector_store: VectorStore,
        store_id: str
    ) -> None:
        """
        Save a vector store to storage.
        
        Args:
            vector_store: Vector store to save
            store_id: Unique identifier for the vector store
        """
        if not isinstance(vector_store, FAISS):
            raise ValueError("Only FAISS vector stores are supported")
        
        store_path = self.vector_dir / store_id
        vector_store.save_local(str(store_path))
    
    def load_vector_store(
        self,
        store_id: str
    ) -> VectorStore:
        """
        Load a vector store from storage.
        
        Args:
            store_id: Unique identifier for the vector store
            
        Returns:
            VectorStore: Loaded vector store
        """
        if not self.embeddings:
            raise ValueError("Embeddings model not initialized")
        
        store_path = self.vector_dir / store_id
        if not store_path.exists():
            raise FileNotFoundError(f"Vector store not found: {store_id}")
        
        return FAISS.load_local(str(store_path), self.embeddings)
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        meta_id: str
    ) -> None:
        """
        Save metadata to storage.
        
        Args:
            metadata: Metadata to save
            meta_id: Unique identifier for the metadata
        """
        meta_path = self.meta_dir / f"{meta_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, meta_id: str) -> Dict[str, Any]:
        """
        Load metadata from storage.
        
        Args:
            meta_id: Unique identifier for the metadata
            
        Returns:
            Dict[str, Any]: Loaded metadata
        """
        meta_path = self.meta_dir / f"{meta_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_id}")
        
        with open(meta_path) as f:
            return json.load(f)
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from storage.
        
        Args:
            doc_id: Unique identifier for the document
        """
        doc_path = self.doc_dir / f"{doc_id}.json"
        meta_path = self.meta_dir / f"{doc_id}.json"
        
        if doc_path.exists():
            doc_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
    
    def delete_vector_store(self, store_id: str) -> None:
        """
        Delete a vector store from storage.
        
        Args:
            store_id: Unique identifier for the vector store
        """
        store_path = self.vector_dir / store_id
        if store_path.exists():
            import shutil
            shutil.rmtree(store_path)
    
    def delete_metadata(self, meta_id: str) -> None:
        """
        Delete metadata from storage.
        
        Args:
            meta_id: Unique identifier for the metadata
        """
        meta_path = self.meta_dir / f"{meta_id}.json"
        if meta_path.exists():
            meta_path.unlink()
    
    def list_documents(self) -> List[str]:
        """
        List all document IDs in storage.
        
        Returns:
            List[str]: List of document IDs
        """
        return [f.stem for f in self.doc_dir.glob("*.json")]
    
    def list_vector_stores(self) -> List[str]:
        """
        List all vector store IDs in storage.
        
        Returns:
            List[str]: List of vector store IDs
        """
        return [f.name for f in self.vector_dir.iterdir() if f.is_dir()]
    
    def list_metadata(self) -> List[str]:
        """
        List all metadata IDs in storage.
        
        Returns:
            List[str]: List of metadata IDs
        """
        return [f.stem for f in self.meta_dir.glob("*.json")]
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Returns:
            Dict[str, Any]: Document statistics
        """
        return {
            "total_documents": len(self.list_documents()),
            "total_vector_stores": len(self.list_vector_stores()),
            "total_metadata": len(self.list_metadata())
        }
    
    def backup(self, backup_dir: Union[str, Path]) -> None:
        """
        Create a backup of the storage.
        
        Args:
            backup_dir: Directory to store the backup
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copytree(self.base_dir, backup_dir / self.base_dir.name)
    
    def restore(self, backup_dir: Union[str, Path]) -> None:
        """
        Restore storage from a backup.
        
        Args:
            backup_dir: Directory containing the backup
        """
        backup_dir = Path(backup_dir)
        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup not found: {backup_dir}")
        
        import shutil
        shutil.rmtree(self.base_dir)
        shutil.copytree(backup_dir / self.base_dir.name, self.base_dir)

__all__ = ['StorageManager'] 