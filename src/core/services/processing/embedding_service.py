"""
Embedding service for the LangGraph project.
This service provides embedding management capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class EmbeddingService:
    """Service for managing embeddings in the LangGraph system."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.embeddings: Dict[str, Embeddings] = {}
        self.vectorstores: Dict[str, FAISS] = {}
        self.history: List[Dict[str, Any]] = []
    
    def create_embeddings(
        self,
        name: str,
        model_name: str = "text-embedding-ada-002",
        **kwargs: Any
    ) -> Embeddings:
        """
        Create new embeddings.
        
        Args:
            name: Embeddings name
            model_name: Model name (default: text-embedding-ada-002)
            **kwargs: Additional arguments for the embeddings
            
        Returns:
            Embeddings: Created embeddings
        """
        embeddings = OpenAIEmbeddings(
            model=model_name,
            **kwargs
        )
        self.embeddings[name] = embeddings
        return embeddings
    
    def get_embeddings(self, name: str) -> Optional[Embeddings]:
        """
        Get embeddings by name.
        
        Args:
            name: Embeddings name
            
        Returns:
            Optional[Embeddings]: Embeddings if found, None otherwise
        """
        return self.embeddings.get(name)
    
    def create_vectorstore(
        self,
        name: str,
        embeddings_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> FAISS:
        """
        Create a new vector store.
        
        Args:
            name: Vector store name
            embeddings_name: Name of embeddings to use
            texts: Texts to store
            metadatas: Optional metadata for each text
            **kwargs: Additional arguments for the vector store
            
        Returns:
            FAISS: Created vector store
        """
        embeddings = self.get_embeddings(embeddings_name)
        if not embeddings:
            raise ValueError(f"Embeddings not found: {embeddings_name}")
        
        try:
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                **kwargs
            )
            self.vectorstores[name] = vectorstore
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'create',
                'text_count': len(texts)
            })
            
            return vectorstore
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'create',
                'error': str(e)
            })
            raise
    
    def get_vectorstore(self, name: str) -> Optional[FAISS]:
        """
        Get a vector store by name.
        
        Args:
            name: Vector store name
            
        Returns:
            Optional[FAISS]: Vector store if found, None otherwise
        """
        return self.vectorstores.get(name)
    
    def add_texts(
        self,
        name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> None:
        """
        Add texts to a vector store.
        
        Args:
            name: Vector store name
            texts: Texts to add
            metadatas: Optional metadata for each text
            **kwargs: Additional arguments for adding texts
        """
        vectorstore = self.get_vectorstore(name)
        if not vectorstore:
            raise ValueError(f"Vector store not found: {name}")
        
        try:
            vectorstore.add_texts(texts=texts, metadatas=metadatas, **kwargs)
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'add_texts',
                'text_count': len(texts)
            })
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'add_texts',
                'error': str(e)
            })
            raise
    
    def similarity_search(
        self,
        name: str,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts.
        
        Args:
            name: Vector store name
            query: Search query
            k: Number of results to return (default: 4)
            **kwargs: Additional arguments for similarity search
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        vectorstore = self.get_vectorstore(name)
        if not vectorstore:
            raise ValueError(f"Vector store not found: {name}")
        
        try:
            results = vectorstore.similarity_search(query, k=k, **kwargs)
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'similarity_search',
                'query': query,
                'k': k,
                'result_count': len(results)
            })
            
            return results
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'similarity_search',
                'query': query,
                'error': str(e)
            })
            raise
    
    def similarity_search_with_score(
        self,
        name: str,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[tuple[Dict[str, Any], float]]:
        """
        Search for similar texts with scores.
        
        Args:
            name: Vector store name
            query: Search query
            k: Number of results to return (default: 4)
            **kwargs: Additional arguments for similarity search
            
        Returns:
            List[tuple[Dict[str, Any], float]]: Search results with scores
        """
        vectorstore = self.get_vectorstore(name)
        if not vectorstore:
            raise ValueError(f"Vector store not found: {name}")
        
        try:
            results = vectorstore.similarity_search_with_score(query, k=k, **kwargs)
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'similarity_search_with_score',
                'query': query,
                'k': k,
                'result_count': len(results)
            })
            
            return results
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'vectorstore': name,
                'action': 'similarity_search_with_score',
                'query': query,
                'error': str(e)
            })
            raise
    
    def get_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the embedding service usage history.
        
        Args:
            name: Optional vector store name to filter history
            
        Returns:
            List[Dict[str, Any]]: Embedding service usage history
        """
        if name:
            return [entry for entry in self.history if entry.get('vectorstore') == name]
        return self.history
    
    def clear_history(self, name: Optional[str] = None) -> None:
        """
        Clear the embedding service usage history.
        
        Args:
            name: Optional vector store name to clear history for
        """
        if name:
            self.history = [entry for entry in self.history if entry.get('vectorstore') != name]
        else:
            self.history = []
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset embeddings and vector stores.
        
        Args:
            name: Optional vector store name to reset
        """
        if name:
            self.embeddings.pop(name, None)
            self.vectorstores.pop(name, None)
        else:
            self.embeddings = {}
            self.vectorstores = {}
            self.history = []

__all__ = ['EmbeddingService'] 