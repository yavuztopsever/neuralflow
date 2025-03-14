"""
Unit tests for document storage and retrieval functionality.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
from src.storage.persistent.document.retriever import DocumentRetriever, DocumentConfig, Document

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "documents"
    storage_dir.mkdir()
    return storage_dir

@pytest.fixture
def document_retriever(temp_storage_dir):
    """Create a document retriever instance."""
    config = DocumentConfig(
        storage_dir=temp_storage_dir,
        vector_store_type="chroma",
        embedder_model="all-MiniLM-L6-v2"
    )
    return DocumentRetriever(config)

@pytest.fixture
def sample_document():
    """Create a sample document."""
    return Document(
        doc_id="test_doc",
        content="This is a test document.",
        metadata={"type": "test", "author": "tester"}
    )

class TestDocumentRetriever:
    """Test suite for document retriever functionality."""
    
    def test_initialization(self, document_retriever, temp_storage_dir):
        """Test document retriever initialization."""
        assert document_retriever.storage_dir == temp_storage_dir
        assert temp_storage_dir.exists()
        assert document_retriever.vector_store is not None
    
    def test_add_get_document(self, document_retriever, sample_document):
        """Test adding and getting a document."""
        # Add document
        doc_id = document_retriever.add_document(sample_document)
        assert doc_id == sample_document.id
        
        # Check file exists
        doc_file = document_retriever.storage_dir / f"{doc_id}.json"
        assert doc_file.exists()
        
        # Get document
        retrieved_doc = document_retriever.get_document(doc_id)
        assert retrieved_doc is not None
        assert retrieved_doc.id == sample_document.id
        assert retrieved_doc.content == sample_document.content
        assert retrieved_doc.metadata == sample_document.metadata
    
    def test_get_nonexistent_document(self, document_retriever):
        """Test getting a non-existent document."""
        doc = document_retriever.get_document("nonexistent")
        assert doc is None
    
    def test_update_document(self, document_retriever, sample_document):
        """Test updating a document."""
        # Add document
        document_retriever.add_document(sample_document)
        
        # Update document
        new_content = "Updated content"
        new_metadata = {"type": "test", "author": "updater"}
        success = document_retriever.update_document(
            doc_id=sample_document.id,
            content=new_content,
            metadata=new_metadata
        )
        assert success
        
        # Get updated document
        updated_doc = document_retriever.get_document(sample_document.id)
        assert updated_doc is not None
        assert updated_doc.content == new_content
        assert updated_doc.metadata == new_metadata
    
    def test_delete_document(self, document_retriever, sample_document):
        """Test deleting a document."""
        # Add document
        document_retriever.add_document(sample_document)
        
        # Delete document
        success = document_retriever.delete_document(sample_document.id)
        assert success
        
        # Check file doesn't exist
        doc_file = document_retriever.storage_dir / f"{sample_document.id}.json"
        assert not doc_file.exists()
        
        # Check document is gone
        doc = document_retriever.get_document(sample_document.id)
        assert doc is None
    
    def test_search_documents(self, document_retriever):
        """Test document search."""
        # Add multiple documents
        docs = [
            Document("doc1", "Python is a great programming language.", {"type": "tech"}),
            Document("doc2", "Python programming is fun.", {"type": "tech"}),
            Document("doc3", "Cats are cute animals.", {"type": "pets"})
        ]
        for doc in docs:
            document_retriever.add_document(doc)
        
        # Search without filter
        results = document_retriever.search_documents(
            query="Python programming",
            top_k=2
        )
        assert len(results) == 2
        assert all(doc.id in ["doc1", "doc2"] for doc in results)
        
        # Search with metadata filter
        results = document_retriever.search_documents(
            query="animals",
            metadata_filter={"type": "pets"}
        )
        assert len(results) == 1
        assert results[0].id == "doc3"
    
    def test_list_documents(self, document_retriever):
        """Test listing documents."""
        # Add multiple documents
        docs = [
            Document("doc1", "Content 1", {"type": "a"}),
            Document("doc2", "Content 2", {"type": "b"}),
            Document("doc3", "Content 3", {"type": "a"})
        ]
        for doc in docs:
            document_retriever.add_document(doc)
        
        # List all documents
        all_docs = document_retriever.list_documents()
        assert len(all_docs) == 3
        assert all(isinstance(doc, Document) for doc in all_docs)
        assert all(doc.id in ["doc1", "doc2", "doc3"] for doc in all_docs)
    
    def test_get_stats(self, document_retriever, sample_document):
        """Test getting storage statistics."""
        # Add a document
        document_retriever.add_document(sample_document)
        
        # Get stats
        stats = document_retriever.get_stats()
        assert 'vector_store' in stats
        assert 'documents' in stats
        assert stats['documents']['total'] == 1
        assert stats['documents']['size'] > 0 