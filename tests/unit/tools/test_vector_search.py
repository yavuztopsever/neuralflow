import os
import pytest
import tempfile
from unittest.mock import Mock, patch
from tools.vector_search import VectorSearch
from config.config import Config

@pytest.fixture
def temp_vector_db_dir():
    """Create a temporary directory for vector DB storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_config(temp_vector_db_dir):
    """Create a mock config with test settings."""
    config = Mock(spec=Config)
    config.VECTOR_DB_DIR = temp_vector_db_dir
    config.EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    return config

@pytest.fixture
def vector_search(mock_config):
    """Create a VectorSearch instance with mock config."""
    with patch('sentence_transformers.SentenceTransformer') as mock_embedder, \
         patch('chromadb.Client') as mock_client:
        # Mock the embedding function
        mock_embedder.return_value.encode.return_value = [[0.1] * 384]  # 384 is the dimension for all-MiniLM-L6-v2
        
        # Mock the ChromaDB client
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_collection.name = "documents"
        mock_collection.get.return_value = {"metadata": {"hnsw:space": "cosine"}}
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        vector_search = VectorSearch(config=mock_config)
        return vector_search

def test_initialization(vector_search):
    """Test proper initialization of VectorSearch."""
    assert vector_search.config is not None
    assert vector_search.client is not None
    assert vector_search.collection is not None
    assert vector_search.embedding_function is not None

def test_add_documents(vector_search):
    """Test adding documents to the vector store."""
    documents = [
        {"content": "Test document 1", "metadata": {"source": "test1"}},
        {"content": "Test document 2", "metadata": {"source": "test2"}}
    ]
    
    vector_search.add_documents(documents)
    
    # Verify that add was called with correct parameters
    vector_search.collection.add.assert_called_once()
    call_args = vector_search.collection.add.call_args[1]
    
    assert len(call_args['documents']) == 2
    assert len(call_args['metadatas']) == 2
    assert len(call_args['embeddings']) == 2
    assert len(call_args['ids']) == 2

def test_search(vector_search):
    """Test searching for similar documents."""
    query = "test query"
    n_results = 3
    
    # Mock the search results
    mock_results = {
        'documents': [['doc1', 'doc2', 'doc3']],
        'metadatas': [{'source': 'test1'}, {'source': 'test2'}, {'source': 'test3'}],
        'distances': [[0.1, 0.2, 0.3]]
    }
    vector_search.collection.query.return_value = mock_results
    
    results = vector_search.search(query, n_results)
    
    # Verify the search was performed with correct parameters
    vector_search.collection.query.assert_called_once()
    call_args = vector_search.collection.query.call_args[1]
    
    assert len(call_args['query_embeddings']) == 1
    assert call_args['n_results'] == n_results
    assert results == mock_results

def test_clear_collection(vector_search):
    """Test clearing the vector store collection."""
    vector_search.clear_collection()
    
    # Verify that delete was called
    vector_search.collection.delete.assert_called_once()
    
    # Verify that get_or_create_collection was called to recreate the collection
    vector_search.client.get_or_create_collection.assert_called()

def test_get_collection_stats(vector_search):
    """Test getting collection statistics."""
    stats = vector_search.get_collection_stats()
    
    assert isinstance(stats, dict)
    assert 'count' in stats
    assert 'name' in stats
    assert 'metadata' in stats
    assert stats['name'] == "documents"

def test_add_document(vector_search):
    """Test adding a single document."""
    doc_id = "test_doc_1"
    text = "Test document content"
    
    vector_search.add_document(doc_id, text)
    
    # Verify that add was called with correct parameters
    vector_search.collection.add.assert_called_once()
    call_args = vector_search.collection.add.call_args[1]
    
    assert len(call_args['documents']) == 1
    assert len(call_args['metadatas']) == 1
    assert len(call_args['embeddings']) == 1
    assert len(call_args['ids']) == 1
    assert call_args['ids'][0] == doc_id

def test_search_similar(vector_search):
    """Test searching for similar documents using the search_similar method."""
    query = "test query"
    top_k = 3
    
    # Mock the search results
    mock_results = {
        'documents': [['doc1', 'doc2', 'doc3']],
        'metadatas': [{'source': 'test1'}, {'source': 'test2'}, {'source': 'test3'}],
        'distances': [[0.1, 0.2, 0.3]]
    }
    vector_search.collection.query.return_value = mock_results
    
    results = vector_search.search_similar(query, top_k)
    
    # Verify the search was performed with correct parameters
    vector_search.collection.query.assert_called_once()
    call_args = vector_search.collection.query.call_args[1]
    
    assert len(call_args['query_embeddings']) == 1
    assert call_args['n_results'] == top_k
    assert results == mock_results['documents'][0]

def test_error_handling(vector_search):
    """Test error handling in various operations."""
    # Test error in add_documents
    with pytest.raises(Exception):
        vector_search.collection.add.side_effect = Exception("Test error")
        vector_search.add_documents([{"content": "test"}])
    
    # Test error in search
    with pytest.raises(Exception):
        vector_search.collection.query.side_effect = Exception("Test error")
        vector_search.search("test query")
    
    # Test error in clear_collection
    with pytest.raises(Exception):
        vector_search.collection.delete.side_effect = Exception("Test error")
        vector_search.clear_collection()

def test_embedding_function(vector_search):
    """Test the embedding function."""
    text = "Test text"
    embedding = vector_search._embed_text(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Dimension for all-MiniLM-L6-v2
    assert all(isinstance(x, float) for x in embedding)

def test_mock_embedding(vector_search):
    """Test the mock embedding function."""
    text = "Test text"
    embedding = vector_search._mock_embedding(text)
    
    assert isinstance(embedding, list)
    assert all(x == 0.0 for x in embedding)  # Mock embeddings are all zeros
    assert len(embedding) == 1536  # Default dimension 