"""
Unit tests for the context handler component.
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from typing import Dict, Any, List

from graph.context_handler import ContextHandler


@pytest.mark.asyncio
async def test_gather_context_basic(
    mock_memory_manager, 
    mock_vector_search, 
    mock_document_handler,
    test_config,
    sample_user_query
):
    """Test that context gathering combines data from all sources correctly."""
    context_handler = ContextHandler(
        memory_manager=mock_memory_manager,
        vector_search=mock_vector_search,
        document_handler=mock_document_handler,
        config=test_config
    )
    
    # Set up mock returns
    mock_memory_manager.get_from_memory.return_value = {
        "short_term": ["Previous message about context"],
        "mid_term": ["Older message about workflows"],
        "long_term": []
    }
    
    mock_vector_search.search.return_value = [
        {"text": "LangGraph enables stateful workflows", "score": 0.95}
    ]
    
    # Execute function under test
    context = await context_handler.gather_context(sample_user_query)
    
    # Assertions
    assert "memory" in context
    assert "vector_results" in context
    assert len(context["memory"]["short_term"]) == 1
    assert len(context["memory"]["mid_term"]) == 1
    assert len(context["vector_results"]) == 1
    
    # Verify interactions
    mock_memory_manager.get_from_memory.assert_called_once()
    mock_vector_search.search.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_context_sufficiency_with_sufficient_context(
    mock_model_manager,
    test_config,
    sample_context
):
    """Test context sufficiency evaluation with sufficient context."""
    # Create context handler with mocked dependencies
    context_handler = ContextHandler(
        memory_manager=MagicMock(),
        vector_search=MagicMock(),
        document_handler=MagicMock(),
        config=test_config
    )
    
    # Patch the model manager's method
    context_handler.model_manager = mock_model_manager
    mock_model_manager.generate_response.return_value = json.dumps({"is_sufficient": True, "confidence": 0.85})
    
    # Execute function under test
    result = await context_handler.evaluate_context_sufficiency(
        query=sample_context["query"],
        context=sample_context
    )
    
    # Assertions
    assert result is True
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_context_sufficiency_with_insufficient_context(
    mock_model_manager,
    test_config,
    sample_context
):
    """Test context sufficiency evaluation with insufficient context."""
    # Create context handler with mocked dependencies
    context_handler = ContextHandler(
        memory_manager=MagicMock(),
        vector_search=MagicMock(),
        document_handler=MagicMock(),
        config=test_config
    )
    
    # Patch the model manager's method
    context_handler.model_manager = mock_model_manager
    mock_model_manager.generate_response.return_value = json.dumps({"is_sufficient": False, "confidence": 0.35})
    
    # Execute function under test
    result = await context_handler.evaluate_context_sufficiency(
        query=sample_context["query"],
        context=sample_context
    )
    
    # Assertions
    assert result is False
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_process_document_references(
    mock_document_handler,
    test_config
):
    """Test processing of document references in context handler."""
    # Create context handler with mocked dependencies
    context_handler = ContextHandler(
        memory_manager=MagicMock(),
        vector_search=MagicMock(),
        document_handler=mock_document_handler,
        config=test_config
    )
    
    document_refs = [
        {"id": "doc1", "path": "/test/path/doc1.txt"},
        {"id": "doc2", "path": "/test/path/doc2.pdf"}
    ]
    
    mock_document_handler.process_document.side_effect = [
        {"id": "doc1", "chunks": ["Doc1 chunk1", "Doc1 chunk2"]},
        {"id": "doc2", "chunks": ["Doc2 chunk1", "Doc2 chunk2"]}
    ]
    
    # Execute function under test
    processed_docs = await context_handler.process_document_references(document_refs)
    
    # Assertions
    assert len(processed_docs) == 2
    assert "chunks" in processed_docs[0]
    assert "chunks" in processed_docs[1]
    assert mock_document_handler.process_document.call_count == 2


@pytest.mark.asyncio
async def test_extract_knowledge_graph(
    test_config,
    sample_context
):
    """Test extraction of knowledge graph information."""
    # Create context handler with mocked dependencies
    context_handler = ContextHandler(
        memory_manager=MagicMock(),
        vector_search=MagicMock(),
        document_handler=MagicMock(),
        config=test_config
    )
    
    # Mock the graph store read
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            sample_context["knowledge_graph"]
        )
        
        # Execute function under test
        graph_data = await context_handler.extract_knowledge_graph("langgraph")
        
        # Assertions
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert len(graph_data["nodes"]) > 0