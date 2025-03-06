"""
Unit tests for the response generation component.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from typing import Dict, Any, List

from graph.response_generation import ResponseGenerator


@pytest.mark.asyncio
async def test_generate_response_basic(
    mock_model_manager,
    test_config,
    sample_context,
    sample_user_query
):
    """Test basic response generation with context."""
    # Create response generator with mocked dependencies
    response_generator = ResponseGenerator(
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up mock return
    expected_response = "LangGraph is a library for building stateful, multi-actor applications with LLMs."
    mock_model_manager.generate_response.return_value = expected_response
    
    # Execute function under test
    response = await response_generator.generate_response(
        query=sample_user_query,
        context=sample_context
    )
    
    # Assertions
    assert response == expected_response
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_format_prompt_with_context(
    test_config,
    sample_context,
    sample_user_query
):
    """Test that the prompt formatter correctly includes context."""
    # Create response generator with mocked dependencies
    response_generator = ResponseGenerator(
        model_manager=MagicMock(),
        config=test_config
    )
    
    # Execute function under test
    formatted_prompt = response_generator.format_prompt(
        query=sample_user_query,
        context=sample_context
    )
    
    # Assertions
    assert sample_user_query in formatted_prompt
    assert "context" in formatted_prompt.lower()
    
    # Check for specific context elements
    assert "LangGraph is a library" in formatted_prompt


@pytest.mark.asyncio
async def test_apply_response_style(
    mock_model_manager,
    test_config
):
    """Test applying a specific response style."""
    # Create response generator with mocked dependencies
    response_generator = ResponseGenerator(
        model_manager=mock_model_manager,
        config=test_config
    )
    
    base_response = "LangGraph is a library for building LLM applications."
    style = "technical"
    
    # Set up mock return
    styled_response = "LangGraph is a framework for constructing stateful, multi-actor applications with Large Language Models (LLMs)."
    mock_model_manager.apply_style.return_value = styled_response
    
    # Execute function under test
    result = await response_generator.apply_response_style(
        response=base_response,
        style=style
    )
    
    # Assertions
    assert result == styled_response
    mock_model_manager.apply_style.assert_called_once_with(
        response=base_response,
        style=style
    )


@pytest.mark.asyncio
async def test_generate_response_with_streaming(
    mock_model_manager,
    test_config,
    sample_context,
    sample_user_query
):
    """Test response generation with streaming enabled."""
    # Create response generator with mocked dependencies
    response_generator = ResponseGenerator(
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up mock return for streaming
    stream_chunks = ["LangGraph ", "is ", "a ", "library ", "for ", "building ", "stateful ", "workflows."]
    mock_model_manager.stream_response.return_value = stream_chunks
    
    # Execute function under test with streaming
    chunks = []
    async for chunk in response_generator.stream_response(
        query=sample_user_query,
        context=sample_context
    ):
        chunks.append(chunk)
    
    # Assertions
    assert chunks == stream_chunks
    mock_model_manager.stream_response.assert_called_once()


@pytest.mark.asyncio
async def test_handle_error_response(
    mock_model_manager,
    test_config,
    sample_user_query
):
    """Test handling of error conditions in response generation."""
    # Create response generator with mocked dependencies
    response_generator = ResponseGenerator(
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up model manager to raise an exception
    mock_model_manager.generate_response.side_effect = Exception("Model error")
    
    # Execute function under test
    response = await response_generator.generate_response(
        query=sample_user_query,
        context={}
    )
    
    # Assertions - should return a fallback response
    assert "I apologize" in response
    assert "error" in response.lower()
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_prioritize_response_urgent(
    mock_model_manager,
    test_config,
    sample_context,
    sample_user_query
):
    """Test response prioritization for urgent queries."""
    # Create response generator with mocked dependencies
    response_generator = ResponseGenerator(
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set the query priority
    priority = "urgent"
    
    # Set up mock return
    priority_response = "URGENT: LangGraph is a library for building LLM applications."
    mock_model_manager.generate_response.return_value = "LangGraph is a library for building LLM applications."
    
    # Patch the prioritize method
    with patch.object(
        response_generator, 
        'prioritize_response', 
        return_value=priority_response
    ) as mock_prioritize:
        
        # Execute function under test
        response = await response_generator.generate_response(
            query=sample_user_query,
            context=sample_context,
            priority=priority
        )
        
        # Assertions
        assert response == priority_response
        mock_prioritize.assert_called_once()
        mock_model_manager.generate_response.assert_called_once()