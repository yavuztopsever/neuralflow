#!/usr/bin/env python3
"""
Unit tests for the response generation component.
Tests the core functionality of generating responses based on context and execution results.
"""

import pytest
import asyncio
from typing import Dict, Any

from models.gguf_wrapper.mock_llm import MockLLM
from graph.response_generation import ResponseGenerator

class TestResponseGenerator:
    """Test suite for the ResponseGenerator class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockLLM()
    
    @pytest.fixture
    def response_generator(self, mock_llm):
        """Create a response generator instance for testing."""
        return ResponseGenerator(
            model="mock",
            temperature=0.7,
            model_instance=mock_llm
        )
    
    @pytest.mark.asyncio
    async def test_basic_response_generation(self, response_generator):
        """Test basic response generation with a simple query."""
        context = {"user_query": "What is LangGraph?"}
        execution_result = {}
        
        response = await response_generator.generate(execution_result, context)
        
        assert response is not None
        assert len(response) > 0
        assert "langgraph" in response.lower()
    
    @pytest.mark.asyncio
    async def test_response_with_context(self, response_generator):
        """Test response generation with additional context."""
        context = {
            "user_query": "How does memory work?",
            "memory_context": "The system uses hierarchical memory stores"
        }
        execution_result = {
            "memory_operation": "retrieve",
            "memory_type": "short_term"
        }
        
        response = await response_generator.generate(execution_result, context)
        
        assert response is not None
        assert len(response) > 0
        assert "memory" in response.lower()
    
    @pytest.mark.asyncio
    async def test_response_with_error_handling(self, response_generator):
        """Test response generation with error handling."""
        context = {"user_query": "Invalid query"}
        execution_result = {"error": "Test error"}
        
        response = await response_generator.generate(execution_result, context)
        
        assert response is not None
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_response_with_empty_context(self, response_generator):
        """Test response generation with empty context."""
        context = {}
        execution_result = {}
        
        response = await response_generator.generate(execution_result, context)
        
        assert response is not None
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_response_with_sentiment_analysis(self, response_generator):
        """Test response generation with sentiment analysis."""
        context = {
            "user_query": "I love this system!",
            "sentiment": "positive"
        }
        execution_result = {
            "sentiment_score": 0.9,
            "sentiment_label": "POSITIVE"
        }
        
        response = await response_generator.generate(execution_result, context)
        
        assert response is not None
        assert len(response) > 0
        assert any(word in response.lower() for word in ["great", "excellent", "wonderful"]) 