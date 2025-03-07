"""
Unit tests for response generation and handling functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.core.nodes.output import OutputNode
from src.core.nodes.reasoning import ReasoningNode
from models.gguf_wrapper.mock_llm import MockLLM
from graph.response_generation import ResponseGenerator

class TestResponseService:
    """Test suite for response service functionality."""
    
    @pytest.fixture
    def output_node(self):
        """Create an output node for testing."""
        return OutputNode("test_output", "output", {})
    
    @pytest.fixture
    def reasoning_node(self):
        """Create a reasoning node for testing."""
        return ReasoningNode("test_reasoning", "reasoning", {})
    
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
    async def test_response_assembly(self, output_node):
        """Test response assembly operations."""
        response_data = {
            "content": "Test response content",
            "citations": ["citation1", "citation2"],
            "metadata": {"key": "value"}
        }
        
        response = await output_node.assemble_response(response_data)
        assert response is not None
        assert isinstance(response, dict)
        assert "content" in response
        assert "citations" in response
        assert "metadata" in response
    
    @pytest.mark.asyncio
    async def test_response_formatting(self, output_node):
        """Test response formatting operations."""
        response_data = {
            "content": "Test response content",
            "citations": ["citation1", "citation2"],
            "metadata": {"key": "value"}
        }
        
        formatted_response = await output_node.format_response(response_data)
        assert formatted_response is not None
        assert isinstance(formatted_response, str)
        assert len(formatted_response) > 0
    
    @pytest.mark.asyncio
    async def test_response_delivery(self, output_node):
        """Test response delivery operations."""
        response_data = {
            "content": "Test response content",
            "citations": ["citation1", "citation2"],
            "metadata": {"key": "value"}
        }
        
        delivered_response = await output_node.deliver_response(response_data)
        assert delivered_response is not None
        assert isinstance(delivered_response, dict)
        assert "status" in delivered_response
        assert "response" in delivered_response
    
    @pytest.mark.asyncio
    async def test_response_validation(self, output_node):
        """Test response validation operations."""
        valid_response = {
            "content": "Test response content",
            "citations": ["citation1", "citation2"],
            "metadata": {"key": "value"}
        }
        
        is_valid = await output_node.validate_response(valid_response)
        assert is_valid
        
        invalid_response = {
            "content": "",  # Empty content
            "citations": [],
            "metadata": {}
        }
        
        is_valid = await output_node.validate_response(invalid_response)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_response_error_handling(self, output_node):
        """Test response error handling."""
        with pytest.raises(ValueError):
            await output_node.assemble_response(None)
        
        invalid_data = {"content": "test"}
        with pytest.raises(ValueError):
            await output_node.assemble_response(invalid_data)
    
    @pytest.mark.asyncio
    async def test_response_customization(self, output_node):
        """Test response customization operations."""
        response_data = {
            "content": "Test response content",
            "citations": ["citation1", "citation2"],
            "metadata": {"key": "value"}
        }
        
        custom_format = {
            "style": "markdown",
            "include_citations": True,
            "include_metadata": False
        }
        
        customized_response = await output_node.customize_response(response_data, custom_format)
        assert customized_response is not None
        assert isinstance(customized_response, str)
        assert "citation" in customized_response.lower()
        assert "metadata" not in customized_response.lower()
    
    @pytest.mark.asyncio
    async def test_response_metrics(self, output_node):
        """Test response metrics collection."""
        response_data = {
            "content": "Test response content",
            "citations": ["citation1", "citation2"],
            "metadata": {"key": "value"}
        }
        
        metrics = await output_node.collect_metrics(response_data)
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "length" in metrics
        assert "citation_count" in metrics
        assert "processing_time" in metrics
    
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