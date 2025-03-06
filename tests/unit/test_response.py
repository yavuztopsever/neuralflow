"""
Unit tests for response generation functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.core.nodes.output import OutputNode
from src.core.nodes.reasoning import ReasoningNode

class TestResponseGeneration:
    """Test suite for response generation functionality."""
    
    @pytest.fixture
    def output_node(self):
        """Create an output node for testing."""
        return OutputNode("test_output", "output", {})
    
    @pytest.fixture
    def reasoning_node(self):
        """Create a reasoning node for testing."""
        return ReasoningNode("test_reasoning", "reasoning", {})
    
    @pytest.mark.asyncio
    async def test_response_assembly(self, output_node):
        """Test response assembly operations."""
        # Test assembling response from components
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
        # Test formatting response
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
        # Test delivering response
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
        # Test validating response
        valid_response = {
            "content": "Test response content",
            "citations": ["citation1", "citation2"],
            "metadata": {"key": "value"}
        }
        
        is_valid = await output_node.validate_response(valid_response)
        assert is_valid
        
        # Test invalid response
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
        # Test handling invalid response data
        with pytest.raises(ValueError):
            await output_node.assemble_response(None)
        
        # Test handling missing required fields
        invalid_data = {"content": "test"}
        with pytest.raises(ValueError):
            await output_node.assemble_response(invalid_data)
    
    @pytest.mark.asyncio
    async def test_response_customization(self, output_node):
        """Test response customization operations."""
        # Test customizing response format
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
        # Test collecting response metrics
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