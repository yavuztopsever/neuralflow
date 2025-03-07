"""
Unit tests for search tool functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.tools.search import SearchTool

class TestSearchTool:
    """Test suite for search tool functionality."""
    
    @pytest.fixture
    def search_tool(self):
        """Create a search tool for testing."""
        return SearchTool()
    
    @pytest.mark.asyncio
    async def test_search_operations(self, search_tool):
        """Test search tool operations."""
        query = "test query"
        results = await search_tool.web_search(query)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0
        assert all("title" in r and "url" in r for r in results)
        
        doc_results = await search_tool.document_search(query)
        assert doc_results is not None
        assert isinstance(doc_results, list)
        assert all("content" in r and "score" in r for r in doc_results)
        
        semantic_results = await search_tool.semantic_search(query)
        assert semantic_results is not None
        assert isinstance(semantic_results, list)
        assert all("content" in r and "relevance" in r for r in semantic_results)
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_tool):
        """Test search tool error handling."""
        with pytest.raises(ValueError):
            await search_tool.web_search(None)
    
    @pytest.mark.asyncio
    async def test_search_metrics(self, search_tool):
        """Test search tool metrics collection."""
        search_metrics = await search_tool.collect_metrics()
        assert search_metrics is not None
        assert isinstance(search_metrics, dict)
        assert "total_searches" in search_metrics
        assert "average_response_time" in search_metrics
    
    @pytest.mark.asyncio
    async def test_search_optimization(self, search_tool):
        """Test search tool optimization operations."""
        search_params = {
            "max_results": 10,
            "timeout": 5,
            "cache_size": 100
        }
        
        optimized_search = await search_tool.optimize(search_params)
        assert optimized_search is not None
        assert optimized_search.is_optimized 