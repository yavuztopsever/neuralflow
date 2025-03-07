"""
Unit tests for analysis tool functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.tools.analysis import AnalysisTool

class TestAnalysisTool:
    """Test suite for analysis tool functionality."""
    
    @pytest.fixture
    def analysis_tool(self):
        """Create an analysis tool for testing."""
        return AnalysisTool()
    
    @pytest.mark.asyncio
    async def test_analysis_operations(self, analysis_tool):
        """Test analysis tool operations."""
        text = "This is a positive text."
        sentiment = await analysis_tool.analyze_sentiment(text)
        assert sentiment is not None
        assert isinstance(sentiment, dict)
        assert "score" in sentiment
        assert "label" in sentiment
        
        topics = await analysis_tool.analyze_topics(text)
        assert topics is not None
        assert isinstance(topics, list)
        assert all("topic" in t and "confidence" in t for t in topics)
        
        entities = await analysis_tool.analyze_entities(text)
        assert entities is not None
        assert isinstance(entities, list)
        assert all("entity" in e and "type" in e for e in entities)
    
    @pytest.mark.asyncio
    async def test_analysis_error_handling(self, analysis_tool):
        """Test analysis tool error handling."""
        with pytest.raises(ValueError):
            await analysis_tool.analyze_sentiment(None)
    
    @pytest.mark.asyncio
    async def test_analysis_metrics(self, analysis_tool):
        """Test analysis tool metrics collection."""
        analysis_metrics = await analysis_tool.collect_metrics()
        assert analysis_metrics is not None
        assert isinstance(analysis_metrics, dict)
        assert "total_analyses" in analysis_metrics
        assert "average_processing_time" in analysis_metrics
    
    @pytest.mark.asyncio
    async def test_analysis_optimization(self, analysis_tool):
        """Test analysis tool optimization operations."""
        analysis_params = {
            "batch_size": 5,
            "cache_enabled": True,
            "max_text_length": 1000
        }
        
        optimized_analysis = await analysis_tool.optimize(analysis_params)
        assert optimized_analysis is not None
        assert optimized_analysis.is_optimized 