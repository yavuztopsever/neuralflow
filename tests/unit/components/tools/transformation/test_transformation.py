"""
Unit tests for transformation tool functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.tools.transformation import TransformationTool

class TestTransformationTool:
    """Test suite for transformation tool functionality."""
    
    @pytest.fixture
    def transformation_tool(self):
        """Create a transformation tool for testing."""
        return TransformationTool()
    
    @pytest.mark.asyncio
    async def test_transformation_operations(self, transformation_tool):
        """Test transformation tool operations."""
        text = "Test text"
        transformed_text = await transformation_tool.transform_text(text, "uppercase")
        assert transformed_text == "TEST TEXT"
        
        data = {"key": "value"}
        transformed_data = await transformation_tool.transform_format(data, "yaml")
        assert transformed_data is not None
        assert isinstance(transformed_data, str)
        assert "key:" in transformed_data
        
        translated_text = await transformation_tool.translate_text(text, "es")
        assert translated_text is not None
        assert isinstance(translated_text, str)
        assert len(translated_text) > 0
    
    @pytest.mark.asyncio
    async def test_transformation_error_handling(self, transformation_tool):
        """Test transformation tool error handling."""
        with pytest.raises(ValueError):
            await transformation_tool.transform_text(None, "uppercase")
    
    @pytest.mark.asyncio
    async def test_transformation_metrics(self, transformation_tool):
        """Test transformation tool metrics collection."""
        transformation_metrics = await transformation_tool.collect_metrics()
        assert transformation_metrics is not None
        assert isinstance(transformation_metrics, dict)
        assert "total_transformations" in transformation_metrics
        assert "average_transformation_time" in transformation_metrics
    
    @pytest.mark.asyncio
    async def test_transformation_optimization(self, transformation_tool):
        """Test transformation tool optimization operations."""
        transformation_params = {
            "cache_size": 100,
            "max_batch_size": 10,
            "timeout": 5
        }
        
        optimized_transformation = await transformation_tool.optimize(transformation_params)
        assert optimized_transformation is not None
        assert optimized_transformation.is_optimized 