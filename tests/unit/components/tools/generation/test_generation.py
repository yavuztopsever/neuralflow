"""
Unit tests for generation tool functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.tools.generation import GenerationTool

class TestGenerationTool:
    """Test suite for generation tool functionality."""
    
    @pytest.fixture
    def generation_tool(self):
        """Create a generation tool for testing."""
        return GenerationTool()
    
    @pytest.mark.asyncio
    async def test_generation_operations(self, generation_tool):
        """Test generation tool operations."""
        prompt = "Generate a test text"
        generated_text = await generation_tool.generate_text(prompt)
        assert generated_text is not None
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0
        
        code_prompt = "Generate a test function"
        generated_code = await generation_tool.generate_code(code_prompt)
        assert generated_code is not None
        assert isinstance(generated_code, str)
        assert "def" in generated_code or "function" in generated_code
        
        image_prompt = "Generate a test image"
        generated_image = await generation_tool.generate_image(image_prompt)
        assert generated_image is not None
        assert isinstance(generated_image, bytes)
        assert len(generated_image) > 0
    
    @pytest.mark.asyncio
    async def test_generation_error_handling(self, generation_tool):
        """Test generation tool error handling."""
        with pytest.raises(ValueError):
            await generation_tool.generate_text(None)
    
    @pytest.mark.asyncio
    async def test_generation_metrics(self, generation_tool):
        """Test generation tool metrics collection."""
        generation_metrics = await generation_tool.collect_metrics()
        assert generation_metrics is not None
        assert isinstance(generation_metrics, dict)
        assert "total_generations" in generation_metrics
        assert "average_generation_time" in generation_metrics
    
    @pytest.mark.asyncio
    async def test_generation_optimization(self, generation_tool):
        """Test generation tool optimization operations."""
        generation_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "batch_size": 3
        }
        
        optimized_generation = await generation_tool.optimize(generation_params)
        assert optimized_generation is not None
        assert optimized_generation.is_optimized 