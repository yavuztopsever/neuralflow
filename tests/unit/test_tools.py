"""
Unit tests for tools functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.tools.search import SearchTool
from src.tools.analysis import AnalysisTool
from src.tools.generation import GenerationTool
from src.tools.transformation import TransformationTool

class TestTools:
    """Test suite for tools functionality."""
    
    @pytest.fixture
    def search_tool(self):
        """Create a search tool for testing."""
        return SearchTool()
    
    @pytest.fixture
    def analysis_tool(self):
        """Create an analysis tool for testing."""
        return AnalysisTool()
    
    @pytest.fixture
    def generation_tool(self):
        """Create a generation tool for testing."""
        return GenerationTool()
    
    @pytest.fixture
    def transformation_tool(self):
        """Create a transformation tool for testing."""
        return TransformationTool()
    
    @pytest.mark.asyncio
    async def test_search_operations(self, search_tool):
        """Test search tool operations."""
        # Test web search
        query = "test query"
        results = await search_tool.web_search(query)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) > 0
        assert all("title" in r and "url" in r for r in results)
        
        # Test document search
        doc_results = await search_tool.document_search(query)
        assert doc_results is not None
        assert isinstance(doc_results, list)
        assert all("content" in r and "score" in r for r in doc_results)
        
        # Test semantic search
        semantic_results = await search_tool.semantic_search(query)
        assert semantic_results is not None
        assert isinstance(semantic_results, list)
        assert all("content" in r and "relevance" in r for r in semantic_results)
    
    @pytest.mark.asyncio
    async def test_analysis_operations(self, analysis_tool):
        """Test analysis tool operations."""
        # Test sentiment analysis
        text = "This is a positive text."
        sentiment = await analysis_tool.analyze_sentiment(text)
        assert sentiment is not None
        assert isinstance(sentiment, dict)
        assert "score" in sentiment
        assert "label" in sentiment
        
        # Test topic analysis
        topics = await analysis_tool.analyze_topics(text)
        assert topics is not None
        assert isinstance(topics, list)
        assert all("topic" in t and "confidence" in t for t in topics)
        
        # Test entity analysis
        entities = await analysis_tool.analyze_entities(text)
        assert entities is not None
        assert isinstance(entities, list)
        assert all("entity" in e and "type" in e for e in entities)
    
    @pytest.mark.asyncio
    async def test_generation_operations(self, generation_tool):
        """Test generation tool operations."""
        # Test text generation
        prompt = "Generate a test text"
        generated_text = await generation_tool.generate_text(prompt)
        assert generated_text is not None
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0
        
        # Test code generation
        code_prompt = "Generate a test function"
        generated_code = await generation_tool.generate_code(code_prompt)
        assert generated_code is not None
        assert isinstance(generated_code, str)
        assert "def" in generated_code or "function" in generated_code
        
        # Test image generation
        image_prompt = "Generate a test image"
        generated_image = await generation_tool.generate_image(image_prompt)
        assert generated_image is not None
        assert isinstance(generated_image, bytes)
        assert len(generated_image) > 0
    
    @pytest.mark.asyncio
    async def test_transformation_operations(self, transformation_tool):
        """Test transformation tool operations."""
        # Test text transformation
        text = "Test text"
        transformed_text = await transformation_tool.transform_text(text, "uppercase")
        assert transformed_text == "TEST TEXT"
        
        # Test format transformation
        data = {"key": "value"}
        transformed_data = await transformation_tool.transform_format(data, "yaml")
        assert transformed_data is not None
        assert isinstance(transformed_data, str)
        assert "key:" in transformed_data
        
        # Test language transformation
        translated_text = await transformation_tool.translate_text(text, "es")
        assert translated_text is not None
        assert isinstance(translated_text, str)
        assert len(translated_text) > 0
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self, search_tool, analysis_tool, generation_tool, transformation_tool):
        """Test tool error handling."""
        # Test search tool errors
        with pytest.raises(ValueError):
            await search_tool.web_search(None)
        
        # Test analysis tool errors
        with pytest.raises(ValueError):
            await analysis_tool.analyze_sentiment(None)
        
        # Test generation tool errors
        with pytest.raises(ValueError):
            await generation_tool.generate_text(None)
        
        # Test transformation tool errors
        with pytest.raises(ValueError):
            await transformation_tool.transform_text(None, "uppercase")
    
    @pytest.mark.asyncio
    async def test_tool_metrics(self, search_tool, analysis_tool, generation_tool, transformation_tool):
        """Test tool metrics collection."""
        # Test search tool metrics
        search_metrics = await search_tool.collect_metrics()
        assert search_metrics is not None
        assert isinstance(search_metrics, dict)
        assert "total_searches" in search_metrics
        assert "average_response_time" in search_metrics
        
        # Test analysis tool metrics
        analysis_metrics = await analysis_tool.collect_metrics()
        assert analysis_metrics is not None
        assert isinstance(analysis_metrics, dict)
        assert "total_analyses" in analysis_metrics
        assert "average_processing_time" in analysis_metrics
        
        # Test generation tool metrics
        generation_metrics = await generation_tool.collect_metrics()
        assert generation_metrics is not None
        assert isinstance(generation_metrics, dict)
        assert "total_generations" in generation_metrics
        assert "average_generation_time" in generation_metrics
        
        # Test transformation tool metrics
        transformation_metrics = await transformation_tool.collect_metrics()
        assert transformation_metrics is not None
        assert isinstance(transformation_metrics, dict)
        assert "total_transformations" in transformation_metrics
        assert "average_transformation_time" in transformation_metrics
    
    @pytest.mark.asyncio
    async def test_tool_optimization(self, search_tool, analysis_tool, generation_tool, transformation_tool):
        """Test tool optimization operations."""
        # Test search tool optimization
        search_params = {
            "max_results": 10,
            "timeout": 5,
            "cache_size": 100
        }
        
        optimized_search = await search_tool.optimize(search_params)
        assert optimized_search is not None
        assert optimized_search.is_optimized
        
        # Test analysis tool optimization
        analysis_params = {
            "batch_size": 5,
            "cache_enabled": True,
            "max_text_length": 1000
        }
        
        optimized_analysis = await analysis_tool.optimize(analysis_params)
        assert optimized_analysis is not None
        assert optimized_analysis.is_optimized
        
        # Test generation tool optimization
        generation_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "batch_size": 3
        }
        
        optimized_generation = await generation_tool.optimize(generation_params)
        assert optimized_generation is not None
        assert optimized_generation.is_optimized
        
        # Test transformation tool optimization
        transformation_params = {
            "cache_size": 100,
            "max_batch_size": 10,
            "timeout": 5
        }
        
        optimized_transformation = await transformation_tool.optimize(transformation_params)
        assert optimized_transformation is not None
        assert optimized_transformation.is_optimized 