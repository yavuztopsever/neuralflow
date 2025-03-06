"""
Unit tests for utility functions.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.utils.text import TextProcessor
from src.utils.embeddings import EmbeddingGenerator
from src.utils.logging import Logger
from src.utils.metrics import MetricsCollector

class TestUtils:
    """Test suite for utility functions."""
    
    @pytest.fixture
    def text_processor(self):
        """Create a text processor for testing."""
        return TextProcessor()
    
    @pytest.fixture
    def embedding_generator(self):
        """Create an embedding generator for testing."""
        return EmbeddingGenerator()
    
    @pytest.fixture
    def logger(self):
        """Create a logger for testing."""
        return Logger()
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing."""
        return MetricsCollector()
    
    @pytest.mark.asyncio
    async def test_text_processing(self, text_processor):
        """Test text processing operations."""
        # Test text cleaning
        text = "  Test Text  "
        cleaned_text = await text_processor.clean_text(text)
        assert cleaned_text == "Test Text"
        
        # Test text tokenization
        tokens = await text_processor.tokenize(text)
        assert len(tokens) == 2
        assert tokens[0] == "Test"
        assert tokens[1] == "Text"
        
        # Test text normalization
        normalized_text = await text_processor.normalize_text(text)
        assert normalized_text == "test text"
        
        # Test text summarization
        long_text = "This is a long text that needs to be summarized. " * 5
        summary = await text_processor.summarize_text(long_text, max_length=50)
        assert len(summary) <= 50
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, embedding_generator):
        """Test embedding generation operations."""
        # Test generating embeddings
        text = "Test text for embedding"
        embedding = await embedding_generator.generate_embedding(text)
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        
        # Test batch embedding generation
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await embedding_generator.generate_batch_embeddings(texts)
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Test embedding similarity
        similarity = await embedding_generator.calculate_similarity(
            embedding,
            embedding
        )
        assert similarity == 1.0
    
    @pytest.mark.asyncio
    async def test_logging(self, logger):
        """Test logging operations."""
        # Test info logging
        await logger.info("Test info message")
        assert logger.has_log("info", "Test info message")
        
        # Test error logging
        await logger.error("Test error message")
        assert logger.has_log("error", "Test error message")
        
        # Test warning logging
        await logger.warning("Test warning message")
        assert logger.has_log("warning", "Test warning message")
        
        # Test debug logging
        await logger.debug("Test debug message")
        assert logger.has_log("debug", "Test debug message")
        
        # Test log rotation
        for i in range(100):
            await logger.info(f"Test message {i}")
        assert logger.get_log_count() <= 100
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, metrics_collector):
        """Test metrics collection operations."""
        # Test collecting metrics
        metrics = {
            "operation": "test",
            "duration": 0.1,
            "success": True
        }
        
        await metrics_collector.collect(metrics)
        assert metrics_collector.get_metrics_count() > 0
        
        # Test aggregating metrics
        aggregated = await metrics_collector.aggregate_metrics()
        assert aggregated is not None
        assert isinstance(aggregated, dict)
        assert "average_duration" in aggregated
        assert "success_rate" in aggregated
        
        # Test exporting metrics
        exported = await metrics_collector.export_metrics()
        assert exported is not None
        assert isinstance(exported, str)
        assert len(exported) > 0
    
    @pytest.mark.asyncio
    async def test_utility_error_handling(self, text_processor, embedding_generator, logger, metrics_collector):
        """Test utility error handling."""
        # Test text processor errors
        with pytest.raises(ValueError):
            await text_processor.clean_text(None)
        
        # Test embedding generator errors
        with pytest.raises(ValueError):
            await embedding_generator.generate_embedding(None)
        
        # Test logger errors
        with pytest.raises(ValueError):
            await logger.info(None)
        
        # Test metrics collector errors
        with pytest.raises(ValueError):
            await metrics_collector.collect(None)
    
    @pytest.mark.asyncio
    async def test_utility_optimization(self, text_processor, embedding_generator, logger, metrics_collector):
        """Test utility optimization operations."""
        # Test text processor optimization
        text_params = {
            "max_length": 100,
            "min_length": 10,
            "batch_size": 5
        }
        
        optimized_processor = await text_processor.optimize(text_params)
        assert optimized_processor is not None
        assert optimized_processor.is_optimized
        
        # Test embedding generator optimization
        embedding_params = {
            "model": "fast",
            "dimension": 64,
            "batch_size": 10
        }
        
        optimized_generator = await embedding_generator.optimize(embedding_params)
        assert optimized_generator is not None
        assert optimized_generator.is_optimized
        
        # Test logger optimization
        logger_params = {
            "max_logs": 1000,
            "rotation_size": 100,
            "compression": True
        }
        
        optimized_logger = await logger.optimize(logger_params)
        assert optimized_logger is not None
        assert optimized_logger.is_optimized
        
        # Test metrics collector optimization
        metrics_params = {
            "max_metrics": 1000,
            "aggregation_interval": 60,
            "export_format": "json"
        }
        
        optimized_collector = await metrics_collector.optimize(metrics_params)
        assert optimized_collector is not None
        assert optimized_collector.is_optimized 