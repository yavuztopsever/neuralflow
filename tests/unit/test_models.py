"""
Unit tests for model functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.models.llm import LLMModel
from src.models.embeddings import EmbeddingModel
from src.models.classifier import ClassifierModel

class TestModels:
    """Test suite for model functionality."""
    
    @pytest.fixture
    def llm_model(self):
        """Create an LLM model for testing."""
        return LLMModel(
            model_name="test-model",
            temperature=0.7,
            max_tokens=100
        )
    
    @pytest.fixture
    def embedding_model(self):
        """Create an embedding model for testing."""
        return EmbeddingModel(
            model_name="test-embedding",
            dimension=128
        )
    
    @pytest.fixture
    def classifier_model(self):
        """Create a classifier model for testing."""
        return ClassifierModel(
            model_name="test-classifier",
            num_classes=2
        )
    
    @pytest.mark.asyncio
    async def test_llm_operations(self, llm_model):
        """Test LLM model operations."""
        # Test text generation
        prompt = "Test prompt"
        response = await llm_model.generate(prompt)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Test batch generation
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await llm_model.generate_batch(prompts)
        assert len(responses) == len(prompts)
        assert all(len(r) > 0 for r in responses)
        
        # Test model parameters
        assert llm_model.temperature == 0.7
        assert llm_model.max_tokens == 100
        
        # Test model configuration
        config = await llm_model.get_config()
        assert config is not None
        assert isinstance(config, dict)
        assert "model_name" in config
        assert "temperature" in config
    
    @pytest.mark.asyncio
    async def test_embedding_operations(self, embedding_model):
        """Test embedding model operations."""
        # Test embedding generation
        text = "Test text for embedding"
        embedding = await embedding_model.generate_embedding(text)
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 128
        assert all(isinstance(x, float) for x in embedding)
        
        # Test batch embedding generation
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await embedding_model.generate_batch_embeddings(texts)
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 128 for emb in embeddings)
        
        # Test embedding similarity
        similarity = await embedding_model.calculate_similarity(
            embedding,
            embedding
        )
        assert similarity == 1.0
        
        # Test model parameters
        assert embedding_model.dimension == 128
    
    @pytest.mark.asyncio
    async def test_classifier_operations(self, classifier_model):
        """Test classifier model operations."""
        # Test classification
        text = "Test text for classification"
        prediction = await classifier_model.classify(text)
        assert prediction is not None
        assert isinstance(prediction, dict)
        assert "class" in prediction
        assert "confidence" in prediction
        
        # Test batch classification
        texts = ["Text 1", "Text 2", "Text 3"]
        predictions = await classifier_model.classify_batch(texts)
        assert len(predictions) == len(texts)
        assert all("class" in p and "confidence" in p for p in predictions)
        
        # Test model parameters
        assert classifier_model.num_classes == 2
        
        # Test model evaluation
        test_data = [
            {"text": "Positive text", "label": 1},
            {"text": "Negative text", "label": 0}
        ]
        metrics = await classifier_model.evaluate(test_data)
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
    
    @pytest.mark.asyncio
    async def test_model_error_handling(self, llm_model, embedding_model, classifier_model):
        """Test model error handling."""
        # Test LLM model errors
        with pytest.raises(ValueError):
            await llm_model.generate(None)
        
        # Test embedding model errors
        with pytest.raises(ValueError):
            await embedding_model.generate_embedding(None)
        
        # Test classifier model errors
        with pytest.raises(ValueError):
            await classifier_model.classify(None)
    
    @pytest.mark.asyncio
    async def test_model_metrics(self, llm_model, embedding_model, classifier_model):
        """Test model metrics collection."""
        # Test LLM model metrics
        llm_metrics = await llm_model.collect_metrics()
        assert llm_metrics is not None
        assert isinstance(llm_metrics, dict)
        assert "total_tokens" in llm_metrics
        assert "average_response_time" in llm_metrics
        
        # Test embedding model metrics
        embedding_metrics = await embedding_model.collect_metrics()
        assert embedding_metrics is not None
        assert isinstance(embedding_metrics, dict)
        assert "total_embeddings" in embedding_metrics
        assert "average_generation_time" in embedding_metrics
        
        # Test classifier model metrics
        classifier_metrics = await classifier_model.collect_metrics()
        assert classifier_metrics is not None
        assert isinstance(classifier_metrics, dict)
        assert "total_predictions" in classifier_metrics
        assert "average_confidence" in classifier_metrics
    
    @pytest.mark.asyncio
    async def test_model_optimization(self, llm_model, embedding_model, classifier_model):
        """Test model optimization operations."""
        # Test LLM model optimization
        llm_params = {
            "temperature": 0.8,
            "max_tokens": 200,
            "top_p": 0.9
        }
        
        optimized_llm = await llm_model.optimize(llm_params)
        assert optimized_llm is not None
        assert optimized_llm.temperature == 0.8
        assert optimized_llm.max_tokens == 200
        
        # Test embedding model optimization
        embedding_params = {
            "dimension": 256,
            "batch_size": 32,
            "normalize": True
        }
        
        optimized_embedding = await embedding_model.optimize(embedding_params)
        assert optimized_embedding is not None
        assert optimized_embedding.dimension == 256
        
        # Test classifier model optimization
        classifier_params = {
            "num_classes": 3,
            "threshold": 0.5,
            "batch_size": 16
        }
        
        optimized_classifier = await classifier_model.optimize(classifier_params)
        assert optimized_classifier is not None
        assert optimized_classifier.num_classes == 3 