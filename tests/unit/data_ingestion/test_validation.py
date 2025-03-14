"""
Unit tests for data validation functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.data_ingestion.training.validation import DataValidator, ValidationResult

class TestDataValidator:
    """Test suite for data validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing."""
        return DataValidator(
            min_sequence_length=2,
            max_sequence_length=100,
            min_examples=2,
            max_examples=1000,
            min_embedding_dim=384,
            max_embedding_dim=768,
            quality_threshold=0.5,
            min_unique_words=5,
            max_duplicate_ratio=0.5,
            min_tfidf_score=0.05,
            outlier_threshold=3.0
        )
    
    @pytest.fixture
    def sample_embedding_data(self):
        """Create sample embedding data for testing."""
        return {
            "texts": [
                "This is a test sentence for embedding validation.",
                "Another test sentence with different words.",
                "A third sentence to test data quality."
            ],
            "embeddings": [
                [0.1] * 384,
                [0.2] * 384,
                [0.3] * 384
            ]
        }
    
    @pytest.fixture
    def sample_llm_data(self):
        """Create sample LLM data for testing."""
        return {
            "examples": [
                {
                    "input": "What is the capital of France?",
                    "output": "The capital of France is Paris."
                },
                {
                    "input": "What is the largest planet?",
                    "output": "Jupiter is the largest planet in our solar system."
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            is_valid=True,
            message="Test validation",
            metrics={"test": 1.0}
        )
        
        assert result.is_valid
        assert result.message == "Test validation"
        assert result.metrics == {"test": 1.0}
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_embedding_data_validation(self, validator, sample_embedding_data):
        """Test embedding data validation."""
        result = await validator.validate_embedding_data(
            sample_embedding_data["texts"],
            sample_embedding_data["embeddings"]
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "metrics" in result
        
        # Check metrics
        metrics = result.metrics
        assert "num_samples" in metrics
        assert "avg_sequence_length" in metrics
        assert "embedding_dim" in metrics
        assert "unique_words" in metrics
        assert "quality_score" in metrics
        assert "duplicate_ratio" in metrics
        assert "tfidf_score" in metrics
    
    @pytest.mark.asyncio
    async def test_llm_data_validation(self, validator, sample_llm_data):
        """Test LLM data validation."""
        result = await validator.validate_llm_data(sample_llm_data["examples"])
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert "metrics" in result
        
        # Check metrics
        metrics = result.metrics
        assert "num_samples" in metrics
        assert "avg_input_length" in metrics
        assert "avg_output_length" in metrics
        assert "unique_words" in metrics
        assert "quality_score" in metrics
        assert "duplicate_ratio" in metrics
        assert "tfidf_score" in metrics
    
    @pytest.mark.asyncio
    async def test_embedding_data_analysis(self, validator, sample_embedding_data):
        """Test embedding data analysis."""
        metrics = await validator._analyze_embedding_data(
            sample_embedding_data["texts"],
            sample_embedding_data["embeddings"]
        )
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "num_samples" in metrics
        assert "avg_sequence_length" in metrics
        assert "embedding_dim" in metrics
        assert "unique_words" in metrics
        assert "quality_score" in metrics
        assert "duplicate_ratio" in metrics
        assert "tfidf_score" in metrics
    
    @pytest.mark.asyncio
    async def test_llm_data_analysis(self, validator, sample_llm_data):
        """Test LLM data analysis."""
        metrics = await validator._analyze_llm_data(sample_llm_data["examples"])
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "num_samples" in metrics
        assert "avg_input_length" in metrics
        assert "avg_output_length" in metrics
        assert "unique_words" in metrics
        assert "quality_score" in metrics
        assert "duplicate_ratio" in metrics
        assert "tfidf_score" in metrics
    
    @pytest.mark.asyncio
    async def test_text_quality_computation(self, validator):
        """Test text quality computation."""
        texts = [
            "This is a high quality sentence with good vocabulary.",
            "Another well-formed sentence with proper grammar.",
            "A third sentence that maintains good quality."
        ]
        
        quality_score = await validator._compute_text_quality(texts)
        
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
        assert quality_score > 0.5  # Should be high for good quality text
    
    @pytest.mark.asyncio
    async def test_example_quality_computation(self, validator):
        """Test example quality computation."""
        examples = [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris."
            },
            {
                "input": "What is the largest planet?",
                "output": "Jupiter is the largest planet in our solar system."
            }
        ]
        
        quality_score = await validator._compute_example_quality(examples)
        
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
        assert quality_score > 0.5  # Should be high for good quality examples
    
    @pytest.mark.asyncio
    async def test_duplicate_ratio_computation(self, validator):
        """Test duplicate ratio computation."""
        texts = [
            "This is a unique sentence.",
            "This is a unique sentence.",  # Duplicate
            "Another unique sentence.",
            "Another unique sentence."  # Duplicate
        ]
        
        ratio = await validator._compute_duplicate_ratio(texts)
        
        assert isinstance(ratio, float)
        assert 0 <= ratio <= 1
        assert ratio == 0.5  # 2 out of 4 are duplicates
    
    @pytest.mark.asyncio
    async def test_tfidf_score_computation(self, validator):
        """Test TF-IDF score computation."""
        texts = [
            "This is a test sentence with unique words.",
            "Another test sentence with different unique words.",
            "A third test sentence with more unique words."
        ]
        
        score = await validator._compute_tfidf_score(texts)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0  # Should be positive for diverse text
    
    @pytest.mark.asyncio
    async def test_outlier_detection(self, validator):
        """Test outlier detection in embeddings."""
        embeddings = [
            [0.1] * 384,  # Normal
            [0.2] * 384,  # Normal
            [10.0] * 384,  # Outlier
            [0.3] * 384   # Normal
        ]
        
        has_outliers = await validator._detect_outliers(embeddings)
        
        assert isinstance(has_outliers, bool)
        assert has_outliers  # Should detect the outlier
    
    @pytest.mark.asyncio
    async def test_duplicate_check(self, validator):
        """Test duplicate example detection."""
        examples = [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris."
            },
            {
                "input": "What is the capital of France?",  # Duplicate input
                "output": "The capital of France is Paris."
            }
        ]
        
        has_duplicates = await validator._check_duplicates(examples)
        
        assert isinstance(has_duplicates, bool)
        assert has_duplicates  # Should detect the duplicate
    
    @pytest.mark.asyncio
    async def test_embedding_data_cleaning(self, validator):
        """Test embedding data cleaning."""
        texts = [
            "This is a test sentence.",
            "This is a test sentence.",  # Duplicate
            "Another test sentence.",
            "   "  # Empty after cleaning
        ]
        embeddings = [
            [0.1] * 384,
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]
        
        cleaned_texts, cleaned_embeddings = await validator.clean_embedding_data(texts, embeddings)
        
        assert len(cleaned_texts) == 2  # Should remove duplicates and empty
        assert len(cleaned_embeddings) == 2
        assert cleaned_texts[0] != cleaned_texts[1]  # Should be unique
    
    @pytest.mark.asyncio
    async def test_llm_data_cleaning(self, validator):
        """Test LLM data cleaning."""
        examples = [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris."
            },
            {
                "input": "What is the capital of France?",  # Duplicate
                "output": "The capital of France is Paris."
            },
            {
                "input": "   ",  # Empty
                "output": "   "   # Empty
            }
        ]
        
        cleaned_examples = await validator.clean_llm_data(examples)
        
        assert len(cleaned_examples) == 1  # Should remove duplicates and empty
        assert cleaned_examples[0]["input"] == "What is the capital of France?" 