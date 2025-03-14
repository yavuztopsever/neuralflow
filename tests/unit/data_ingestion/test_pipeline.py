"""
Unit tests for data science pipeline functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import tempfile

from src.core.data_ingestion.training.pipeline import DataSciencePipeline
from src.core.data_ingestion.training.validation import DataValidator
from src.core.data_ingestion.training.augmentation import DataAugmentor

class TestDataSciencePipeline:
    """Test suite for data science pipeline functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pipeline(self, temp_output_dir):
        """Create a pipeline instance for testing."""
        return DataSciencePipeline(
            output_dir=temp_output_dir,
            n_splits=3,
            random_state=42
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "embedding": {
                "texts": ["text1", "text2", "text3"],
                "embeddings": [
                    [0.1] * 384,
                    [0.2] * 384,
                    [0.3] * 384
                ]
            },
            "llm": {
                "examples": [
                    {
                        "input": "input1",
                        "output": "output1"
                    },
                    {
                        "input": "input2",
                        "output": "output2"
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.output_dir.exists()
        assert pipeline.n_splits == 3
        assert pipeline.random_state == 42
        assert isinstance(pipeline.validator, DataValidator)
        assert isinstance(pipeline.data_augmentor, DataAugmentor)
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self, pipeline, sample_data):
        """Test running the complete pipeline."""
        results = await pipeline.run_pipeline(
            data=sample_data,
            model_type="all",
            session_id="test_session"
        )
        
        assert results is not None
        assert "timestamp" in results
        assert "model_type" in results
        assert "session_id" in results
        assert "steps" in results
        
        # Check pipeline steps
        steps = results["steps"]
        assert "validation" in steps
        assert "cleaning" in steps
        assert "augmentation" in steps
        assert "feature_engineering" in steps
        assert "cross_validation" in steps
        assert "training" in steps
    
    @pytest.mark.asyncio
    async def test_validation_step(self, pipeline, sample_data):
        """Test data validation step."""
        validation_results = await pipeline._run_validation(sample_data, "all")
        
        assert validation_results is not None
        assert "embedding" in validation_results
        assert "llm" in validation_results
        
        # Check validation results
        embedding_validation = validation_results["embedding"]
        assert embedding_validation.is_valid
        assert "metrics" in embedding_validation
        
        llm_validation = validation_results["llm"]
        assert llm_validation.is_valid
        assert "metrics" in llm_validation
    
    @pytest.mark.asyncio
    async def test_cleaning_step(self, pipeline, sample_data):
        """Test data cleaning step."""
        cleaned_data = await pipeline._run_cleaning(sample_data, "all")
        
        assert cleaned_data is not None
        assert "embedding" in cleaned_data
        assert "llm" in cleaned_data
        
        # Check cleaned data
        embedding_data = cleaned_data["embedding"]
        assert "texts" in embedding_data
        assert "embeddings" in embedding_data
        assert len(embedding_data["texts"]) > 0
        assert len(embedding_data["embeddings"]) > 0
        
        llm_data = cleaned_data["llm"]
        assert "examples" in llm_data
        assert len(llm_data["examples"]) > 0
    
    @pytest.mark.asyncio
    async def test_augmentation_step(self, pipeline, sample_data):
        """Test data augmentation step."""
        augmented_data = await pipeline._run_augmentation(sample_data, "all")
        
        assert augmented_data is not None
        assert "embedding" in augmented_data
        assert "llm" in augmented_data
        
        # Check augmented data
        embedding_data = augmented_data["embedding"]
        assert len(embedding_data["texts"]) >= len(sample_data["embedding"]["texts"])
        assert len(embedding_data["embeddings"]) >= len(sample_data["embedding"]["embeddings"])
        
        llm_data = augmented_data["llm"]
        assert len(llm_data["examples"]) >= len(sample_data["llm"]["examples"])
    
    @pytest.mark.asyncio
    async def test_feature_engineering_step(self, pipeline, sample_data):
        """Test feature engineering step."""
        engineered_data = await pipeline._run_feature_engineering(sample_data, "all")
        
        assert engineered_data is not None
        assert "embedding" in engineered_data
        assert "llm" in engineered_data
        
        # Check engineered features
        embedding_features = engineered_data["embedding"]
        assert "texts" in embedding_features
        assert "embeddings" in embedding_features
        
        llm_features = engineered_data["llm"]
        assert "examples" in llm_features
    
    @pytest.mark.asyncio
    async def test_cross_validation_step(self, pipeline, sample_data):
        """Test cross-validation step."""
        cv_results = await pipeline._run_cross_validation(sample_data, "all")
        
        assert cv_results is not None
        assert "embedding" in cv_results
        assert "llm" in cv_results
        
        # Check cross-validation results
        embedding_cv = cv_results["embedding"]
        assert "scores" in embedding_cv
        assert "fold_sizes" in embedding_cv
        
        llm_cv = cv_results["llm"]
        assert "scores" in llm_cv
        assert "fold_sizes" in llm_cv
    
    @pytest.mark.asyncio
    async def test_training_config_preparation(self, pipeline, sample_data):
        """Test training configuration preparation."""
        config = await pipeline._prepare_training_config(sample_data, "all")
        
        assert config is not None
        assert "embedding" in config
        assert "llm" in config
        
        # Check configuration parameters
        embedding_config = config["embedding"]
        assert "model_type" in embedding_config
        assert "hyperparameters" in embedding_config
        
        llm_config = config["llm"]
        assert "model_type" in llm_config
        assert "hyperparameters" in llm_config
    
    @pytest.mark.asyncio
    async def test_report_generation(self, pipeline, sample_data):
        """Test report generation."""
        # Run pipeline to get results
        results = await pipeline.run_pipeline(sample_data, "all")
        
        # Generate reports
        await pipeline._generate_reports(results)
        
        # Check report files
        report_dir = pipeline.output_dir / "reports"
        assert report_dir.exists()
        assert (report_dir / "validation_report.png").exists()
        assert (report_dir / "feature_importance.png").exists()
        assert (report_dir / "cross_validation.png").exists()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, pipeline):
        """Test error handling in pipeline."""
        # Test invalid data
        with pytest.raises(ValueError):
            await pipeline.run_pipeline({}, "all")
        
        # Test invalid model type
        with pytest.raises(ValueError):
            await pipeline.run_pipeline({"test": "data"}, "invalid_type")
    
    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline, sample_data):
        """Test cleanup functionality."""
        # Run pipeline to create output files
        await pipeline.run_pipeline(sample_data, "all")
        
        # Cleanup
        await pipeline.cleanup()
        
        # Verify cleanup
        assert not pipeline.output_dir.exists() 