"""
Data handling and processing tests for the NeuralFlow system.
Tests data loading, preprocessing, validation, and transformation.
"""
import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from neuralflow.data.loader import DataLoader
from neuralflow.data.preprocessor import DataPreprocessor
from neuralflow.data.validator import DataValidator
from neuralflow.data.transformer import DataTransformer
from neuralflow.data.schema import DataSchema
from neuralflow.core.config import SystemConfig

class TestData:
    """Test suite for data handling and processing."""
    
    @pytest.fixture
    def system_config(self):
        """Create system configuration for testing."""
        return SystemConfig(
            max_workers=4,
            cache_size=1000,
            max_memory_mb=1024,
            timeout_seconds=30,
            retry_attempts=3,
            batch_size=32,
            enable_metrics=True
        )
    
    @pytest.fixture
    def data_schema(self):
        """Create data schema for testing."""
        return DataSchema({
            "user_id": str,
            "query": str,
            "timestamp": datetime,
            "metadata": Dict[str, Any]
        })
    
    @pytest.fixture
    async def data_components(self, system_config, data_schema):
        """Create data components for testing."""
        loader = DataLoader(config=system_config)
        preprocessor = DataPreprocessor(config=system_config)
        validator = DataValidator(config=system_config, schema=data_schema)
        transformer = DataTransformer(config=system_config)
        
        return {
            "config": system_config,
            "schema": data_schema,
            "loader": loader,
            "preprocessor": preprocessor,
            "validator": validator,
            "transformer": transformer
        }
    
    @pytest.mark.asyncio
    async def test_data_loading(self, data_components):
        """Test data loading functionality."""
        # Test file loading
        data = await data_components["loader"].load_file("test_data.json")
        assert data is not None
        assert isinstance(data, List)
        
        # Test batch loading
        batches = await data_components["loader"].load_batches("test_data.json", batch_size=10)
        assert batches is not None
        assert len(batches) > 0
        
        # Test streaming loading
        async for chunk in data_components["loader"].stream_file("test_data.json"):
            assert chunk is not None
            assert len(chunk) > 0
    
    @pytest.mark.asyncio
    async def test_data_preprocessing(self, data_components):
        """Test data preprocessing functionality."""
        # Test text preprocessing
        text = "Test text with special characters!@#$%^&*()"
        processed = await data_components["preprocessor"].preprocess_text(text)
        assert processed is not None
        assert "special" not in processed
        
        # Test batch preprocessing
        texts = ["Text 1", "Text 2", "Text 3"]
        processed_batch = await data_components["preprocessor"].preprocess_batch(texts)
        assert processed_batch is not None
        assert len(processed_batch) == len(texts)
        
        # Test streaming preprocessing
        async for processed_chunk in data_components["preprocessor"].stream_preprocess(texts):
            assert processed_chunk is not None
            assert len(processed_chunk) > 0
    
    @pytest.mark.asyncio
    async def test_data_validation(self, data_components):
        """Test data validation functionality."""
        # Test single record validation
        record = {
            "user_id": "test_user",
            "query": "test query",
            "timestamp": datetime.now(),
            "metadata": {"source": "test"}
        }
        is_valid = await data_components["validator"].validate_record(record)
        assert is_valid
        
        # Test batch validation
        records = [record] * 3
        validation_results = await data_components["validator"].validate_batch(records)
        assert all(validation_results)
        
        # Test schema validation
        invalid_record = {
            "user_id": 123,  # Should be string
            "query": "test query",
            "timestamp": "invalid",  # Should be datetime
            "metadata": "invalid"  # Should be dict
        }
        is_valid = await data_components["validator"].validate_record(invalid_record)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_data_transformation(self, data_components):
        """Test data transformation functionality."""
        # Test single record transformation
        record = {
            "user_id": "test_user",
            "query": "test query",
            "timestamp": datetime.now(),
            "metadata": {"source": "test"}
        }
        transformed = await data_components["transformer"].transform_record(record)
        assert transformed is not None
        assert "transformed_timestamp" in transformed
        
        # Test batch transformation
        records = [record] * 3
        transformed_batch = await data_components["transformer"].transform_batch(records)
        assert transformed_batch is not None
        assert len(transformed_batch) == len(records)
        
        # Test streaming transformation
        async for transformed_chunk in data_components["transformer"].stream_transform(records):
            assert transformed_chunk is not None
            assert len(transformed_chunk) > 0
    
    @pytest.mark.asyncio
    async def test_data_pipeline(self, data_components):
        """Test complete data pipeline."""
        # Load data
        data = await data_components["loader"].load_file("test_data.json")
        
        # Preprocess data
        processed_data = await data_components["preprocessor"].preprocess_batch(data)
        
        # Validate data
        validation_results = await data_components["validator"].validate_batch(processed_data)
        assert all(validation_results)
        
        # Transform data
        transformed_data = await data_components["transformer"].transform_batch(processed_data)
        
        # Verify pipeline results
        assert transformed_data is not None
        assert len(transformed_data) == len(data)
        assert all(isinstance(record, dict) for record in transformed_data)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, data_components):
        """Test error handling in data processing."""
        # Test invalid file loading
        with pytest.raises(FileNotFoundError):
            await data_components["loader"].load_file("nonexistent.json")
        
        # Test invalid data preprocessing
        with pytest.raises(ValueError):
            await data_components["preprocessor"].preprocess_text(None)
        
        # Test invalid data validation
        with pytest.raises(ValueError):
            await data_components["validator"].validate_record(None)
        
        # Test invalid data transformation
        with pytest.raises(ValueError):
            await data_components["transformer"].transform_record(None)
    
    @pytest.mark.asyncio
    async def test_performance(self, data_components):
        """Test data processing performance."""
        # Generate large test dataset
        large_data = [{
            "user_id": f"user_{i}",
            "query": f"test query {i}",
            "timestamp": datetime.now(),
            "metadata": {"source": "test"}
        } for i in range(1000)]
        
        # Test batch processing performance
        start_time = datetime.now()
        processed = await data_components["preprocessor"].preprocess_batch(large_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert processing_time < 5.0  # Should process 1000 records within 5 seconds
        
        # Test streaming performance
        start_time = datetime.now()
        async for chunk in data_components["preprocessor"].stream_preprocess(large_data):
            assert len(chunk) > 0
        streaming_time = (datetime.now() - start_time).total_seconds()
        
        assert streaming_time < 10.0  # Should stream process 1000 records within 10 seconds 