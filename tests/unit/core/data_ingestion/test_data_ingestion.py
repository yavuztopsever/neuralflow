"""
Unit tests for the data ingestion components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import BaseModel

from src.core.data_ingestion.config import DataIngestionConfig
from src.core.data_ingestion.workflow_nodes import (
    DataIngestionNode,
    DataValidationNode,
    DataTransformationNode,
    DataEnrichmentNode
)

@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    return {
        "id": "test_id",
        "content": "test content",
        "metadata": {
            "source": "test_source",
            "timestamp": "2024-03-07T00:00:00Z"
        }
    }

@pytest.fixture
def mock_validation_config():
    """Create mock validation configuration."""
    return {
        "required_fields": ["id", "content"],
        "field_types": {
            "id": "string",
            "content": "string",
            "metadata": "object"
        },
        "validation_rules": {
            "content": {
                "min_length": 1,
                "max_length": 1000
            }
        }
    }

@pytest.fixture
def mock_transformation_config():
    """Create mock transformation configuration."""
    return {
        "transformations": [
            {
                "field": "content",
                "operation": "lowercase"
            },
            {
                "field": "metadata.timestamp",
                "operation": "datetime_format",
                "params": {"format": "%Y-%m-%dT%H:%M:%SZ"}
            }
        ]
    }

@pytest.fixture
def mock_enrichment_config():
    """Create mock enrichment configuration."""
    return {
        "enrichments": [
            {
                "field": "metadata",
                "operation": "add_source_info",
                "params": {
                    "source_type": "document",
                    "version": "1.0"
                }
            }
        ]
    }

def test_data_ingestion_config_initialization():
    """Test initialization of DataIngestionConfig."""
    config = DataIngestionConfig(
        batch_size=100,
        max_workers=4,
        validation_enabled=True,
        transformation_enabled=True,
        enrichment_enabled=True
    )
    
    assert config.batch_size == 100
    assert config.max_workers == 4
    assert config.validation_enabled is True
    assert config.transformation_enabled is True
    assert config.enrichment_enabled is True

def test_data_ingestion_node_initialization():
    """Test initialization of DataIngestionNode."""
    node = DataIngestionNode(node_id="ingestion_node")
    assert node.node_id == "ingestion_node"
    assert node.inputs == {}
    assert node.outputs == {}
    assert node.state == {}

def test_data_validation_node_initialization(mock_validation_config):
    """Test initialization of DataValidationNode."""
    node = DataValidationNode(
        node_id="validation_node",
        config=mock_validation_config
    )
    assert node.node_id == "validation_node"
    assert node.config == mock_validation_config
    assert node.validation_results == []

def test_data_transformation_node_initialization(mock_transformation_config):
    """Test initialization of DataTransformationNode."""
    node = DataTransformationNode(
        node_id="transformation_node",
        config=mock_transformation_config
    )
    assert node.node_id == "transformation_node"
    assert node.config == mock_transformation_config
    assert node.transformation_results == []

def test_data_enrichment_node_initialization(mock_enrichment_config):
    """Test initialization of DataEnrichmentNode."""
    node = DataEnrichmentNode(
        node_id="enrichment_node",
        config=mock_enrichment_config
    )
    assert node.node_id == "enrichment_node"
    assert node.config == mock_enrichment_config
    assert node.enrichment_results == []

def test_data_validation(mock_data, mock_validation_config):
    """Test data validation process."""
    node = DataValidationNode(
        node_id="validation_node",
        config=mock_validation_config
    )
    
    # Test valid data
    result = node.validate_data(mock_data)
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    
    # Test invalid data
    invalid_data = {"id": "test_id"}  # Missing required field
    result = node.validate_data(invalid_data)
    assert result["is_valid"] is False
    assert len(result["errors"]) > 0

def test_data_transformation(mock_data, mock_transformation_config):
    """Test data transformation process."""
    node = DataTransformationNode(
        node_id="transformation_node",
        config=mock_transformation_config
    )
    
    result = node.transform_data(mock_data)
    
    assert result["id"] == mock_data["id"]
    assert result["content"] == mock_data["content"].lower()
    assert "metadata" in result
    assert "timestamp" in result["metadata"]

def test_data_enrichment(mock_data, mock_enrichment_config):
    """Test data enrichment process."""
    node = DataEnrichmentNode(
        node_id="enrichment_node",
        config=mock_enrichment_config
    )
    
    result = node.enrich_data(mock_data)
    
    assert result["id"] == mock_data["id"]
    assert result["content"] == mock_data["content"]
    assert "metadata" in result
    assert "source_type" in result["metadata"]
    assert result["metadata"]["source_type"] == "document"

@pytest.mark.asyncio
async def test_async_data_processing(mock_data):
    """Test async data processing pipeline."""
    # Create nodes
    ingestion_node = DataIngestionNode(node_id="ingestion_node")
    validation_node = DataValidationNode(
        node_id="validation_node",
        config={"required_fields": ["id", "content"]}
    )
    transformation_node = DataTransformationNode(
        node_id="transformation_node",
        config={"transformations": [{"field": "content", "operation": "lowercase"}]}
    )
    enrichment_node = DataEnrichmentNode(
        node_id="enrichment_node",
        config={"enrichments": [{"field": "metadata", "operation": "add_source_info"}]}
    )
    
    # Process data through pipeline
    ingestion_node.set_input("data", mock_data)
    validation_result = await validation_node.process(ingestion_node.outputs)
    transformation_result = await transformation_node.process(validation_result)
    enrichment_result = await enrichment_node.process(transformation_result)
    
    assert enrichment_result["id"] == mock_data["id"]
    assert enrichment_result["content"] == mock_data["content"].lower()
    assert "metadata" in enrichment_result
    assert "source_type" in enrichment_result["metadata"]

def test_error_handling():
    """Test error handling in data processing nodes."""
    node = DataIngestionNode(node_id="test_node")
    
    # Test error setting
    node.set_error("Test error")
    assert node.error == "Test error"
    
    # Test error clearing
    node.clear_error()
    assert node.error is None

def test_node_state_management():
    """Test node state management."""
    node = DataIngestionNode(node_id="test_node")
    
    # Set state
    node.set_state("key", "value")
    assert node.state["key"] == "value"
    
    # Get state
    value = node.get_state("key")
    assert value == "value"
    
    # Update state
    node.update_state("key", "new_value")
    assert node.state["key"] == "new_value"

def test_node_serialization():
    """Test node serialization."""
    node = DataIngestionNode(node_id="test_node")
    node.set_input("input_key", "input_value")
    node.set_output("output_key", "output_value")
    node.set_state("state_key", "state_value")
    
    serialized = node.serialize()
    
    assert serialized["node_id"] == "test_node"
    assert serialized["inputs"]["input_key"] == "input_value"
    assert serialized["outputs"]["output_key"] == "output_value"
    assert serialized["state"]["state_key"] == "state_value"

def test_node_deserialization():
    """Test node deserialization."""
    serialized_data = {
        "node_id": "test_node",
        "inputs": {"input_key": "input_value"},
        "outputs": {"output_key": "output_value"},
        "state": {"state_key": "state_value"}
    }
    
    node = DataIngestionNode.deserialize(serialized_data)
    
    assert node.node_id == "test_node"
    assert node.inputs["input_key"] == "input_value"
    assert node.outputs["output_key"] == "output_value"
    assert node.state["state_key"] == "state_value" 