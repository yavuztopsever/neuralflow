"""
Unit tests for the workflow nodes components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import BaseModel

from src.core.graph.workflow_nodes import (
    WorkflowNode,
    InputNode,
    OutputNode,
    ProcessingNode,
    DecisionNode,
    LoopNode
)

@pytest.fixture
def mock_input_data():
    """Create mock input data for testing."""
    return {
        "data": "test_data",
        "metadata": {"source": "test"}
    }

@pytest.fixture
def mock_processing_config():
    """Create mock processing configuration."""
    return {
        "operation": "transform",
        "parameters": {"key": "value"}
    }

@pytest.fixture
def mock_decision_config():
    """Create mock decision configuration."""
    return {
        "condition": "value > 10",
        "branches": ["branch1", "branch2"]
    }

@pytest.fixture
def mock_loop_config():
    """Create mock loop configuration."""
    return {
        "max_iterations": 3,
        "condition": "counter < max_iterations"
    }

def test_workflow_node_initialization():
    """Test initialization of base WorkflowNode."""
    node = WorkflowNode(node_id="test_node")
    assert node.node_id == "test_node"
    assert node.inputs == {}
    assert node.outputs == {}
    assert node.state == {}

def test_input_node_processing(mock_input_data):
    """Test InputNode processing."""
    node = InputNode(node_id="input_node")
    node.set_input("data", mock_input_data)
    
    result = node.process()
    
    assert result == mock_input_data
    assert node.outputs["data"] == mock_input_data

def test_output_node_processing(mock_input_data):
    """Test OutputNode processing."""
    node = OutputNode(node_id="output_node")
    node.set_input("data", mock_input_data)
    
    result = node.process()
    
    assert result == mock_input_data
    assert node.outputs["data"] == mock_input_data

def test_processing_node_processing(mock_input_data, mock_processing_config):
    """Test ProcessingNode processing."""
    node = ProcessingNode(
        node_id="processing_node",
        config=mock_processing_config
    )
    node.set_input("data", mock_input_data)
    
    result = node.process()
    
    assert result is not None
    assert "processed_data" in node.outputs

def test_decision_node_processing(mock_input_data, mock_decision_config):
    """Test DecisionNode processing."""
    node = DecisionNode(
        node_id="decision_node",
        config=mock_decision_config
    )
    node.set_input("data", mock_input_data)
    
    result = node.process()
    
    assert result is not None
    assert "branch" in node.outputs

def test_loop_node_processing(mock_input_data, mock_loop_config):
    """Test LoopNode processing."""
    node = LoopNode(
        node_id="loop_node",
        config=mock_loop_config
    )
    node.set_input("data", mock_input_data)
    
    result = node.process()
    
    assert result is not None
    assert "iterations" in node.outputs

def test_node_state_management():
    """Test node state management."""
    node = WorkflowNode(node_id="test_node")
    
    # Set state
    node.set_state("key", "value")
    assert node.state["key"] == "value"
    
    # Get state
    value = node.get_state("key")
    assert value == "value"
    
    # Update state
    node.update_state("key", "new_value")
    assert node.state["key"] == "new_value"

def test_node_input_output_management():
    """Test node input/output management."""
    node = WorkflowNode(node_id="test_node")
    
    # Set input
    node.set_input("input_key", "input_value")
    assert node.inputs["input_key"] == "input_value"
    
    # Set output
    node.set_output("output_key", "output_value")
    assert node.outputs["output_key"] == "output_value"
    
    # Get input
    input_value = node.get_input("input_key")
    assert input_value == "input_value"
    
    # Get output
    output_value = node.get_output("output_key")
    assert output_value == "output_value"

def test_node_validation():
    """Test node validation."""
    node = WorkflowNode(node_id="test_node")
    
    # Test required inputs
    node.required_inputs = ["required_input"]
    assert not node.validate()
    
    node.set_input("required_input", "value")
    assert node.validate()

def test_node_error_handling():
    """Test node error handling."""
    node = WorkflowNode(node_id="test_node")
    
    # Test error setting
    node.set_error("Test error")
    assert node.error == "Test error"
    
    # Test error clearing
    node.clear_error()
    assert node.error is None

@pytest.mark.asyncio
async def test_async_node_processing():
    """Test async node processing."""
    class AsyncWorkflowNode(WorkflowNode):
        async def process(self):
            return {"result": "async_result"}
    
    node = AsyncWorkflowNode(node_id="async_node")
    result = await node.process()
    
    assert result == {"result": "async_result"}

def test_node_serialization():
    """Test node serialization."""
    node = WorkflowNode(node_id="test_node")
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
    
    node = WorkflowNode.deserialize(serialized_data)
    
    assert node.node_id == "test_node"
    assert node.inputs["input_key"] == "input_value"
    assert node.outputs["output_key"] == "output_value"
    assert node.state["state_key"] == "state_value" 