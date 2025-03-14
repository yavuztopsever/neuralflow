"""
Unit tests for the graph workflow components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import BaseModel

from src.core.graph.graph_workflow import GraphWorkflow
from src.core.graph.nodes import Node
from src.core.graph.edges import Edge
from src.core.graph.service_connections import ServiceConnection

@pytest.fixture
def mock_node():
    """Create a mock node for testing."""
    return Mock(spec=Node)

@pytest.fixture
def mock_edge():
    """Create a mock edge for testing."""
    return Mock(spec=Edge)

@pytest.fixture
def mock_service_connection():
    """Create a mock service connection for testing."""
    return Mock(spec=ServiceConnection)

@pytest.fixture
def graph_workflow():
    """Create a GraphWorkflow instance for testing."""
    return GraphWorkflow()

def test_graph_workflow_initialization(graph_workflow):
    """Test proper initialization of GraphWorkflow."""
    assert graph_workflow.nodes == {}
    assert graph_workflow.edges == {}
    assert graph_workflow.service_connections == {}

def test_add_node(graph_workflow, mock_node):
    """Test adding a node to the workflow."""
    node_id = "test_node"
    mock_node.node_id = node_id
    
    graph_workflow.add_node(node_id, mock_node)
    
    assert node_id in graph_workflow.nodes
    assert graph_workflow.nodes[node_id] == mock_node

def test_add_edge(graph_workflow, mock_edge):
    """Test adding an edge to the workflow."""
    edge_id = "test_edge"
    mock_edge.edge_id = edge_id
    
    graph_workflow.add_edge(edge_id, mock_edge)
    
    assert edge_id in graph_workflow.edges
    assert graph_workflow.edges[edge_id] == mock_edge

def test_add_service_connection(graph_workflow, mock_service_connection):
    """Test adding a service connection to the workflow."""
    connection_id = "test_connection"
    mock_service_connection.connection_id = connection_id
    
    graph_workflow.add_service_connection(connection_id, mock_service_connection)
    
    assert connection_id in graph_workflow.service_connections
    assert graph_workflow.service_connections[connection_id] == mock_service_connection

def test_remove_node(graph_workflow, mock_node):
    """Test removing a node from the workflow."""
    node_id = "test_node"
    mock_node.node_id = node_id
    graph_workflow.add_node(node_id, mock_node)
    
    graph_workflow.remove_node(node_id)
    
    assert node_id not in graph_workflow.nodes

def test_remove_edge(graph_workflow, mock_edge):
    """Test removing an edge from the workflow."""
    edge_id = "test_edge"
    mock_edge.edge_id = edge_id
    graph_workflow.add_edge(edge_id, mock_edge)
    
    graph_workflow.remove_edge(edge_id)
    
    assert edge_id not in graph_workflow.edges

def test_remove_service_connection(graph_workflow, mock_service_connection):
    """Test removing a service connection from the workflow."""
    connection_id = "test_connection"
    mock_service_connection.connection_id = connection_id
    graph_workflow.add_service_connection(connection_id, mock_service_connection)
    
    graph_workflow.remove_service_connection(connection_id)
    
    assert connection_id not in graph_workflow.service_connections

def test_get_node(graph_workflow, mock_node):
    """Test retrieving a node from the workflow."""
    node_id = "test_node"
    mock_node.node_id = node_id
    graph_workflow.add_node(node_id, mock_node)
    
    retrieved_node = graph_workflow.get_node(node_id)
    
    assert retrieved_node == mock_node

def test_get_edge(graph_workflow, mock_edge):
    """Test retrieving an edge from the workflow."""
    edge_id = "test_edge"
    mock_edge.edge_id = edge_id
    graph_workflow.add_edge(edge_id, mock_edge)
    
    retrieved_edge = graph_workflow.get_edge(edge_id)
    
    assert retrieved_edge == mock_edge

def test_get_service_connection(graph_workflow, mock_service_connection):
    """Test retrieving a service connection from the workflow."""
    connection_id = "test_connection"
    mock_service_connection.connection_id = connection_id
    graph_workflow.add_service_connection(connection_id, mock_service_connection)
    
    retrieved_connection = graph_workflow.get_service_connection(connection_id)
    
    assert retrieved_connection == mock_service_connection

def test_validate_workflow(graph_workflow, mock_node, mock_edge):
    """Test workflow validation."""
    # Add valid nodes and edges
    node1_id = "node1"
    node2_id = "node2"
    edge_id = "edge1"
    
    mock_node1 = Mock(spec=Node)
    mock_node1.node_id = node1_id
    mock_node2 = Mock(spec=Node)
    mock_node2.node_id = node2_id
    
    mock_edge.edge_id = edge_id
    mock_edge.source_node_id = node1_id
    mock_edge.target_node_id = node2_id
    
    graph_workflow.add_node(node1_id, mock_node1)
    graph_workflow.add_node(node2_id, mock_node2)
    graph_workflow.add_edge(edge_id, mock_edge)
    
    # Test validation
    is_valid = graph_workflow.validate_workflow()
    
    assert is_valid is True

def test_invalid_workflow(graph_workflow, mock_edge):
    """Test workflow validation with invalid configuration."""
    # Add edge without corresponding nodes
    edge_id = "edge1"
    mock_edge.edge_id = edge_id
    mock_edge.source_node_id = "nonexistent_node1"
    mock_edge.target_node_id = "nonexistent_node2"
    
    graph_workflow.add_edge(edge_id, mock_edge)
    
    # Test validation
    is_valid = graph_workflow.validate_workflow()
    
    assert is_valid is False

@pytest.mark.asyncio
async def test_execute_workflow(graph_workflow, mock_node):
    """Test workflow execution."""
    # Setup test nodes
    node1_id = "node1"
    node2_id = "node2"
    
    mock_node1 = Mock(spec=Node)
    mock_node1.node_id = node1_id
    mock_node1.execute = AsyncMock(return_value={"result": "node1_result"})
    
    mock_node2 = Mock(spec=Node)
    mock_node2.node_id = node2_id
    mock_node2.execute = AsyncMock(return_value={"result": "node2_result"})
    
    graph_workflow.add_node(node1_id, mock_node1)
    graph_workflow.add_node(node2_id, mock_node2)
    
    # Execute workflow
    result = await graph_workflow.execute()
    
    assert result is not None
    mock_node1.execute.assert_called_once()
    mock_node2.execute.assert_called_once() 