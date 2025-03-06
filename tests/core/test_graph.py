"""
Tests for the graph component of the LangGraph framework.
"""

import pytest
from langgraph.core.graph import Graph, Node, Edge
from langgraph.core.exceptions import GraphError

@pytest.fixture
def empty_graph():
    """Fixture providing an empty graph."""
    return Graph()

@pytest.fixture
def sample_graph():
    """Fixture providing a graph with sample nodes and edges."""
    graph = Graph()
    graph.add_node(Node("input"))
    graph.add_node(Node("process"))
    graph.add_node(Node("output"))
    graph.add_edge(Edge("input", "process"))
    graph.add_edge(Edge("process", "output"))
    return graph

class TestGraphCreation:
    """Test cases for graph creation and basic operations."""
    
    def test_create_empty_graph(self, empty_graph):
        """Test creating an empty graph."""
        assert len(empty_graph.nodes) == 0
        assert len(empty_graph.edges) == 0
    
    def test_add_node(self, empty_graph):
        """Test adding a node to the graph."""
        node = Node("test")
        empty_graph.add_node(node)
        assert len(empty_graph.nodes) == 1
        assert empty_graph.nodes[0].id == "test"
    
    def test_add_duplicate_node(self, empty_graph):
        """Test adding a duplicate node raises an error."""
        node = Node("test")
        empty_graph.add_node(node)
        with pytest.raises(GraphError):
            empty_graph.add_node(node)

class TestGraphOperations:
    """Test cases for graph operations."""
    
    def test_add_edge(self, sample_graph):
        """Test adding an edge between existing nodes."""
        edge = Edge("input", "output")
        sample_graph.add_edge(edge)
        assert len(sample_graph.edges) == 3
    
    def test_add_invalid_edge(self, sample_graph):
        """Test adding an edge with non-existent nodes."""
        with pytest.raises(GraphError):
            sample_graph.add_edge(Edge("invalid", "output"))
    
    def test_remove_node(self, sample_graph):
        """Test removing a node and its associated edges."""
        sample_graph.remove_node("process")
        assert len(sample_graph.nodes) == 2
        assert len(sample_graph.edges) == 0
    
    def test_get_node_connections(self, sample_graph):
        """Test getting connections for a node."""
        connections = sample_graph.get_node_connections("process")
        assert len(connections) == 2
        assert "input" in connections
        assert "output" in connections

class TestGraphValidation:
    """Test cases for graph validation."""
    
    def test_validate_acyclic(self, sample_graph):
        """Test validating an acyclic graph."""
        assert sample_graph.is_acyclic()
    
    def test_validate_cyclic(self, sample_graph):
        """Test detecting a cyclic graph."""
        sample_graph.add_edge(Edge("output", "input"))
        assert not sample_graph.is_acyclic()
    
    def test_validate_connected(self, sample_graph):
        """Test validating a connected graph."""
        assert sample_graph.is_connected()
    
    def test_validate_disconnected(self, sample_graph):
        """Test detecting a disconnected graph."""
        sample_graph.add_node(Node("isolated"))
        assert not sample_graph.is_connected()

@pytest.mark.parametrize("node_id", ["input", "process", "output"])
def test_node_exists(sample_graph, node_id):
    """Test checking if a node exists in the graph."""
    assert sample_graph.has_node(node_id)

@pytest.mark.parametrize("node_id", ["invalid1", "invalid2"])
def test_node_not_exists(sample_graph, node_id):
    """Test checking if a non-existent node is not in the graph."""
    assert not sample_graph.has_node(node_id) 