"""
Shared fixtures for workflow testing.
"""
import pytest
from src.core.graph.nodes import Node
from src.core.graph.edges import Edge
from src.core.graph.workflows import Workflow
from src.core.nodes.input import InputNode
from src.core.nodes.llm import LLMNode
from src.core.nodes.output import OutputNode
from src.core.nodes.memory import MemoryNode
from src.core.nodes.context import ContextNode
from src.core.nodes.reasoning import ReasoningNode

@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    # Create nodes
    input_node = InputNode("input", "input", {})
    llm_node = LLMNode("llm", "llm", {"provider": "openai"})
    memory_node = MemoryNode("memory", "memory", {})
    context_node = ContextNode("context", "context", {})
    reasoning_node = ReasoningNode("reasoning", "reasoning", {})
    output_node = OutputNode("output", "output", {})
    
    # Create edges
    edges = [
        Edge("edge1", "input", "llm", "prompt", {}),
        Edge("edge2", "llm", "memory", "response", {}),
        Edge("edge3", "memory", "context", "memory_data", {}),
        Edge("edge4", "context", "reasoning", "context_data", {}),
        Edge("edge5", "reasoning", "output", "reasoning_results", {})
    ]
    
    # Create workflow
    return Workflow(
        "test_workflow",
        "A test workflow",
        [input_node, llm_node, memory_node, context_node, reasoning_node, output_node],
        edges,
        {}
    )

@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing."""
    return {
        "input": InputNode("test_input", "input", {}),
        "llm": LLMNode("test_llm", "llm", {"provider": "openai"}),
        "memory": MemoryNode("test_memory", "memory", {}),
        "context": ContextNode("test_context", "context", {}),
        "reasoning": ReasoningNode("test_reasoning", "reasoning", {}),
        "output": OutputNode("test_output", "output", {})
    }

@pytest.fixture
def sample_edge():
    """Create a sample edge for testing."""
    return Edge("test_edge", "source", "target", "data", {}) 