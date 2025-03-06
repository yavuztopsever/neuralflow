"""
Unit tests for core LangGraph functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.core.graph.nodes import Node
from src.core.graph.edges import Edge
from src.core.graph.workflows import Workflow
from src.core.engine.engine import WorkflowEngine
from src.core.nodes.input import InputNode
from src.core.nodes.llm import LLMNode
from src.core.nodes.output import OutputNode
from src.core.nodes.memory import MemoryNode
from src.core.nodes.context import ContextNode
from src.core.nodes.reasoning import ReasoningNode

class TestCoreWorkflow:
    """Test suite for core workflow functionality."""
    
    @pytest.fixture
    def sample_workflow(self):
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
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, sample_workflow):
        """Test workflow initialization and validation."""
        assert sample_workflow.workflow_id == "test_workflow"
        assert len(sample_workflow.nodes) == 6
        assert len(sample_workflow.edges) == 5
        
        # Test workflow validation
        with pytest.raises(ValueError):
            Workflow("", "Invalid workflow", [], [], {})
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, sample_workflow):
        """Test workflow execution flow."""
        engine = WorkflowEngine()
        result = await engine.execute_workflow(sample_workflow, {
            "input": "Test input data"
        })
        
        assert result is not None
        assert "output" in result
    
    @pytest.mark.asyncio
    async def test_workflow_state_management(self, sample_workflow):
        """Test workflow state management."""
        engine = WorkflowEngine()
        
        # Test workflow state transitions
        initial_state = await engine.get_workflow_state(sample_workflow)
        assert initial_state == "initialized"
        
        # Execute workflow
        await engine.execute_workflow(sample_workflow, {
            "input": "test input"
        })
        
        final_state = await engine.get_workflow_state(sample_workflow)
        assert final_state == "completed"
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, sample_workflow):
        """Test error handling in workflow execution."""
        engine = WorkflowEngine()
        
        # Test invalid input handling
        with pytest.raises(ValueError):
            await engine.execute_workflow(sample_workflow, {})
        
        # Test node failure handling
        with pytest.raises(Exception):
            await engine.execute_workflow(sample_workflow, {
                "input": None
            }) 