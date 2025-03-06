#!/usr/bin/env python3
"""
Unit tests for the LangGraph system core components.
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

class TestCoreComponents:
    """Test suite for core system components."""
    
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
    async def test_node_creation(self):
        """Test node creation and initialization."""
        # Test input node
        input_node = InputNode("test_input", "input", {})
        assert input_node.node_id == "test_input"
        assert input_node.node_type == "input"
        
        # Test LLM node
        llm_node = LLMNode("test_llm", "llm", {"provider": "openai"})
        assert llm_node.node_id == "test_llm"
        assert llm_node.node_type == "llm"
        assert llm_node.config["provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_edge_creation(self):
        """Test edge creation and validation."""
        # Test valid edge
        edge = Edge("test_edge", "source", "target", "data", {})
        assert edge.edge_id == "test_edge"
        assert edge.source_id == "source"
        assert edge.target_id == "target"
        assert edge.data_key == "data"
        
        # Test edge validation
        with pytest.raises(ValueError):
            Edge("invalid_edge", "", "target", "data", {})
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self, sample_workflow):
        """Test workflow creation and validation."""
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
    async def test_memory_operations(self):
        """Test memory node operations."""
        memory_node = MemoryNode("test_memory", "memory", {})
        
        # Test memory storage
        await memory_node.store("test_key", "test_value")
        
        # Test memory retrieval
        value = await memory_node.retrieve("test_key")
        assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_context_operations(self):
        """Test context node operations."""
        context_node = ContextNode("test_context", "context", {})
        
        # Test context aggregation
        context_data = await context_node.aggregate_context({
            "memory_data": {"key": "value"},
            "current_input": "test input"
        })
        
        assert context_data is not None
        assert "memory_data" in context_data
        assert "current_input" in context_data
    
    @pytest.mark.asyncio
    async def test_reasoning_operations(self):
        """Test reasoning node operations."""
        reasoning_node = ReasoningNode("test_reasoning", "reasoning", {})
        
        # Test reasoning process
        reasoning_result = await reasoning_node.process({
            "context_data": {"key": "value"},
            "query": "test query"
        })
        
        assert reasoning_result is not None
        assert "reasoning_steps" in reasoning_result
        assert "conclusion" in reasoning_result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_workflow):
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