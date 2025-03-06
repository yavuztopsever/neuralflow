#!/usr/bin/env python3
"""
Acceptance tests for the LangGraph system.
Tests system requirements and user stories.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List

from src.config.manager import ConfigManager
from src.core.graph.workflows import Workflow
from src.core.graph.nodes import Node
from src.core.graph.edges import Edge
from src.core.engine.manager import EngineManager
from src.memory.manager import MemoryManager
from src.context.manager import ContextManager
from src.response.manager import ResponseManager
from src.utils.logging.manager import LogManager
from src.core.graph.engine import GraphEngine
from src.core.nodes.input import InputNode
from src.core.nodes.llm import LLMNode
from src.core.nodes.output import OutputNode
from src.core.nodes.memory import MemoryNode
from src.core.nodes.context import ContextNode
from src.core.nodes.reasoning import ReasoningNode

class TestAcceptance:
    """Test suite for system requirements and user stories."""
    
    @pytest.fixture
    def config(self):
        """Initialize configuration for testing."""
        return ConfigManager()
    
    @pytest.fixture
    async def system_components(self, config):
        """Initialize system components for acceptance testing."""
        components = {}
        
        # Initialize core components
        components["config"] = config
        components["log_manager"] = LogManager(config)
        components["engine_manager"] = EngineManager(config)
        components["memory_manager"] = MemoryManager(config)
        components["context_manager"] = ContextManager(config)
        components["response_manager"] = ResponseManager(config)
        
        return components
    
    @pytest.fixture
    def workflow(self):
        """Create a workflow for testing."""
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
    async def test_workflow_initialization(self, system_components):
        """Test workflow initialization and basic structure."""
        # Create a simple workflow
        input_node = Node("input", "input", {})
        llm_node = Node("llm", "llm", {"provider": "openai"})
        output_node = Node("output", "output", {})
        
        edges = [
            Edge("edge1", "input", "llm", "prompt", {}),
            Edge("edge2", "llm", "output", "results", {})
        ]
        
        workflow = Workflow(
            "test_workflow",
            "Test workflow",
            [input_node, llm_node, output_node],
            edges,
            {}
        )
        
        assert workflow.name == "test_workflow"
        assert len(workflow.nodes) == 3
        assert len(workflow.edges) == 2
    
    @pytest.mark.asyncio
    async def test_memory_management(self, system_components):
        """Test memory management system functionality."""
        memory_manager = system_components["memory_manager"]
        
        # Test long-term memory
        test_data = {"type": "long_term", "content": "Important information"}
        await memory_manager.store_long_term(test_data)
        long_term_data = await memory_manager.retrieve_long_term()
        assert len(long_term_data) > 0
        
        # Test mid-term memory
        test_data = {"type": "mid_term", "content": "Recent information"}
        await memory_manager.store_mid_term(test_data)
        mid_term_data = await memory_manager.retrieve_mid_term()
        assert len(mid_term_data) > 0
        
        # Test short-term memory
        test_data = {"type": "short_term", "content": "Current information"}
        await memory_manager.store_short_term(test_data)
        short_term_data = await memory_manager.retrieve_short_term()
        assert len(short_term_data) > 0
    
    @pytest.mark.asyncio
    async def test_context_management(self, system_components):
        """Test context management system functionality."""
        context_manager = system_components["context_manager"]
        
        # Test context aggregation
        test_context = {
            "long_term": ["Important background"],
            "mid_term": ["Recent information"],
            "short_term": ["Current context"]
        }
        aggregated_context = await context_manager.aggregate_context(test_context)
        assert len(aggregated_context) > 0
        
        # Test context summarization
        summary = await context_manager.summarize_context(aggregated_context)
        assert len(summary) > 0
        
        # Test emotion context generation
        emotion_context = await context_manager.generate_emotion_context(aggregated_context)
        assert emotion_context is not None
    
    @pytest.mark.asyncio
    async def test_response_generation(self, system_components):
        """Test response generation system functionality."""
        response_manager = system_components["response_manager"]
        
        # Test response assembly
        test_data = {
            "user_input": "What is LangGraph?",
            "model_response": "LangGraph is a workflow system",
            "reasoning_steps": ["Step 1", "Step 2"]
        }
        response = await response_manager.assemble_response(test_data)
        assert response is not None
        
        # Test response styling
        styled_response = await response_manager.style_response(response)
        assert styled_response is not None
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, system_components):
        """Test complete workflow execution."""
        engine_manager = system_components["engine_manager"]
        
        # Create a test workflow
        input_node = Node("input", "input", {})
        llm_node = Node("llm", "llm", {"provider": "openai"})
        output_node = Node("output", "output", {})
        
        edges = [
            Edge("edge1", "input", "llm", "prompt", {}),
            Edge("edge2", "llm", "output", "results", {})
        ]
        
        workflow = Workflow(
            "complete_workflow",
            "Complete workflow test",
            [input_node, llm_node, output_node],
            edges,
            {}
        )
        
        # Execute workflow
        result = await engine_manager.execute_workflow(
            workflow,
            {"input": "What is LangGraph?"}
        )
        
        assert result is not None
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, system_components):
        """Test system error handling."""
        engine_manager = system_components["engine_manager"]
        
        # Test invalid workflow
        with pytest.raises(Exception):
            await engine_manager.execute_workflow(None, {})
        
        # Test invalid input
        with pytest.raises(Exception):
            await engine_manager.execute_workflow(
                Workflow("test", "test", [], [], {}),
                None
            )
    
    @pytest.mark.asyncio
    async def test_basic_workflow_execution(self, workflow):
        """Test basic workflow execution."""
        engine = GraphEngine()
        result = await engine.execute_workflow(workflow, {
            "input": "Test input data"
        })
        
        assert result is not None
        assert isinstance(result, dict)
        assert "output" in result
        assert len(result["output"]) > 0
    
    @pytest.mark.asyncio
    async def test_memory_management_in_workflow(self, workflow):
        """Test memory management in workflow."""
        engine = GraphEngine()
        
        # Execute workflow with memory
        result1 = await engine.execute_workflow(workflow, {
            "input": "First test input"
        })
        
        # Execute workflow with memory context
        result2 = await engine.execute_workflow(workflow, {
            "input": "Second test input"
        })
        
        assert result1 is not None
        assert result2 is not None
        assert result1["output"] != result2["output"]
    
    @pytest.mark.asyncio
    async def test_context_management_in_workflow(self, workflow):
        """Test context management in workflow."""
        engine = GraphEngine()
        
        # Execute workflow with context
        result = await engine.execute_workflow(workflow, {
            "input": "Test input with context",
            "context": {
                "user_id": "123",
                "session_id": "456"
            }
        })
        
        assert result is not None
        assert isinstance(result, dict)
        assert "output" in result
        assert "context" in result
    
    @pytest.mark.asyncio
    async def test_reasoning_engine_in_workflow(self, workflow):
        """Test reasoning engine in workflow."""
        engine = GraphEngine()
        
        # Execute workflow with reasoning
        result = await engine.execute_workflow(workflow, {
            "input": "Test input requiring reasoning",
            "reasoning_params": {
                "max_steps": 5,
                "confidence_threshold": 0.8
            }
        })
        
        assert result is not None
        assert isinstance(result, dict)
        assert "output" in result
        assert "reasoning_steps" in result
        assert len(result["reasoning_steps"]) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self, workflow):
        """Test error handling in workflow."""
        engine = GraphEngine()
        
        # Test invalid input
        with pytest.raises(ValueError):
            await engine.execute_workflow(workflow, {})
        
        # Test invalid node
        with pytest.raises(ValueError):
            await engine.execute_workflow(workflow, {
                "input": None
            })
        
        # Test workflow failure
        with pytest.raises(Exception):
            await engine.execute_workflow(workflow, {
                "input": "Test input causing failure"
            })
    
    @pytest.mark.asyncio
    async def test_performance_of_workflow(self, workflow):
        """Test workflow performance."""
        engine = GraphEngine()
        
        # Test execution time
        start_time = asyncio.get_event_loop().time()
        result = await engine.execute_workflow(workflow, {
            "input": "Test input for performance"
        })
        end_time = asyncio.get_event_loop().time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Test memory usage
        metrics = await engine.collect_metrics()
        assert metrics is not None
        assert "memory_usage" in metrics
        assert metrics["memory_usage"] < 1000  # Should use less than 1000MB
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_of_workflows(self, workflow):
        """Test concurrent workflow execution."""
        engine = GraphEngine()
        
        # Execute multiple workflows concurrently
        tasks = [
            engine.execute_workflow(workflow, {
                "input": f"Test input {i}"
            })
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r is not None for r in results)
        assert all("output" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_state_management_of_workflow(self, workflow):
        """Test workflow state management."""
        engine = GraphEngine()
        
        # Test workflow state transitions
        initial_state = await engine.get_workflow_state(workflow)
        assert initial_state == "initialized"
        
        # Execute workflow
        result = await engine.execute_workflow(workflow, {
            "input": "Test input"
        })
        
        final_state = await engine.get_workflow_state(workflow)
        assert final_state == "completed"
        
        # Test state persistence
        await engine.persist_workflow_state(workflow)
        assert await engine.check_workflow_state_persistence(workflow)
    
    @pytest.mark.asyncio
    async def test_optimization_of_workflow(self, workflow):
        """Test workflow optimization."""
        engine = GraphEngine()
        
        # Test workflow optimization
        optimization_params = {
            "parallel_execution": True,
            "caching_enabled": True,
            "max_concurrent_nodes": 3
        }
        
        optimized_workflow = await engine.optimize_workflow(workflow, optimization_params)
        assert optimized_workflow is not None
        assert optimized_workflow.is_optimized
        
        # Test optimized execution
        result = await engine.execute_workflow(optimized_workflow, {
            "input": "Test input for optimization"
        })
        
        assert result is not None
        assert isinstance(result, dict)
        assert "output" in result
        assert "optimization_metrics" in result 