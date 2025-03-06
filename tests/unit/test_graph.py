"""
Unit tests for graph functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.core.graph.nodes import Node
from src.core.graph.edges import Edge
from src.core.graph.workflows import Workflow
from src.core.graph.engine import GraphEngine

class TestGraph:
    """Test suite for graph functionality."""
    
    @pytest.fixture
    def graph_engine(self):
        """Create a graph engine for testing."""
        return GraphEngine()
    
    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow for testing."""
        # Create nodes
        input_node = Node("input", "input", {})
        llm_node = Node("llm", "llm", {"provider": "openai"})
        memory_node = Node("memory", "memory", {})
        context_node = Node("context", "context", {})
        reasoning_node = Node("reasoning", "reasoning", {})
        output_node = Node("output", "output", {})
        
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
    async def test_node_operations(self, graph_engine):
        """Test node operations."""
        # Test node creation
        node = Node("test_node", "test_type", {"key": "value"})
        assert node.node_id == "test_node"
        assert node.node_type == "test_type"
        assert node.config == {"key": "value"}
        
        # Test node validation
        with pytest.raises(ValueError):
            Node("", "test_type", {})
        
        # Test node execution
        result = await node.execute({"input": "test"})
        assert result is not None
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_edge_operations(self, graph_engine):
        """Test edge operations."""
        # Test edge creation
        edge = Edge("test_edge", "source", "target", "data", {})
        assert edge.edge_id == "test_edge"
        assert edge.source_id == "source"
        assert edge.target_id == "target"
        assert edge.data_key == "data"
        
        # Test edge validation
        with pytest.raises(ValueError):
            Edge("", "source", "target", "data", {})
        
        # Test edge data transfer
        data = {"key": "value"}
        transferred_data = await edge.transfer_data(data)
        assert transferred_data is not None
        assert transferred_data == data
    
    @pytest.mark.asyncio
    async def test_workflow_operations(self, graph_engine, sample_workflow):
        """Test workflow operations."""
        # Test workflow creation
        assert sample_workflow.workflow_id == "test_workflow"
        assert len(sample_workflow.nodes) == 6
        assert len(sample_workflow.edges) == 5
        
        # Test workflow validation
        with pytest.raises(ValueError):
            Workflow("", "Invalid workflow", [], [], {})
        
        # Test workflow execution
        result = await graph_engine.execute_workflow(sample_workflow, {
            "input": "Test input data"
        })
        assert result is not None
        assert isinstance(result, dict)
        assert "output" in result
    
    @pytest.mark.asyncio
    async def test_graph_engine_operations(self, graph_engine, sample_workflow):
        """Test graph engine operations."""
        # Test workflow registration
        await graph_engine.register_workflow(sample_workflow)
        assert sample_workflow.workflow_id in graph_engine.workflows
        
        # Test workflow execution
        result = await graph_engine.execute_workflow(sample_workflow, {
            "input": "Test input data"
        })
        assert result is not None
        assert isinstance(result, dict)
        
        # Test workflow state management
        state = await graph_engine.get_workflow_state(sample_workflow)
        assert state is not None
        assert isinstance(state, str)
        
        # Test workflow cleanup
        await graph_engine.cleanup_workflow(sample_workflow)
        assert sample_workflow.workflow_id not in graph_engine.workflows
    
    @pytest.mark.asyncio
    async def test_graph_error_handling(self, graph_engine, sample_workflow):
        """Test graph error handling."""
        # Test invalid workflow execution
        with pytest.raises(ValueError):
            await graph_engine.execute_workflow(None, {})
        
        # Test invalid node execution
        with pytest.raises(ValueError):
            await Node("", "test_type", {}).execute({})
        
        # Test invalid edge data transfer
        with pytest.raises(ValueError):
            await Edge("", "source", "target", "data", {}).transfer_data(None)
    
    @pytest.mark.asyncio
    async def test_graph_metrics(self, graph_engine, sample_workflow):
        """Test graph metrics collection."""
        # Test collecting workflow metrics
        metrics = await graph_engine.collect_workflow_metrics(sample_workflow)
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "execution_time" in metrics
        assert "node_count" in metrics
        assert "edge_count" in metrics
        
        # Test collecting engine metrics
        engine_metrics = await graph_engine.collect_engine_metrics()
        assert engine_metrics is not None
        assert isinstance(engine_metrics, dict)
        assert "total_workflows" in engine_metrics
        assert "active_workflows" in engine_metrics
        assert "average_execution_time" in engine_metrics
    
    @pytest.mark.asyncio
    async def test_graph_optimization(self, graph_engine, sample_workflow):
        """Test graph optimization operations."""
        # Test workflow optimization
        optimization_params = {
            "parallel_execution": True,
            "caching_enabled": True,
            "max_concurrent_nodes": 3
        }
        
        optimized_workflow = await graph_engine.optimize_workflow(
            sample_workflow,
            optimization_params
        )
        assert optimized_workflow is not None
        assert optimized_workflow.is_optimized
        
        # Test engine optimization
        engine_params = {
            "max_workflows": 10,
            "cache_size": 1000,
            "timeout": 30
        }
        
        optimized_engine = await graph_engine.optimize(engine_params)
        assert optimized_engine is not None
        assert optimized_engine.is_optimized 