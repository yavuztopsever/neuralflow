"""
Integration tests for the LangGraph system.
"""
import pytest
import asyncio
from datetime import datetime
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
from src.core.state import State
from src.core.events import EventBus
from src.core.metrics import MetricsCollector

class TestIntegration:
    """Test suite for system integration."""
    
    @pytest.fixture
    async def system_components(self):
        """Create system components for testing."""
        # Create core components
        state = State()
        event_bus = EventBus()
        metrics_collector = MetricsCollector()
        
        # Create workflow components
        input_node = InputNode("input", "input", {})
        llm_node = LLMNode("llm", "llm", {"provider": "openai"})
        memory_node = MemoryNode("memory", "memory", {})
        context_node = ContextNode("context", "context", {})
        reasoning_node = ReasoningNode("reasoning", "reasoning", {})
        output_node = OutputNode("output", "output", {})
        
        # Create workflow
        edges = [
            Edge("edge1", "input", "llm", "prompt", {}),
            Edge("edge2", "llm", "memory", "response", {}),
            Edge("edge3", "memory", "context", "memory_data", {}),
            Edge("edge4", "context", "reasoning", "context_data", {}),
            Edge("edge5", "reasoning", "output", "reasoning_results", {})
        ]
        
        workflow = Workflow(
            "test_workflow",
            "A test workflow",
            [input_node, llm_node, memory_node, context_node, reasoning_node, output_node],
            edges,
            {}
        )
        
        # Create engine
        engine = WorkflowEngine()
        
        return {
            "state": state,
            "event_bus": event_bus,
            "metrics_collector": metrics_collector,
            "workflow": workflow,
            "engine": engine
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, system_components):
        """Test complete workflow execution with all components."""
        engine = system_components["engine"]
        workflow = system_components["workflow"]
        
        # Execute workflow
        start_time = datetime.now()
        result = await engine.execute_workflow(workflow, {
            "input": "Test input data"
        })
        end_time = datetime.now()
        
        # Verify results
        assert result is not None
        assert "output" in result
        assert len(result["output"]) > 0
        
        # Verify execution time
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, system_components):
        """Test system metrics collection."""
        engine = system_components["engine"]
        workflow = system_components["workflow"]
        metrics_collector = system_components["metrics_collector"]
        
        # Execute workflow
        await engine.execute_workflow(workflow, {
            "input": "Test input data"
        })
        
        # Collect metrics
        metrics = await metrics_collector.get_metrics()
        
        # Verify metrics
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "workflow_executions" in metrics
        assert "average_execution_time" in metrics
        assert "memory_usage" in metrics
        assert metrics["memory_usage"] < 1000  # Should use less than 1000MB
    
    @pytest.mark.asyncio
    async def test_event_bus_integration(self, system_components):
        """Test event bus integration with workflow execution."""
        engine = system_components["engine"]
        workflow = system_components["workflow"]
        event_bus = system_components["event_bus"]
        
        # Subscribe to events
        events = []
        async def event_handler(event_type, data):
            events.append((event_type, data))
        
        await event_bus.subscribe("workflow_started", event_handler)
        await event_bus.subscribe("workflow_completed", event_handler)
        await event_bus.subscribe("node_started", event_handler)
        await event_bus.subscribe("node_completed", event_handler)
        
        # Execute workflow
        await engine.execute_workflow(workflow, {
            "input": "Test input data"
        })
        
        # Verify events
        assert len(events) > 0
        assert any(event[0] == "workflow_started" for event in events)
        assert any(event[0] == "workflow_completed" for event in events)
        assert any(event[0] == "node_started" for event in events)
        assert any(event[0] == "node_completed" for event in events)
    
    @pytest.mark.asyncio
    async def test_state_management(self, system_components):
        """Test state management during workflow execution."""
        engine = system_components["engine"]
        workflow = system_components["workflow"]
        state = system_components["state"]
        
        # Execute workflow
        await engine.execute_workflow(workflow, {
            "input": "Test input data"
        })
        
        # Verify state
        workflow_state = await state.get_workflow_state(workflow.workflow_id)
        assert workflow_state is not None
        assert workflow_state["status"] == "completed"
        assert "start_time" in workflow_state
        assert "end_time" in workflow_state
        assert "execution_time" in workflow_state
    
    @pytest.mark.asyncio
    async def test_error_handling(self, system_components):
        """Test error handling in integrated system."""
        engine = system_components["engine"]
        workflow = system_components["workflow"]
        event_bus = system_components["event_bus"]
        state = system_components["state"]
        
        # Test invalid input
        with pytest.raises(ValueError):
            await engine.execute_workflow(workflow, {})
        
        # Test invalid workflow
        with pytest.raises(ValueError):
            await engine.execute_workflow(None, {"input": "test"})
        
        # Test state error
        with pytest.raises(ValueError):
            await state.update(None)
        
        # Test event error
        with pytest.raises(ValueError):
            await event_bus.emit(None, {}) 