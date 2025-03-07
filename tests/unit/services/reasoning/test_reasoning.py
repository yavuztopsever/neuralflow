"""
Unit tests for reasoning engine functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List

from src.core.nodes.reasoning import ReasoningNode
from src.core.nodes.context import ContextNode

class TestReasoningEngine:
    """Test suite for reasoning engine functionality."""
    
    @pytest.fixture
    def reasoning_node(self):
        """Create a reasoning node for testing."""
        return ReasoningNode("test_reasoning", "reasoning", {})
    
    @pytest.fixture
    def context_node(self):
        """Create a context node for testing."""
        return ContextNode("test_context", "context", {})
    
    @pytest.mark.asyncio
    async def test_reasoning_process(self, reasoning_node):
        """Test reasoning process operations."""
        # Test processing reasoning steps
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        reasoning_result = await reasoning_node.process(context_data)
        assert reasoning_result is not None
        assert isinstance(reasoning_result, dict)
        assert "reasoning_steps" in reasoning_result
        assert "conclusion" in reasoning_result
    
    @pytest.mark.asyncio
    async def test_reasoning_steps(self, reasoning_node):
        """Test reasoning steps generation."""
        # Test generating reasoning steps
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        steps = await reasoning_node.generate_steps(context_data)
        assert steps is not None
        assert isinstance(steps, list)
        assert len(steps) > 0
        assert all(isinstance(step, dict) for step in steps)
        assert all("step" in step and "reasoning" in step for step in steps)
    
    @pytest.mark.asyncio
    async def test_reasoning_conclusion(self, reasoning_node):
        """Test reasoning conclusion generation."""
        # Test generating conclusion
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        conclusion = await reasoning_node.generate_conclusion(context_data)
        assert conclusion is not None
        assert isinstance(conclusion, dict)
        assert "conclusion" in conclusion
        assert "confidence" in conclusion
    
    @pytest.mark.asyncio
    async def test_reasoning_validation(self, reasoning_node):
        """Test reasoning validation operations."""
        # Test validating reasoning
        valid_reasoning = {
            "reasoning_steps": [
                {"step": 1, "reasoning": "test reasoning"},
                {"step": 2, "reasoning": "test reasoning"}
            ],
            "conclusion": {"conclusion": "test conclusion", "confidence": 0.8}
        }
        
        is_valid = await reasoning_node.validate_reasoning(valid_reasoning)
        assert is_valid
        
        # Test invalid reasoning
        invalid_reasoning = {
            "reasoning_steps": [],
            "conclusion": {"conclusion": "", "confidence": 0.0}
        }
        
        is_valid = await reasoning_node.validate_reasoning(invalid_reasoning)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_reasoning_error_handling(self, reasoning_node):
        """Test reasoning error handling."""
        # Test handling invalid context data
        with pytest.raises(ValueError):
            await reasoning_node.process(None)
        
        # Test handling missing required fields
        invalid_context = {"current_input": "test"}
        with pytest.raises(ValueError):
            await reasoning_node.process(invalid_context)
    
    @pytest.mark.asyncio
    async def test_reasoning_metrics(self, reasoning_node):
        """Test reasoning metrics collection."""
        # Test collecting reasoning metrics
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        metrics = await reasoning_node.collect_metrics(context_data)
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "step_count" in metrics
        assert "processing_time" in metrics
        assert "confidence_score" in metrics
    
    @pytest.mark.asyncio
    async def test_reasoning_optimization(self, reasoning_node):
        """Test reasoning optimization operations."""
        # Test optimizing reasoning process
        context_data = {
            "memory_data": {"key": "value"},
            "current_input": "test input",
            "session_data": {"user_id": "123"}
        }
        
        optimization_params = {
            "max_steps": 5,
            "min_confidence": 0.7,
            "timeout": 10
        }
        
        optimized_result = await reasoning_node.optimize_reasoning(context_data, optimization_params)
        assert optimized_result is not None
        assert isinstance(optimized_result, dict)
        assert "reasoning_steps" in optimized_result
        assert "conclusion" in optimized_result
        assert len(optimized_result["reasoning_steps"]) <= optimization_params["max_steps"]
        assert optimized_result["conclusion"]["confidence"] >= optimization_params["min_confidence"] 