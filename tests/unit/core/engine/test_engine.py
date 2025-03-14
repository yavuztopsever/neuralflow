"""
Unit tests for the engine components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import BaseModel
import asyncio

from src.core.engine.cod_reasoning import CodeReasoningEngine
from src.core.engine.executor import Executor
from src.core.engine.scheduler import Scheduler
from src.core.engine.task_executor import TaskExecutor

@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    return {
        "task_id": "test_task",
        "type": "code_execution",
        "parameters": {
            "code": "print('Hello, World!')",
            "language": "python"
        },
        "priority": 1,
        "dependencies": []
    }

@pytest.fixture
def mock_execution_context():
    """Create a mock execution context."""
    return {
        "environment": "test_env",
        "variables": {"var1": "value1"},
        "resources": {"cpu": 1, "memory": "1GB"}
    }

def test_code_reasoning_engine_initialization():
    """Test initialization of CodeReasoningEngine."""
    engine = CodeReasoningEngine()
    assert engine.context == {}
    assert engine.reasoning_history == []
    assert engine.current_task is None

def test_code_reasoning_engine_context_management():
    """Test context management in CodeReasoningEngine."""
    engine = CodeReasoningEngine()
    
    # Set context
    engine.set_context("key", "value")
    assert engine.context["key"] == "value"
    
    # Get context
    value = engine.get_context("key")
    assert value == "value"
    
    # Update context
    engine.update_context("key", "new_value")
    assert engine.context["key"] == "new_value"
    
    # Clear context
    engine.clear_context()
    assert engine.context == {}

def test_code_reasoning_engine_reasoning():
    """Test code reasoning process."""
    engine = CodeReasoningEngine()
    
    # Test reasoning with code
    code = """
def add(a, b):
    return a + b
    """
    result = engine.reason_about_code(code)
    
    assert result is not None
    assert "analysis" in result
    assert "suggestions" in result
    assert "complexity" in result

def test_executor_initialization():
    """Test initialization of Executor."""
    executor = Executor()
    assert executor.tasks == []
    assert executor.results == {}
    assert executor.errors == {}

@pytest.mark.asyncio
async def test_executor_task_execution(mock_task, mock_execution_context):
    """Test task execution in Executor."""
    executor = Executor()
    
    # Add task
    executor.add_task(mock_task)
    assert len(executor.tasks) == 1
    
    # Execute task
    result = await executor.execute_task(mock_task["task_id"], mock_execution_context)
    
    assert result is not None
    assert mock_task["task_id"] in executor.results
    assert executor.results[mock_task["task_id"]] == result

def test_scheduler_initialization():
    """Test initialization of Scheduler."""
    scheduler = Scheduler()
    assert scheduler.tasks == []
    assert scheduler.schedule == {}
    assert scheduler.running is False

@pytest.mark.asyncio
async def test_scheduler_task_scheduling(mock_task):
    """Test task scheduling in Scheduler."""
    scheduler = Scheduler()
    
    # Add task
    scheduler.add_task(mock_task)
    assert len(scheduler.tasks) == 1
    
    # Schedule task
    schedule_time = asyncio.get_event_loop().time() + 1.0
    scheduler.schedule_task(mock_task["task_id"], schedule_time)
    
    assert mock_task["task_id"] in scheduler.schedule
    assert scheduler.schedule[mock_task["task_id"]] == schedule_time
    
    # Start scheduler
    scheduler.start()
    assert scheduler.running is True
    
    # Stop scheduler
    scheduler.stop()
    assert scheduler.running is False

def test_task_executor_initialization():
    """Test initialization of TaskExecutor."""
    executor = TaskExecutor()
    assert executor.tasks == {}
    assert executor.results == {}
    assert executor.errors == {}
    assert executor.running is False

@pytest.mark.asyncio
async def test_task_executor_execution(mock_task, mock_execution_context):
    """Test task execution in TaskExecutor."""
    executor = TaskExecutor()
    
    # Add task
    executor.add_task(mock_task)
    assert mock_task["task_id"] in executor.tasks
    
    # Execute task
    result = await executor.execute(mock_task["task_id"], mock_execution_context)
    
    assert result is not None
    assert mock_task["task_id"] in executor.results
    assert executor.results[mock_task["task_id"]] == result

@pytest.mark.asyncio
async def test_task_executor_error_handling(mock_task, mock_execution_context):
    """Test error handling in TaskExecutor."""
    executor = TaskExecutor()
    
    # Add task with invalid code
    invalid_task = mock_task.copy()
    invalid_task["parameters"]["code"] = "invalid python code"
    executor.add_task(invalid_task)
    
    # Execute task
    with pytest.raises(Exception):
        await executor.execute(invalid_task["task_id"], mock_execution_context)
    
    assert invalid_task["task_id"] in executor.errors

@pytest.mark.asyncio
async def test_task_executor_parallel_execution(mock_task, mock_execution_context):
    """Test parallel task execution in TaskExecutor."""
    executor = TaskExecutor()
    
    # Add multiple tasks
    tasks = []
    for i in range(3):
        task = mock_task.copy()
        task["task_id"] = f"task_{i}"
        task["parameters"]["code"] = f"print('Task {i}')"
        tasks.append(task)
        executor.add_task(task)
    
    # Execute tasks in parallel
    results = await asyncio.gather(*[
        executor.execute(task["task_id"], mock_execution_context)
        for task in tasks
    ])
    
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result is not None
        assert f"task_{i}" in executor.results

def test_task_executor_state_management():
    """Test state management in TaskExecutor."""
    executor = TaskExecutor()
    
    # Set state
    executor.set_state("key", "value")
    assert executor.state["key"] == "value"
    
    # Get state
    value = executor.get_state("key")
    assert value == "value"
    
    # Update state
    executor.update_state("key", "new_value")
    assert executor.state["key"] == "new_value"
    
    # Clear state
    executor.clear_state()
    assert executor.state == {}

def test_task_executor_serialization():
    """Test serialization in TaskExecutor."""
    executor = TaskExecutor()
    executor.add_task({"task_id": "test_task", "type": "test"})
    executor.set_state("key", "value")
    
    serialized = executor.serialize()
    
    assert "tasks" in serialized
    assert "results" in serialized
    assert "errors" in serialized
    assert "state" in serialized
    assert serialized["state"]["key"] == "value"

def test_task_executor_deserialization():
    """Test deserialization in TaskExecutor."""
    serialized_data = {
        "tasks": {"test_task": {"task_id": "test_task", "type": "test"}},
        "results": {},
        "errors": {},
        "state": {"key": "value"}
    }
    
    executor = TaskExecutor.deserialize(serialized_data)
    
    assert "test_task" in executor.tasks
    assert executor.state["key"] == "value" 