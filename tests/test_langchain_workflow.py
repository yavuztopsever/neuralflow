"""
Tests for LangChain workflow integration.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.config.langchain_config import LangChainConfig
from src.core.workflow.workflow_manager import WorkflowManager, WorkflowConfig

@pytest.fixture
def langchain_config():
    """Create a mock LangChain configuration."""
    return LangChainConfig(
        openai_api_key="test-key",
        model_name="gpt-4-turbo-preview",
        temperature=0.7
    )

@pytest.fixture
def workflow_config(langchain_config):
    """Create a workflow configuration with LangChain config."""
    return WorkflowConfig(
        max_context_items=5,
        max_parallel_tasks=3,
        response_format="text",
        include_sources=True,
        include_metadata=True,
        execution_mode="safe",
        priority=0,
        add_thinking=True,
        langchain_config=langchain_config
    )

@pytest.fixture
def workflow_manager(workflow_config):
    """Create a workflow manager instance."""
    return WorkflowManager(workflow_config)

@pytest.mark.asyncio
async def test_workflow_initialization(workflow_manager):
    """Test workflow manager initialization."""
    assert workflow_manager.config is not None
    assert workflow_manager.langchain_manager is not None
    assert workflow_manager.tools is not None
    assert workflow_manager.workflow_chain is not None
    assert workflow_manager.agent is not None

@pytest.mark.asyncio
async def test_user_input_node(workflow_manager):
    """Test user input node processing."""
    state = {"user_query": "test query"}
    result = await workflow_manager._user_input_node(state)
    
    assert "retrieved_context" in result
    assert "execution_result" in result
    assert "final_response" in result
    assert "needs_more_context" in result
    assert "next" in result
    assert result["next"] == "context_retrieval"

@pytest.mark.asyncio
async def test_context_retrieval_node(workflow_manager):
    """Test context retrieval node processing."""
    state = {"user_query": "test query"}
    result = await workflow_manager._context_retrieval_node(state)
    
    assert "retrieved_context" in result
    assert "context_processed" in result
    assert "next" in result
    assert result["next"] == "task_execution"

@pytest.mark.asyncio
async def test_task_execution_node(workflow_manager):
    """Test task execution node processing."""
    state = {"user_query": "test query"}
    result = await workflow_manager._task_execution_node(state)
    
    assert "execution_result" in result
    assert "next" in result
    assert result["next"] == "response_generation"

@pytest.mark.asyncio
async def test_response_generation_node(workflow_manager):
    """Test response generation node processing."""
    state = {"user_query": "test query"}
    result = await workflow_manager._response_generation_node(state)
    
    assert "final_response" in result
    assert "next" in result
    assert result["next"] == END

@pytest.mark.asyncio
async def test_workflow_run(workflow_manager):
    """Test complete workflow execution."""
    query = "test query"
    response = await workflow_manager.run(query)
    
    assert response is not None
    assert isinstance(response, str)

@pytest.mark.asyncio
async def test_workflow_run_with_progress(workflow_manager):
    """Test workflow execution with progress updates."""
    query = "test query"
    progress_updates = []
    
    async for progress, message in workflow_manager.run_with_progress(query):
        progress_updates.append((progress, message))
    
    assert len(progress_updates) > 0
    for progress, message in progress_updates:
        assert 0 <= progress <= 1
        assert isinstance(message, str)

@pytest.mark.asyncio
async def test_error_handling(workflow_manager):
    """Test error handling in workflow execution."""
    with patch.object(workflow_manager.agent, 'run', side_effect=Exception("Test error")):
        response = await workflow_manager.run("test query")
        assert "Error executing workflow" in response

@pytest.mark.asyncio
async def test_state_management(workflow_manager):
    """Test workflow state management."""
    test_state = {"test": "value"}
    workflow_manager.set_state(test_state)
    retrieved_state = workflow_manager.get_state()
    assert retrieved_state == test_state

@pytest.mark.asyncio
async def test_checkpoint_management(workflow_manager):
    """Test workflow checkpoint management."""
    checkpoint_id = "test_checkpoint"
    workflow_manager.save_checkpoint(checkpoint_id)
    workflow_manager.load_checkpoint(checkpoint_id)
    # No assertions needed as we're just testing that the methods don't raise exceptions 