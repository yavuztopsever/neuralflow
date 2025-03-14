"""
Unit tests for the WorkflowManager class.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
from pydantic import BaseModel

from src.core.workflow.workflow_manager import (
    WorkflowManager,
    WorkflowConfig,
    WorkflowState
)
from src.config.langchain_config import LangChainConfig

@pytest.fixture
def mock_langchain_config():
    """Create a mock LangChain configuration."""
    config = Mock(spec=LangChainConfig)
    config.vector_store = Mock()
    config.llm = Mock()
    return config

@pytest.fixture
def workflow_config(mock_langchain_config):
    """Create a workflow configuration for testing."""
    return WorkflowConfig(
        max_context_items=3,
        max_parallel_tasks=2,
        response_format="text",
        include_sources=True,
        include_metadata=False,
        execution_mode="safe",
        priority=0,
        add_thinking=True,
        langchain_config=mock_langchain_config,
        model_name="gpt-3.5-turbo"
    )

@pytest.fixture
def workflow_manager(workflow_config):
    """Create a WorkflowManager instance for testing."""
    with patch('src.core.workflow.workflow_manager.ChatOpenAI') as mock_chat_openai, \
         patch('src.core.workflow.workflow_manager.ConversationBufferMemory') as mock_memory, \
         patch('src.core.workflow.workflow_manager.LangChainTools') as mock_tools:
        
        # Setup mocks
        mock_chat_openai.return_value = Mock()
        mock_memory.return_value = Mock()
        mock_tools.return_value = Mock()
        
        manager = WorkflowManager(config=workflow_config)
        return manager

def test_workflow_manager_initialization(workflow_manager, workflow_config):
    """Test proper initialization of WorkflowManager."""
    assert workflow_manager.config == workflow_config
    assert workflow_manager.state_manager is not None
    assert workflow_manager.workflow_graph is not None

def test_workflow_state_creation():
    """Test creation of WorkflowState with default values."""
    state = WorkflowState()
    assert state.user_query == ""
    assert state.retrieved_context == {}
    assert state.execution_result == {}
    assert state.final_response is None
    assert state.error is None
    assert state.priority == 0
    assert state.memory == []
    assert state.thinking == []
    assert state.conversation_id is None
    assert state.context_processed is False
    assert state.needs_more_context is False

@pytest.mark.asyncio
async def test_user_input_node(workflow_manager):
    """Test the user input node processing."""
    test_state = WorkflowState(user_query="test query")
    
    result = await workflow_manager._user_input_node(test_state)
    
    assert isinstance(result, dict)
    assert "retrieved_context" in result
    assert "execution_result" in result
    assert "final_response" in result
    assert "needs_more_context" in result
    assert "next" in result
    assert result["next"] == "context_retrieval"

@pytest.mark.asyncio
async def test_context_retrieval_node(workflow_manager):
    """Test the context retrieval node processing."""
    test_state = WorkflowState(user_query="test query")
    
    # Mock the search tool response
    workflow_manager.tools.get_search_tool.return_value.run.return_value = "test context"
    
    result = await workflow_manager._context_retrieval_node(test_state)
    
    assert isinstance(result, dict)
    assert "retrieved_context" in result
    assert "context_processed" in result
    assert "next" in result
    assert result["next"] == "task_execution"
    assert result["context_processed"] is True

@pytest.mark.asyncio
async def test_task_execution_node(workflow_manager):
    """Test the task execution node processing."""
    test_state = WorkflowState(user_query="test query")
    
    # Mock the agent response
    workflow_manager.agent.run.return_value = "test result"
    
    result = await workflow_manager._task_execution_node(test_state)
    
    assert isinstance(result, dict)
    assert "execution_result" in result
    assert "next" in result
    assert result["next"] == "response_generation"

@pytest.mark.asyncio
async def test_workflow_run(workflow_manager):
    """Test the main workflow run method."""
    test_query = "test query"
    
    # Mock the workflow graph run method
    workflow_manager.workflow_graph.run = AsyncMock(return_value="test response")
    
    result = await workflow_manager.run(test_query)
    
    assert result == "test response"
    workflow_manager.workflow_graph.run.assert_called_once()

def test_workflow_state_management(workflow_manager):
    """Test state management methods."""
    test_state = {"user_query": "test", "priority": 1}
    
    workflow_manager.set_state(test_state)
    current_state = workflow_manager.get_state()
    
    assert current_state == test_state

@pytest.mark.asyncio
async def test_error_handling(workflow_manager):
    """Test error handling in workflow nodes."""
    test_state = WorkflowState(user_query="test query")
    
    # Mock an error in context retrieval
    workflow_manager.tools.get_search_tool.return_value.run.side_effect = Exception("Test error")
    
    result = await workflow_manager._context_retrieval_node(test_state)
    
    assert "error" in result
    assert "next" in result
    assert result["next"] == "END"

def test_checkpoint_management(workflow_manager):
    """Test checkpoint saving and loading."""
    checkpoint_id = "test_checkpoint"
    
    # Mock checkpoint operations
    workflow_manager.workflow_graph.save_checkpoint = Mock()
    workflow_manager.workflow_graph.load_checkpoint = Mock()
    
    workflow_manager.save_checkpoint(checkpoint_id)
    workflow_manager.load_checkpoint(checkpoint_id)
    
    workflow_manager.workflow_graph.save_checkpoint.assert_called_once_with(checkpoint_id)
    workflow_manager.workflow_graph.load_checkpoint.assert_called_once_with(checkpoint_id) 