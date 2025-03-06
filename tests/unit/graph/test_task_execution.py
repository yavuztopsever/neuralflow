"""
Unit tests for the task execution component.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from typing import Dict, Any, List

from graph.task_execution import TaskExecutor


@pytest.mark.asyncio
async def test_execute_task_web_search(
    mock_web_search,
    mock_model_manager,
    test_config,
    sample_user_query
):
    """Test execution of a web search task."""
    # Create task executor with mocked dependencies
    task_executor = TaskExecutor(
        web_search=mock_web_search,
        function_caller=MagicMock(),
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up mock return for web search
    web_results = [
        {"title": "LangGraph Docs", "snippet": "Official documentation", "url": "https://example.com/docs"},
        {"title": "LangGraph Tutorial", "snippet": "Learn to use LangGraph", "url": "https://example.com/tutorial"}
    ]
    mock_web_search.search.return_value = web_results
    
    # Set up model manager mock
    mock_model_manager.generate_response.return_value = json.dumps({
        "task_type": "web_search",
        "search_query": "langgraph documentation"
    })
    
    # Execute function under test
    result = await task_executor.execute_task(
        query=sample_user_query,
        context={"insufficient": True}
    )
    
    # Assertions
    assert result["status"] == "success"
    assert "data" in result
    assert "web_results" in result["data"]
    assert len(result["data"]["web_results"]) == 2
    mock_web_search.search.assert_called_once()
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_execute_task_function_call(
    mock_function_caller,
    mock_model_manager,
    test_config,
    sample_user_query
):
    """Test execution of a function call task."""
    # Create task executor with mocked dependencies
    task_executor = TaskExecutor(
        web_search=MagicMock(),
        function_caller=mock_function_caller,
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up model manager mock
    mock_model_manager.generate_response.return_value = json.dumps({
        "task_type": "function_call",
        "function_name": "calculate_similarity",
        "function_args": {"text1": "sample", "text2": "example"}
    })
    
    # Set up function caller mock
    function_result = {"similarity": 0.85}
    mock_function_caller.execute_function.return_value = {
        "status": "success", 
        "result": function_result
    }
    
    # Execute function under test
    result = await task_executor.execute_task(
        query=sample_user_query,
        context={"insufficient": True}
    )
    
    # Assertions
    assert result["status"] == "success"
    assert "data" in result
    assert "function_result" in result["data"]
    assert result["data"]["function_result"] == function_result
    mock_function_caller.execute_function.assert_called_once()
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_determine_task_type(
    mock_model_manager,
    test_config,
    sample_user_query
):
    """Test determination of task type based on query and context."""
    # Create task executor with mocked dependencies
    task_executor = TaskExecutor(
        web_search=MagicMock(),
        function_caller=MagicMock(),
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up model manager mock
    task_decision = {
        "task_type": "web_search",
        "reasoning": "Need additional information from the web",
        "search_query": "langgraph documentation examples"
    }
    mock_model_manager.generate_response.return_value = json.dumps(task_decision)
    
    # Execute function under test
    result = await task_executor.determine_task_type(
        query=sample_user_query,
        context={"insufficient": True}
    )
    
    # Assertions
    assert result == task_decision
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_execute_task_error_handling(
    mock_web_search,
    mock_model_manager,
    test_config,
    sample_user_query
):
    """Test error handling during task execution."""
    # Create task executor with mocked dependencies
    task_executor = TaskExecutor(
        web_search=mock_web_search,
        function_caller=MagicMock(),
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up model manager mock
    mock_model_manager.generate_response.return_value = json.dumps({
        "task_type": "web_search",
        "search_query": "langgraph documentation"
    })
    
    # Make web search raise an exception
    mock_web_search.search.side_effect = Exception("Search API error")
    
    # Execute function under test
    result = await task_executor.execute_task(
        query=sample_user_query,
        context={"insufficient": True}
    )
    
    # Assertions
    assert result["status"] == "error"
    assert "error" in result
    assert "Search API error" in result["error"]
    mock_web_search.search.assert_called_once()
    mock_model_manager.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_execute_multiple_tasks(
    mock_web_search,
    mock_function_caller,
    mock_model_manager,
    test_config,
    sample_user_query
):
    """Test execution of multiple sequential tasks."""
    # Create task executor with mocked dependencies
    task_executor = TaskExecutor(
        web_search=mock_web_search,
        function_caller=mock_function_caller,
        model_manager=mock_model_manager,
        config=test_config
    )
    
    # Set up model manager mock for two different task decisions
    mock_model_manager.generate_response.side_effect = [
        # First call - decide to do web search
        json.dumps({
            "task_type": "web_search",
            "search_query": "langgraph documentation"
        }),
        # Second call - decide to do function call
        json.dumps({
            "task_type": "function_call",
            "function_name": "summarize_results",
            "function_args": {"text": "search results"}
        })
    ]
    
    # Set up web search mock
    web_results = [{"title": "LangGraph Docs", "snippet": "Documentation", "url": "https://example.com"}]
    mock_web_search.search.return_value = web_results
    
    # Set up function caller mock
    function_result = {"summary": "LangGraph is a framework for LLM apps"}
    mock_function_caller.execute_function.return_value = {
        "status": "success", 
        "result": function_result
    }
    
    # Execute first task
    result1 = await task_executor.execute_task(
        query=sample_user_query,
        context={"insufficient": True}
    )
    
    # Update context with first result
    context = {
        "insufficient": True,
        "web_results": web_results
    }
    
    # Execute second task
    result2 = await task_executor.execute_task(
        query=sample_user_query,
        context=context
    )
    
    # Assertions
    assert result1["status"] == "success"
    assert "web_results" in result1["data"]
    
    assert result2["status"] == "success"
    assert "function_result" in result2["data"]
    assert result2["data"]["function_result"] == function_result
    
    assert mock_model_manager.generate_response.call_count == 2
    mock_web_search.search.assert_called_once()
    mock_function_caller.execute_function.assert_called_once()