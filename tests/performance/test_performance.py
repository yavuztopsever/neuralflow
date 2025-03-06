#!/usr/bin/env python3
"""
Performance tests for the LangGraph system.
Tests response times, memory usage, and system stability under load.
"""

import pytest
import asyncio
import time
import uuid
import psutil
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from graph.graph_workflow import create_workflow_graph, run_agent
from graph.context_handler import ContextHandler
from graph.task_execution import TaskExecution
from graph.response_generation import ResponseGenerator
from tools.memory_manager import MemoryManager
from models.gguf_wrapper.mock_llm import MockLLM

from src.core.graph.workflows import Workflow
from src.core.graph.engine import GraphEngine
from src.core.nodes.input import InputNode
from src.core.nodes.llm import LLMNode
from src.core.nodes.output import OutputNode
from src.core.nodes.memory import MemoryNode
from src.core.nodes.context import ContextNode
from src.core.nodes.reasoning import ReasoningNode
from src.ui.interface import Interface
from src.storage.vector import VectorStore
from src.storage.cache import Cache
from src.storage.database import Database

class TestPerformance:
    """Test suite for system performance metrics."""
    
    @pytest.fixture
    async def pipeline_components(self):
        """Initialize pipeline components for performance testing."""
        components = {}
        
        # Create the workflow graph
        components["workflow_graph"] = create_workflow_graph()
        
        # Create memory manager
        components["memory_manager"] = MemoryManager()
        
        # Create mock document handler
        components["document_handler"] = None
        
        # Create context handler
        components["context_handler"] = ContextHandler(
            memory_manager=components["memory_manager"],
            document_handler=components["document_handler"]
        )
        
        # Create task execution
        components["task_execution"] = TaskExecution(
            memory_manager=components["memory_manager"]
        )
        
        # Create response generator with mock LLM
        mock_llm = MockLLM()
        components["response_generator"] = ResponseGenerator(
            model="mock",
            temperature=0.7,
            model_instance=mock_llm
        )
        
        return components
    
    @pytest.mark.asyncio
    async def test_response_time(self, pipeline_components):
        """Test response time for a single query."""
        conversation_id = str(uuid.uuid4())
        query = "What is LangGraph?"
        
        start_time = time.time()
        response = await run_agent(
            user_query=query,
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 2.0  # Response should be generated within 2 seconds
        assert response is not None
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, pipeline_components):
        """Test memory usage during operation."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        conversation_id = str(uuid.uuid4())
        query = "Tell me about memory management"
        
        await run_agent(
            user_query=query,
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, pipeline_components):
        """Test system performance under concurrent load."""
        conversation_ids = [str(uuid.uuid4()) for _ in range(5)]
        queries = [
            "What is LangGraph?",
            "How does memory work?",
            "Tell me about task execution",
            "What are the main features?",
            "How does context handling work?"
        ]
        
        start_time = time.time()
        
        # Run concurrent requests
        tasks = []
        for conv_id, query in zip(conversation_ids, queries):
            task = run_agent(
                user_query=query,
                components=pipeline_components,
                conversation_id=conv_id,
                add_thinking=True
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 10.0  # All requests should complete within 10 seconds
        assert len(responses) == len(queries)
        assert all(len(r) > 0 for r in responses)
    
    @pytest.mark.asyncio
    async def test_long_conversation(self, pipeline_components):
        """Test system performance during a long conversation."""
        conversation_id = str(uuid.uuid4())
        queries = [
            "Hello, how are you?",
            "What is LangGraph?",
            "How does memory work?",
            "Tell me about task execution",
            "What are the main features?",
            "How does context handling work?",
            "Can you explain the workflow?",
            "What about error handling?",
            "How is performance optimized?",
            "Thank you for the information"
        ]
        
        start_time = time.time()
        responses = []
        
        for query in queries:
            response = await run_agent(
                user_query=query,
                components=pipeline_components,
                conversation_id=conversation_id,
                add_thinking=True
            )
            responses.append(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Long conversation should complete within reasonable time
        assert total_time < 30.0  # 30 seconds for 10 queries
        assert len(responses) == len(queries)
        assert all(len(r) > 0 for r in responses)
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, pipeline_components):
        """Test memory cleanup after operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform multiple operations
        for _ in range(5):
            conversation_id = str(uuid.uuid4())
            await run_agent(
                user_query="Test query",
                components=pipeline_components,
                conversation_id=conversation_id,
                add_thinking=True
            )
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory should be cleaned up properly
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase

    @pytest.fixture
    def system_components(self):
        """Create system components for testing."""
        # Create storage components
        vector_store = VectorStore()
        cache = Cache()
        database = Database()
        
        # Create UI interface
        interface = Interface()
        
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
        engine = GraphEngine()
        
        return {
            "vector_store": vector_store,
            "cache": cache,
            "database": database,
            "interface": interface,
            "workflow": workflow,
            "engine": engine
        }
    
    @pytest.mark.asyncio
    async def test_workflow_execution_performance(self, system_components):
        """Test workflow execution performance."""
        # Test single workflow execution
        start_time = time.time()
        result = await system_components["engine"].execute_workflow(
            system_components["workflow"],
            {"input": "Test input"}
        )
        execution_time = time.time() - start_time
        
        assert execution_time < 5.0  # Should execute within 5 seconds
        assert result is not None
        assert "output" in result
        
        # Test multiple workflow executions
        execution_times = []
        for _ in range(10):
            start_time = time.time()
            await system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": "Test input"}
            )
            execution_times.append(time.time() - start_time)
        
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < 5.0  # Average execution time should be under 5 seconds
    
    @pytest.mark.asyncio
    async def test_storage_performance(self, system_components):
        """Test storage performance."""
        # Test vector store performance
        vector_times = []
        for i in range(100):
            start_time = time.time()
            await system_components["vector_store"].store({
                "id": f"test_vector_{i}",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"test": f"data_{i}"}
            })
            vector_times.append(time.time() - start_time)
        
        avg_vector_time = sum(vector_times) / len(vector_times)
        assert avg_vector_time < 0.1  # Average storage time should be under 100ms
        
        # Test cache performance
        cache_times = []
        for i in range(100):
            start_time = time.time()
            await system_components["cache"].store({
                "key": f"test_key_{i}",
                "value": f"test_value_{i}",
                "ttl": 3600
            })
            cache_times.append(time.time() - start_time)
        
        avg_cache_time = sum(cache_times) / len(cache_times)
        assert avg_cache_time < 0.05  # Average cache time should be under 50ms
        
        # Test database performance
        db_times = []
        for i in range(100):
            start_time = time.time()
            await system_components["database"].store({
                "collection": "test_collection",
                "document": {
                    "id": f"test_doc_{i}",
                    "data": f"test_data_{i}"
                }
            })
            db_times.append(time.time() - start_time)
        
        avg_db_time = sum(db_times) / len(db_times)
        assert avg_db_time < 0.1  # Average database time should be under 100ms
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, system_components):
        """Test memory usage."""
        process = psutil.Process()
        
        # Test initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        assert initial_memory < 100  # Initial memory usage should be under 100MB
        
        # Test memory usage during workflow execution
        for _ in range(10):
            await system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": "Test input"}
            )
        
        current_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        assert current_memory < 500  # Memory usage should be under 500MB
        
        # Test memory usage during storage operations
        for i in range(1000):
            await system_components["vector_store"].store({
                "id": f"test_vector_{i}",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"test": f"data_{i}"}
            })
        
        final_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        assert final_memory < 1000  # Memory usage should be under 1GB
    
    @pytest.mark.asyncio
    async def test_concurrent_performance(self, system_components):
        """Test concurrent performance."""
        # Test concurrent workflow execution
        start_time = time.time()
        tasks = [
            system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": f"Test input {i}"}
            )
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        assert len(results) == 10
        assert concurrent_time < 10.0  # Should complete within 10 seconds
        
        # Test concurrent storage operations
        start_time = time.time()
        storage_tasks = [
            system_components["vector_store"].store({
                "id": f"test_vector_{i}",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"test": f"data_{i}"}
            })
            for i in range(100)
        ]
        
        storage_results = await asyncio.gather(*storage_tasks)
        storage_time = time.time() - start_time
        
        assert len(storage_results) == 100
        assert storage_time < 2.0  # Should complete within 2 seconds
    
    @pytest.mark.asyncio
    async def test_optimization_performance(self, system_components):
        """Test optimization performance."""
        # Test workflow optimization
        start_time = time.time()
        optimized_workflow = await system_components["engine"].optimize_workflow(
            system_components["workflow"],
            {
                "parallel_execution": True,
                "caching_enabled": True,
                "max_concurrent_nodes": 3
            }
        )
        optimization_time = time.time() - start_time
        
        assert optimization_time < 1.0  # Should optimize within 1 second
        assert optimized_workflow.is_optimized
        
        # Test optimized execution
        start_time = time.time()
        result = await system_components["engine"].execute_workflow(
            optimized_workflow,
            {"input": "Test input"}
        )
        optimized_time = time.time() - start_time
        
        assert optimized_time < 3.0  # Should execute faster after optimization
        assert result is not None
        assert "output" in result
    
    @pytest.mark.asyncio
    async def test_system_load_performance(self, system_components):
        """Test system load performance."""
        # Test under normal load
        start_time = time.time()
        for _ in range(100):
            await system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": "Test input"}
            )
        normal_load_time = time.time() - start_time
        
        # Test under high load
        start_time = time.time()
        tasks = [
            system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": f"Test input {i}"}
            )
            for i in range(100)
        ]
        await asyncio.gather(*tasks)
        high_load_time = time.time() - start_time
        
        # High load should not be significantly slower than normal load
        assert high_load_time < normal_load_time * 1.5  # Should not be 50% slower
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, system_components):
        """Test resource cleanup performance."""
        # Test memory cleanup
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Perform operations
        for i in range(100):
            await system_components["vector_store"].store({
                "id": f"test_vector_{i}",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"test": f"data_{i}"}
            })
        
        # Cleanup
        start_time = time.time()
        await system_components["vector_store"].cleanup()
        cleanup_time = time.time() - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024
        assert cleanup_time < 1.0  # Should cleanup within 1 second
        assert final_memory < initial_memory * 1.2  # Should not use more than 20% more memory 