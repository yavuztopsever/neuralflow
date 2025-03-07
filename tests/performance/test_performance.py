#!/usr/bin/env python3
"""
Performance tests for the NeuralFlow system.
Tests response times, memory usage, and system stability under load.
"""

import pytest
import asyncio
import time
import uuid
import psutil
import gc
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from neuralflow.core.workflow import Workflow
from neuralflow.core.engine import WorkflowEngine
from neuralflow.core.nodes import (
    InputNode, LLMNode, MemoryNode, ContextNode,
    ReasoningNode, OutputNode, Edge
)
from neuralflow.models.llm import LLMModel
from neuralflow.models.embeddings import EmbeddingModel
from neuralflow.models.classifier import ClassifierModel
from neuralflow.storage.vector import VectorStore
from neuralflow.storage.cache import Cache
from neuralflow.storage.database import Database
from neuralflow.ui.interface import Interface
from neuralflow.core.config import SystemConfig

class TestPerformance:
    """Test suite for system performance metrics."""
    
    @pytest.fixture
    def system_config(self):
        """Create system configuration for testing."""
        return SystemConfig(
            max_workers=4,
            cache_size=1000,
            max_memory_mb=1024,
            timeout_seconds=30,
            retry_attempts=3,
            batch_size=32,
            enable_metrics=True
        )
    
    @pytest.fixture
    async def system_components(self, system_config):
        """Create system components for testing."""
        # Create storage components
        vector_store = VectorStore(config=system_config)
        cache = Cache(config=system_config)
        database = Database(config=system_config)
        
        # Create UI interface
        interface = Interface(config=system_config)
        
        # Create model components
        llm_model = LLMModel(config=system_config)
        embedding_model = EmbeddingModel(config=system_config)
        classifier_model = ClassifierModel(config=system_config)
        
        # Create workflow components
        input_node = InputNode("input", "input", {})
        llm_node = LLMNode("llm", "llm", {"model": llm_model})
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
        engine = WorkflowEngine(config=system_config)
        
        return {
            "config": system_config,
            "workflow": workflow,
            "engine": engine,
            "vector_store": vector_store,
            "cache": cache,
            "database": database,
            "interface": interface,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "classifier_model": classifier_model
        }
    
    @pytest.mark.asyncio
    async def test_response_time(self, system_components):
        """Test response time for a single query."""
        start_time = time.time()
        
        # Execute workflow
        result = await system_components["engine"].execute_workflow(
            system_components["workflow"],
            {"input": "Test query"}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Collect metrics
        metrics = await system_components["engine"].collect_metrics()
        
        assert response_time < 2.0  # Response should be generated within 2 seconds
        assert result is not None
        assert "output" in result
        assert metrics["average_response_time"] < 2.0
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, system_components):
        """Test memory usage during operation."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Execute workflow
        await system_components["engine"].execute_workflow(
            system_components["workflow"],
            {"input": "Test query"}
        )
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
        
        # Check memory metrics
        metrics = await system_components["engine"].collect_metrics()
        assert metrics["memory_usage"] < 100 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, system_components):
        """Test system performance under concurrent load."""
        queries = [
            "Test query 1",
            "Test query 2",
            "Test query 3",
            "Test query 4",
            "Test query 5"
        ]
        
        start_time = time.time()
        
        # Run concurrent requests
        tasks = []
        for query in queries:
            task = system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": query}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        metrics = await system_components["engine"].collect_metrics()
        
        assert total_time < 10.0  # All requests should complete within 10 seconds
        assert len(results) == len(queries)
        assert all("output" in r for r in results)
        assert metrics["concurrent_requests"] == len(queries)
        assert metrics["average_concurrent_time"] < 10.0
    
    @pytest.mark.asyncio
    async def test_long_conversation(self, system_components):
        """Test system performance during a long conversation."""
        queries = [
            "Initial query",
            "Follow-up question 1",
            "Follow-up question 2",
            "Follow-up question 3",
            "Follow-up question 4",
            "Follow-up question 5",
            "Follow-up question 6",
            "Follow-up question 7",
            "Follow-up question 8",
            "Final question"
        ]
        
        start_time = time.time()
        responses = []
        
        for query in queries:
            result = await system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": query}
            )
            responses.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        metrics = await system_components["engine"].collect_metrics()
        
        assert total_time < 30.0  # 30 seconds for 10 queries
        assert len(responses) == len(queries)
        assert all("output" in r for r in responses)
        assert metrics["total_queries"] == len(queries)
        assert metrics["average_conversation_time"] < 30.0
    
    @pytest.mark.asyncio
    async def test_storage_performance(self, system_components):
        """Test storage system performance."""
        # Test vector store performance
        vector_start = time.time()
        for i in range(100):
            await system_components["vector_store"].store(
                f"test_vector_{i}",
                [0.1] * 128
            )
        vector_time = time.time() - vector_start
        
        # Test cache performance
        cache_start = time.time()
        for i in range(100):
            await system_components["cache"].set(
                f"test_key_{i}",
                f"test_value_{i}"
            )
        cache_time = time.time() - cache_start
        
        # Test database performance
        db_start = time.time()
        for i in range(100):
            await system_components["database"].store({
                "collection": "test_collection",
                "document": {
                    "id": f"test_doc_{i}",
                    "data": f"test_data_{i}"
                }
            })
        db_time = time.time() - db_start
        
        assert vector_time < 5.0  # Vector store operations should be fast
        assert cache_time < 1.0   # Cache operations should be very fast
        assert db_time < 10.0     # Database operations should be reasonable
    
    @pytest.mark.asyncio
    async def test_model_performance(self, system_components):
        """Test model performance metrics."""
        # Test LLM performance
        llm_start = time.time()
        response = await system_components["llm_model"].generate("Test prompt")
        llm_time = time.time() - llm_start
        
        # Test embedding performance
        embedding_start = time.time()
        embedding = await system_components["embedding_model"].generate_embedding("Test text")
        embedding_time = time.time() - embedding_start
        
        # Test classifier performance
        classifier_start = time.time()
        prediction = await system_components["classifier_model"].classify("Test text")
        classifier_time = time.time() - classifier_start
        
        assert llm_time < 3.0      # LLM generation should be reasonable
        assert embedding_time < 1.0 # Embedding generation should be fast
        assert classifier_time < 1.0 # Classification should be fast
    
    @pytest.mark.asyncio
    async def test_system_load(self, system_components):
        """Test system performance under load."""
        # Simulate high load
        tasks = []
        for i in range(20):  # 20 concurrent tasks
            task = system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": f"Load test query {i}"}
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        metrics = await system_components["engine"].collect_metrics()
        
        assert total_time < 60.0  # System should handle load within 60 seconds
        assert len(results) == 20
        assert metrics["active_tasks"] <= system_components["config"].max_workers
        assert metrics["queue_size"] == 0  # No tasks should be queued
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, system_components):
        """Test resource cleanup after operations."""
        # Perform multiple operations
        for i in range(10):
            await system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": f"Cleanup test query {i}"}
            )
        
        # Force cleanup
        await system_components["engine"].cleanup()
        await system_components["vector_store"].cleanup()
        await system_components["cache"].cleanup()
        await system_components["database"].cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Check resource usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        metrics = await system_components["engine"].collect_metrics()
        
        assert memory_usage < 200 * 1024 * 1024  # Less than 200MB after cleanup
        assert metrics["active_connections"] == 0
        assert metrics["active_tasks"] == 0
        assert metrics["queue_size"] == 0 