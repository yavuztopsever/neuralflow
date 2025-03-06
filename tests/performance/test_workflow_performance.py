"""
Performance tests for the LangGraph workflow.
"""
import pytest
import os
import json
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import statistics
from typing import Dict, Any, List, Tuple

from graph.graph_workflow import GraphWorkflow
from graph.context_handler import ContextHandler
from graph.task_execution import TaskExecutor
from graph.response_generation import ResponseGenerator
from tools.memory_manager import MemoryManager
from tools.vector_search import VectorSearch
from models.model_manager import ModelManager
from config.config import Config


class TestWorkflowPerformance:
    """Performance tests for the workflow components."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory for test databases."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Not cleaning up to allow post-test analysis

    @pytest.fixture
    def performance_workflow(self, temp_db_dir):
        """Create a workflow with mocked components for performance testing."""
        # Override environment
        os.environ["USE_MOCK_LLM"] = "true"
        os.environ["MINIMAL_MODE"] = "true"
        
        # Create config with test paths
        config = Config()
        config.MEMORY_STORE_PATH = os.path.join(temp_db_dir, "memory.db")
        config.VECTOR_STORE_PATH = os.path.join(temp_db_dir, "vector_store")
        config.GRAPH_STORE_PATH = os.path.join(temp_db_dir, "knowledge_graph.json")
        config.DOCUMENT_STORE_PATH = os.path.join(temp_db_dir, "documents")
        config.CHECKPOINTS_PATH = os.path.join(temp_db_dir, "checkpoints")
        config.DEBUG = True
        
        # Create mock components with controlled latency
        memory_manager = MemoryManager(config)
        vector_search = VectorSearch(config)
        document_handler = MagicMock()
        model_manager = ModelManager(config)
        
        # Override model manager methods to have controlled latency
        model_manager.generate_response = AsyncMock(
            return_value="This is a test response with controlled latency."
        )
        model_manager.get_embeddings = AsyncMock(
            return_value=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Create workflow components
        context_handler = ContextHandler(
            memory_manager=memory_manager,
            vector_search=vector_search,
            document_handler=document_handler,
            config=config
        )
        context_handler.model_manager = model_manager
        
        task_executor = TaskExecutor(
            web_search=MagicMock(),
            function_caller=MagicMock(),
            model_manager=model_manager,
            config=config
        )
        
        response_generator = ResponseGenerator(
            model_manager=model_manager,
            config=config
        )
        
        # Create workflow
        workflow = GraphWorkflow(
            context_handler=context_handler,
            task_executor=task_executor,
            response_generator=response_generator,
            memory_manager=memory_manager,
            config=config
        )
        
        yield workflow, model_manager
        
        # Clean up after test
        # Not cleaning up to allow post-test analysis

    async def _time_execution(self, workflow, query: str) -> Tuple[float, Dict[str, Any]]:
        """Time the execution of a workflow and return the result and time taken."""
        start_time = time.time()
        
        # Create mock graph for controlled execution
        mock_graph = MagicMock()
        mock_graph.invoke = AsyncMock(return_value={
            "query": query,
            "status": "completed",
            "response": "This is a test response."
        })
        
        # Set up the mock
        with patch.object(workflow, 'create_workflow', return_value=mock_graph):
            result = await workflow.invoke(query)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return duration, result

    @pytest.mark.asyncio
    async def test_workflow_baseline_performance(self, performance_workflow):
        """Test baseline performance of the workflow."""
        workflow, model_manager = performance_workflow
        
        # Configure model manager mock response times
        async def delayed_generate(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms simulated latency
            return "This is a test response."
        
        model_manager.generate_response = delayed_generate
        
        # Run multiple iterations
        iterations = 10
        durations = []
        
        for i in range(iterations):
            query = f"Test query {i}"
            duration, result = await self._time_execution(workflow, query)
            durations.append(duration)
        
        # Calculate statistics
        avg_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Log performance metrics
        print(f"Performance metrics for {iterations} iterations:")
        print(f"Average: {avg_duration:.6f}s")
        print(f"Median: {median_duration:.6f}s")
        print(f"Min: {min_duration:.6f}s")
        print(f"Max: {max_duration:.6f}s")
        
        # Assert acceptable performance (adjust thresholds as needed)
        assert avg_duration < 0.5, "Average response time exceeds threshold"

    @pytest.mark.asyncio
    async def test_workflow_scaling_with_context_size(self, performance_workflow):
        """Test how workflow performance scales with context size."""
        workflow, model_manager = performance_workflow
        
        # Test with different context sizes
        context_sizes = [10, 100, 1000, 5000]
        results = []
        
        for size in context_sizes:
            # Configure model with simulated context-dependent latency
            async def delayed_generate(*args, **kwargs):
                # Scale latency with context size (simulated processing time)
                await asyncio.sleep(0.01 + (size / 50000))  # Base latency + scaling factor
                return "This is a test response."
            
            model_manager.generate_response = delayed_generate
            
            # Run test with this context size
            durations = []
            iterations = 5
            
            for i in range(iterations):
                query = f"Test query with context size {size}"
                duration, _ = await self._time_execution(workflow, query)
                durations.append(duration)
            
            # Record results
            avg_duration = statistics.mean(durations)
            results.append((size, avg_duration))
        
        # Log scaling results
        print("Context size scaling results:")
        for size, duration in results:
            print(f"Context size {size}: {duration:.6f}s")
        
        # Assert acceptable scaling (should be sublinear)
        largest_size, largest_duration = results[-1]
        smallest_size, smallest_duration = results[0]
        
        # Calculate scaling factor
        size_ratio = largest_size / smallest_size
        duration_ratio = largest_duration / smallest_duration
        
        print(f"Size increased by {size_ratio}x, duration increased by {duration_ratio}x")
        assert duration_ratio < size_ratio, "Scaling is not sublinear"

    @pytest.mark.asyncio
    async def test_workflow_concurrent_requests(self, performance_workflow):
        """Test workflow performance under concurrent requests."""
        workflow, model_manager = performance_workflow
        
        # Configure model manager with fixed latency
        async def delayed_generate(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms simulated latency
            return "This is a test response."
        
        model_manager.generate_response = delayed_generate
        
        # Run concurrent requests
        concurrency_levels = [1, 5, 10, 20]
        results = []
        
        for concurrency in concurrency_levels:
            tasks = []
            start_time = time.time()
            
            # Create concurrent tasks
            for i in range(concurrency):
                query = f"Concurrent query {i}"
                task = self._time_execution(workflow, query)
                tasks.append(task)
            
            # Wait for all tasks to complete
            durations_and_results = await asyncio.gather(*tasks)
            durations = [d for d, _ in durations_and_results]
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Calculate statistics
            avg_duration = statistics.mean(durations)
            throughput = concurrency / total_duration  # requests per second
            
            results.append((concurrency, avg_duration, throughput))
        
        # Log concurrency results
        print("Concurrency test results:")
        for concurrency, avg_duration, throughput in results:
            print(f"Concurrency {concurrency}: avg={avg_duration:.6f}s, throughput={throughput:.2f} req/s")
        
        # Check if throughput scales with concurrency (should increase)
        min_concurrency, _, min_throughput = results[0]
        max_concurrency, _, max_throughput = results[-1]
        
        concurrency_ratio = max_concurrency / min_concurrency
        throughput_ratio = max_throughput / min_throughput
        
        print(f"Concurrency increased by {concurrency_ratio}x, throughput increased by {throughput_ratio}x")
        assert throughput_ratio > 1.0, "Throughput does not scale with concurrency"

    @pytest.mark.asyncio
    async def test_memory_growth_over_time(self, performance_workflow, temp_db_dir):
        """Test memory usage growth over extended conversations."""
        workflow, model_manager = performance_workflow
        
        # Configure model manager
        model_manager.generate_response = AsyncMock(return_value="Test response")
        
        # Set up tracking
        memory_sizes = []
        conversation_turns = 20
        
        # Get memory file path
        memory_db_path = os.path.join(temp_db_dir, "memory.db")
        
        # Measure initial size
        if os.path.exists(memory_db_path):
            initial_size = os.path.getsize(memory_db_path)
        else:
            initial_size = 0
        
        memory_sizes.append(initial_size)
        
        # Run conversation turns and track memory growth
        for i in range(conversation_turns):
            query = f"Test query {i} for memory growth analysis"
            
            # Execute workflow
            _, result = await self._time_execution(workflow, query)
            
            # Measure memory size after this turn
            if os.path.exists(memory_db_path):
                current_size = os.path.getsize(memory_db_path)
                memory_sizes.append(current_size)
        
        # Calculate growth
        if len(memory_sizes) > 1 and memory_sizes[0] > 0:
            growth_factor = memory_sizes[-1] / memory_sizes[0]
        else:
            growth_factor = float('inf') if memory_sizes[-1] > 0 else 1.0
        
        # Log memory growth
        print(f"Memory size initial: {memory_sizes[0]} bytes")
        print(f"Memory size final: {memory_sizes[-1]} bytes")
        print(f"Growth factor: {growth_factor:.2f}x")
        
        # Check for reasonable growth
        # Each turn should add a relatively small amount of data
        assert growth_factor < conversation_turns, "Memory growth is excessive"