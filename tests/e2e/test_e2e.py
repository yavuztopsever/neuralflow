#!/usr/bin/env python3
"""
End-to-end tests for the LangGraph system.
Tests the complete system flow from user input to response generation.
"""

import pytest
import asyncio
import uuid
from typing import Dict, Any, List

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

class TestEndToEnd:
    """Test suite for end-to-end system functionality."""
    
    @pytest.fixture
    async def pipeline_components(self):
        """Initialize pipeline components for end-to-end testing."""
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
    async def test_complete_user_journey(self, pipeline_components):
        """Test a complete user journey with multiple interactions."""
        conversation_id = str(uuid.uuid4())
        
        # Initial greeting
        response = await run_agent(
            user_query="Hello, I'm new to LangGraph",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert len(response) > 0
        
        # Ask about features
        response = await run_agent(
            user_query="What are the main features?",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "feature" in response.lower()
        
        # Ask about memory system
        response = await run_agent(
            user_query="How does the memory system work?",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "memory" in response.lower()
        
        # Ask about task execution
        response = await run_agent(
            user_query="Can you explain task execution?",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "task" in response.lower()
        
        # Thank the system
        response = await run_agent(
            user_query="Thank you for the information!",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_context_persistence(self, pipeline_components):
        """Test that context is maintained throughout the conversation."""
        conversation_id = str(uuid.uuid4())
        
        # Store personal information
        await run_agent(
            user_query="My name is Alice and I'm a developer",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        
        # Ask about stored information
        response = await run_agent(
            user_query="What do you know about me?",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "alice" in response.lower()
        assert "developer" in response.lower()
    
    @pytest.mark.asyncio
    async def test_task_execution_flow(self, pipeline_components):
        """Test the complete task execution flow."""
        conversation_id = str(uuid.uuid4())
        
        # Request a calculation
        response = await run_agent(
            user_query="Calculate 5 * 3 and explain the process",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "15" in response
        assert "calculate" in response.lower()
        
        # Request another calculation
        response = await run_agent(
            user_query="Now calculate 10 + 20",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "30" in response
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, pipeline_components):
        """Test system recovery from errors."""
        conversation_id = str(uuid.uuid4())
        
        # First, establish context
        await run_agent(
            user_query="My favorite color is blue",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        
        # Send an invalid query
        response = await run_agent(
            user_query="Invalid query that should trigger error handling",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert len(response) > 0
        
        # System should recover and continue
        response = await run_agent(
            user_query="What's my favorite color?",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "blue" in response.lower()
    
    @pytest.mark.asyncio
    async def test_complex_workflow(self, pipeline_components):
        """Test a complex workflow with multiple components."""
        conversation_id = str(uuid.uuid4())
        
        # Start with a complex query
        response = await run_agent(
            user_query="Explain how LangGraph handles memory management and task execution",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "memory" in response.lower()
        assert "task" in response.lower()
        
        # Ask for specific details
        response = await run_agent(
            user_query="Can you give me an example of task execution?",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "example" in response.lower()
        
        # Request a practical demonstration
        response = await run_agent(
            user_query="Show me how to use the memory system",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert "memory" in response.lower()
        
        # Verify the system maintains context
        response = await run_agent(
            user_query="Summarize what we've discussed",
            components=pipeline_components,
            conversation_id=conversation_id,
            add_thinking=True
        )
        assert response is not None
        assert any(word in response.lower() for word in ["memory", "task", "example"])

class TestE2E:
    """Test suite for end-to-end tests."""
    
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
    async def test_complete_system_flow(self, system_components):
        """Test complete system flow."""
        # Initialize components
        await system_components["vector_store"].initialize()
        await system_components["cache"].initialize()
        await system_components["database"].initialize()
        await system_components["interface"].initialize()
        
        # Register workflow
        await system_components["engine"].register_workflow(system_components["workflow"])
        
        # Process user input
        user_input = "Test user input"
        interface_response = await system_components["interface"].handle_input(user_input)
        assert interface_response is not None
        assert "status" in interface_response
        assert interface_response["status"] == "success"
        
        # Execute workflow
        workflow_result = await system_components["engine"].execute_workflow(
            system_components["workflow"],
            {"input": user_input}
        )
        assert workflow_result is not None
        assert "output" in workflow_result
        
        # Store results
        await system_components["vector_store"].store({
            "id": "test_vector",
            "vector": [0.1, 0.2, 0.3],
            "metadata": workflow_result
        })
        
        await system_components["cache"].store({
            "key": "test_key",
            "value": workflow_result,
            "ttl": 3600
        })
        
        await system_components["database"].store({
            "collection": "test_collection",
            "document": {
                "id": "test_doc",
                "data": workflow_result
            }
        })
        
        # Verify storage
        vector_result = await system_components["vector_store"].retrieve("test_vector")
        assert vector_result is not None
        assert vector_result["metadata"] == workflow_result
        
        cache_result = await system_components["cache"].retrieve("test_key")
        assert cache_result == workflow_result
        
        db_result = await system_components["database"].retrieve(
            "test_collection",
            "test_doc"
        )
        assert db_result["data"] == workflow_result
    
    @pytest.mark.asyncio
    async def test_system_error_handling(self, system_components):
        """Test system error handling."""
        # Test component initialization errors
        with pytest.raises(Exception):
            await system_components["vector_store"].initialize()
        
        # Test workflow execution errors
        with pytest.raises(ValueError):
            await system_components["engine"].execute_workflow(
                system_components["workflow"],
                {}
            )
        
        # Test storage errors
        with pytest.raises(ValueError):
            await system_components["vector_store"].store(None)
        
        # Test interface errors
        with pytest.raises(ValueError):
            await system_components["interface"].handle_input(None)
    
    @pytest.mark.asyncio
    async def test_system_performance(self, system_components):
        """Test system performance."""
        # Test system initialization time
        start_time = asyncio.get_event_loop().time()
        await system_components["vector_store"].initialize()
        await system_components["cache"].initialize()
        await system_components["database"].initialize()
        await system_components["interface"].initialize()
        init_time = asyncio.get_event_loop().time() - start_time
        assert init_time < 2.0  # Should initialize within 2 seconds
        
        # Test workflow execution time
        start_time = asyncio.get_event_loop().time()
        await system_components["engine"].execute_workflow(
            system_components["workflow"],
            {"input": "Test input"}
        )
        exec_time = asyncio.get_event_loop().time() - start_time
        assert exec_time < 5.0  # Should execute within 5 seconds
        
        # Test storage operations time
        start_time = asyncio.get_event_loop().time()
        await system_components["vector_store"].store({
            "id": "test_vector",
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"test": "data"}
        })
        storage_time = asyncio.get_event_loop().time() - start_time
        assert storage_time < 1.0  # Should store within 1 second
    
    @pytest.mark.asyncio
    async def test_system_concurrency(self, system_components):
        """Test system concurrency."""
        # Test concurrent workflow execution
        tasks = [
            system_components["engine"].execute_workflow(
                system_components["workflow"],
                {"input": f"Test input {i}"}
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(r is not None for r in results)
        
        # Test concurrent storage operations
        storage_tasks = [
            system_components["vector_store"].store({
                "id": f"test_vector_{i}",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"test": f"data_{i}"}
            })
            for i in range(5)
        ]
        
        storage_results = await asyncio.gather(*storage_tasks)
        assert len(storage_results) == 5
        assert all(r is not None for r in storage_results)
    
    @pytest.mark.asyncio
    async def test_system_reliability(self, system_components):
        """Test system reliability."""
        # Test component recovery
        await system_components["vector_store"].recover()
        assert await system_components["vector_store"].check_recovery()
        
        await system_components["cache"].recover()
        assert await system_components["cache"].check_recovery()
        
        await system_components["database"].recover()
        assert await system_components["database"].check_recovery()
        
        # Test data persistence
        await system_components["vector_store"].persist()
        assert await system_components["vector_store"].check_persistence()
        
        await system_components["cache"].persist()
        assert await system_components["cache"].check_persistence()
        
        await system_components["database"].persist()
        assert await system_components["database"].check_persistence()
    
    @pytest.mark.asyncio
    async def test_system_optimization(self, system_components):
        """Test system optimization."""
        # Test component optimization
        vector_params = {
            "index_type": "hnsw",
            "dimension": 128,
            "max_elements": 1000
        }
        
        optimized_vector = await system_components["vector_store"].optimize(vector_params)
        assert optimized_vector is not None
        assert optimized_vector.is_optimized
        
        cache_params = {
            "max_size": 1000,
            "eviction_policy": "lru",
            "ttl": 3600
        }
        
        optimized_cache = await system_components["cache"].optimize(cache_params)
        assert optimized_cache is not None
        assert optimized_cache.is_optimized
        
        db_params = {
            "index_fields": ["id", "data"],
            "max_connections": 10,
            "pool_size": 5
        }
        
        optimized_db = await system_components["database"].optimize(db_params)
        assert optimized_db is not None
        assert optimized_db.is_optimized
        
        # Test optimized system performance
        start_time = asyncio.get_event_loop().time()
        await system_components["engine"].execute_workflow(
            system_components["workflow"],
            {"input": "Test input for optimization"}
        )
        optimized_time = asyncio.get_event_loop().time() - start_time
        assert optimized_time < 3.0  # Should execute faster after optimization 