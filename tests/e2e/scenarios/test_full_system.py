"""
End-to-end tests for the full system workflow.
"""
import pytest
import os
import json
import asyncio
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from main import create_app
from graph.graph_workflow import GraphWorkflow
from models.gguf_wrapper.mock_llm import MockLLM
from config.config import Config


class TestFullSystemE2E:
    """End-to-end tests for the full LangGraph system."""

    @pytest.fixture
    def mock_env_vars(self):
        """Set up test environment variables."""
        original_env = os.environ.copy()
        
        os.environ["DEBUG"] = "true"
        os.environ["TEST_MODE"] = "true"
        os.environ["USE_MOCK_LLM"] = "true"
        os.environ["MINIMAL_MODE"] = "true"
        
        yield
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    @pytest.fixture
    def test_db_dir(self):
        """Create a temporary directory for test databases."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        # Not cleaning up to allow for inspection of test results
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_basic_query_response(self, mock_env_vars, test_db_dir):
        """Test a basic query-response flow through the entire system."""
        # Patch config to use test paths
        with patch("config.config.Config") as MockConfig:
            config = Config()
            config.MEMORY_STORE_PATH = os.path.join(test_db_dir, "memory.db")
            config.VECTOR_STORE_PATH = os.path.join(test_db_dir, "vector_store")
            config.GRAPH_STORE_PATH = os.path.join(test_db_dir, "knowledge_graph.json")
            config.DOCUMENT_STORE_PATH = os.path.join(test_db_dir, "documents")
            config.CHECKPOINTS_PATH = os.path.join(test_db_dir, "checkpoints")
            config.DEBUG = True
            config.USE_MOCK_LLM = True
            MockConfig.return_value = config
            
            # Create mock LLM responses
            with patch("models.gguf_wrapper.mock_llm.MockLLM") as MockLLMClass:
                mock_llm = MockLLM()
                
                # Configure the mock LLM to provide appropriate responses
                mock_llm.invoke = MagicMock(side_effect=[
                    # Context sufficiency evaluation
                    json.dumps({"is_sufficient": True, "confidence": 0.9}),
                    # Response generation
                    "LangGraph is a framework for building stateful workflows with LLMs. It allows you to orchestrate complex, multi-step processes that involve LLMs, tools, and human interaction."
                ])
                
                MockLLMClass.return_value = mock_llm
                
                # Create the application (this will initialize all components)
                app = await create_app()
                
                # Send a query through the system
                query = "What is LangGraph and how does it work?"
                response = await app.process_query(query)
                
                # Assertions
                assert response is not None
                assert isinstance(response, str)
                assert "LangGraph" in response
                assert "framework" in response
                assert "stateful" in response
                
                # Verify the mock LLM was called the expected number of times
                assert mock_llm.invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_env_vars, test_db_dir):
        """Test a multi-turn conversation with the system."""
        # Patch config to use test paths
        with patch("config.config.Config") as MockConfig:
            config = Config()
            config.MEMORY_STORE_PATH = os.path.join(test_db_dir, "memory.db")
            config.VECTOR_STORE_PATH = os.path.join(test_db_dir, "vector_store")
            config.GRAPH_STORE_PATH = os.path.join(test_db_dir, "knowledge_graph.json")
            config.DOCUMENT_STORE_PATH = os.path.join(test_db_dir, "documents")
            config.CHECKPOINTS_PATH = os.path.join(test_db_dir, "checkpoints")
            config.DEBUG = True
            config.USE_MOCK_LLM = True
            MockConfig.return_value = config
            
            # Create mock LLM responses
            with patch("models.gguf_wrapper.mock_llm.MockLLM") as MockLLMClass:
                mock_llm = MockLLM()
                
                # Configure the mock LLM to provide appropriate responses for different turns
                mock_llm.invoke = MagicMock(side_effect=[
                    # First turn
                    json.dumps({"is_sufficient": True, "confidence": 0.9}),
                    "LangGraph is a framework for building stateful workflows with LLMs.",
                    
                    # Second turn
                    json.dumps({"is_sufficient": True, "confidence": 0.9}),
                    "You can create a workflow in LangGraph by defining nodes for each step of your process and connecting them with edges.",
                    
                    # Third turn
                    json.dumps({"is_sufficient": True, "confidence": 0.9}),
                    "Yes, LangGraph supports features like state persistence, checkpointing, and handling of asynchronous operations."
                ])
                
                MockLLMClass.return_value = mock_llm
                
                # Create the application
                app = await create_app()
                
                # Create a unique conversation ID for this test
                conversation_id = "test-conversation-123"
                
                # First turn
                response1 = await app.process_query(
                    "What is LangGraph?",
                    conversation_id=conversation_id
                )
                
                # Second turn
                response2 = await app.process_query(
                    "How do I create a workflow?",
                    conversation_id=conversation_id
                )
                
                # Third turn
                response3 = await app.process_query(
                    "Does it support advanced features like state persistence?",
                    conversation_id=conversation_id
                )
                
                # Assertions
                assert "LangGraph is a framework" in response1
                assert "define" in response2 and "nodes" in response2
                assert "persistence" in response3 and "checkpointing" in response3
                
                # Verify memory was used appropriately (context accumulation)
                assert mock_llm.invoke.call_count == 6  # 2 calls per turn

    @pytest.mark.asyncio
    async def test_web_search_fallback(self, mock_env_vars, test_db_dir):
        """Test that the system falls back to web search when context is insufficient."""
        # Patch config to use test paths
        with patch("config.config.Config") as MockConfig:
            config = Config()
            config.MEMORY_STORE_PATH = os.path.join(test_db_dir, "memory.db")
            config.VECTOR_STORE_PATH = os.path.join(test_db_dir, "vector_store")
            config.GRAPH_STORE_PATH = os.path.join(test_db_dir, "knowledge_graph.json")
            config.DOCUMENT_STORE_PATH = os.path.join(test_db_dir, "documents")
            config.CHECKPOINTS_PATH = os.path.join(test_db_dir, "checkpoints")
            config.DEBUG = True
            config.USE_MOCK_LLM = True
            MockConfig.return_value = config
            
            # Create mock LLM responses
            with patch("models.gguf_wrapper.mock_llm.MockLLM") as MockLLMClass:
                mock_llm = MockLLM()
                
                # Configure the mock LLM to indicate insufficient context and then use web search
                mock_llm.invoke = MagicMock(side_effect=[
                    # Context evaluation - insufficient
                    json.dumps({"is_sufficient": False, "confidence": 0.3}),
                    
                    # Task determination - web search
                    json.dumps({
                        "task_type": "web_search",
                        "search_query": "latest version of LangGraph features"
                    }),
                    
                    # Response generation with web results
                    "According to the latest information, LangGraph version 0.1.2 includes enhancements to state management, improved error handling, and better integration with external tools."
                ])
                
                MockLLMClass.return_value = mock_llm
                
                # Mock the web search component
                with patch("tools.web_search.WebSearch") as MockWebSearch:
                    mock_web_search = MagicMock()
                    mock_web_search.search = AsyncMock(return_value=[
                        {
                            "title": "LangGraph 0.1.2 Release Notes",
                            "snippet": "Version 0.1.2 includes state management enhancements, error handling improvements, and better tool integration.",
                            "url": "https://example.com/langgraph-release"
                        }
                    ])
                    MockWebSearch.return_value = mock_web_search
                    
                    # Create the application
                    app = await create_app()
                    
                    # Send query that will require web search
                    query = "What are the latest features in the newest version of LangGraph?"
                    response = await app.process_query(query)
                    
                    # Assertions
                    assert response is not None
                    assert "0.1.2" in response
                    assert "state management" in response
                    assert "error handling" in response
                    
                    # Verify web search was called
                    mock_web_search.search.assert_called_once()
                    assert mock_llm.invoke.call_count == 3

    @pytest.mark.asyncio
    async def test_document_processing(self, mock_env_vars, test_db_dir):
        """Test that the system can process document references."""
        # Patch config to use test paths
        with patch("config.config.Config") as MockConfig:
            config = Config()
            config.MEMORY_STORE_PATH = os.path.join(test_db_dir, "memory.db")
            config.VECTOR_STORE_PATH = os.path.join(test_db_dir, "vector_store")
            config.GRAPH_STORE_PATH = os.path.join(test_db_dir, "knowledge_graph.json")
            config.DOCUMENT_STORE_PATH = os.path.join(test_db_dir, "documents")
            config.CHECKPOINTS_PATH = os.path.join(test_db_dir, "checkpoints")
            config.DEBUG = True
            config.USE_MOCK_LLM = True
            MockConfig.return_value = config
            
            # Create a test document
            test_doc_path = os.path.join(test_db_dir, "test_document.txt")
            os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)
            with open(test_doc_path, "w") as f:
                f.write("This is a test document about LangGraph, a framework for building stateful workflows.")
            
            # Create mock LLM responses
            with patch("models.gguf_wrapper.mock_llm.MockLLM") as MockLLMClass:
                mock_llm = MockLLM()
                
                # Configure the mock LLM responses
                mock_llm.invoke = MagicMock(side_effect=[
                    # Context sufficiency
                    json.dumps({"is_sufficient": True, "confidence": 0.95}),
                    
                    # Response using document
                    "Based on the document, LangGraph is a framework for building stateful workflows. The document mentions capabilities for workflow creation and state management."
                ])
                
                MockLLMClass.return_value = mock_llm
                
                # Mock document handler to process our test document
                with patch("tools.document_handler.DocumentHandler") as MockDocHandler:
                    mock_doc_handler = MagicMock()
                    mock_doc_handler.process_document = AsyncMock(return_value={
                        "id": "test-doc-1",
                        "content": "This is a test document about LangGraph, a framework for building stateful workflows.",
                        "chunks": ["This is a test document about LangGraph", "LangGraph is a framework for building stateful workflows"]
                    })
                    MockDocHandler.return_value = mock_doc_handler
                    
                    # Create the application
                    app = await create_app()
                    
                    # Process query with document reference
                    query = "Summarize this document about LangGraph"
                    doc_references = [{"id": "test-doc-1", "path": test_doc_path}]
                    
                    response = await app.process_query(
                        query,
                        document_references=doc_references
                    )
                    
                    # Assertions
                    assert response is not None
                    assert "LangGraph" in response
                    assert "framework" in response
                    assert "stateful workflows" in response
                    
                    # Verify document handler was used
                    mock_doc_handler.process_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback(self, mock_env_vars, test_db_dir):
        """Test that the system can recover from errors and provide fallback responses."""
        # Patch config to use test paths
        with patch("config.config.Config") as MockConfig:
            config = Config()
            config.MEMORY_STORE_PATH = os.path.join(test_db_dir, "memory.db")
            config.VECTOR_STORE_PATH = os.path.join(test_db_dir, "vector_store")
            config.GRAPH_STORE_PATH = os.path.join(test_db_dir, "knowledge_graph.json")
            config.DOCUMENT_STORE_PATH = os.path.join(test_db_dir, "documents")
            config.CHECKPOINTS_PATH = os.path.join(test_db_dir, "checkpoints")
            config.DEBUG = True
            config.USE_MOCK_LLM = True
            MockConfig.return_value = config
            
            # Create mock LLM responses
            with patch("models.gguf_wrapper.mock_llm.MockLLM") as MockLLMClass:
                mock_llm = MockLLM()
                
                # Configure the mock LLM to provide a response after recovery
                mock_llm.invoke = MagicMock(return_value="LangGraph is a framework for building stateful workflows with LLMs.")
                
                MockLLMClass.return_value = mock_llm
                
                # Introduce an error in vector search
                with patch("tools.vector_search.VectorSearch") as MockVectorSearch:
                    mock_vector_search = MagicMock()
                    # Make the search method raise an exception
                    mock_vector_search.search = AsyncMock(side_effect=Exception("Vector store connection error"))
                    MockVectorSearch.return_value = mock_vector_search
                    
                    # Create the application
                    app = await create_app()
                    
                    # Process query
                    query = "What is LangGraph?"
                    response = await app.process_query(query)
                    
                    # Assertions
                    assert response is not None
                    assert "LangGraph" in response
                    assert "framework" in response
                    
                    # The system should still generate a response despite the error
                    mock_vector_search.search.assert_called_once()
                    assert mock_llm.invoke.called

    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_env_vars, test_db_dir):
        """Test streaming responses from the system."""
        # Patch config to use test paths
        with patch("config.config.Config") as MockConfig:
            config = Config()
            config.MEMORY_STORE_PATH = os.path.join(test_db_dir, "memory.db")
            config.VECTOR_STORE_PATH = os.path.join(test_db_dir, "vector_store")
            config.GRAPH_STORE_PATH = os.path.join(test_db_dir, "knowledge_graph.json")
            config.DOCUMENT_STORE_PATH = os.path.join(test_db_dir, "documents")
            config.CHECKPOINTS_PATH = os.path.join(test_db_dir, "checkpoints")
            config.DEBUG = True
            config.USE_MOCK_LLM = True
            MockConfig.return_value = config
            
            # Create mock LLM responses
            with patch("models.gguf_wrapper.mock_llm.MockLLM") as MockLLMClass:
                mock_llm = MockLLM()
                
                # Configure the mock LLM
                mock_llm.invoke = MagicMock(return_value=json.dumps({"is_sufficient": True, "confidence": 0.9}))
                
                # Configure streaming response
                stream_chunks = ["LangGraph ", "is ", "a ", "framework ", "for ", "building ", "stateful ", "workflows ", "with ", "LLMs."]
                
                async def mock_stream(*args, **kwargs):
                    for chunk in stream_chunks:
                        yield chunk
                
                mock_llm.stream = mock_stream
                
                MockLLMClass.return_value = mock_llm
                
                # Create the application
                app = await create_app()
                
                # Process streaming query
                query = "What is LangGraph?"
                chunks = []
                async for chunk in app.stream_query(query):
                    chunks.append(chunk)
                
                # Assertions
                assert len(chunks) == len(stream_chunks)
                assert "".join(chunks) == "LangGraph is a framework for building stateful workflows with LLMs."
                
                # Verify the mock LLM was called appropriately
                assert mock_llm.invoke.called