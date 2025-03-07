"""
Global pytest fixtures for all test types.
"""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import tempfile
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the src directory to the Python path
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import common test fixtures and utilities
from neuralflow.core.workflow.workflow_manager import WorkflowManager, WorkflowConfig, WorkflowState
from neuralflow.core.services.context.context_handler import ContextHandler
from neuralflow.core.services.response.response_generation import ResponseGenerator
from neuralflow.core.tools.memory.memory_manager import MemoryManager
from neuralflow.core.tools.vector.vector_search import VectorSearch

# Test directories
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
DATA_DIR = TEST_DIR / "data"

# ----------------------
# Configuration Fixtures
# ----------------------

@pytest.fixture
def test_config():
    """Provides a test configuration with test-specific paths and settings."""
    config = Config()
    config.MEMORY_STORE_PATH = "src/storage/test_memory.db"
    config.VECTOR_STORE_PATH = "src/storage/test_vector_store"
    config.GRAPH_STORE_PATH = "src/storage/test_graph_store/knowledge_graph.json"
    config.DOCUMENT_STORE_PATH = "src/storage/test_documents"
    config.CHECKPOINTS_PATH = "models/test_checkpoints"
    config.MODEL_PATH = "models/test_models"
    config.DEBUG = True
    config.STATE_SAVE_INTERVAL = 0  # Disable state saving for faster tests
    config.USE_DISTRIBUTED = False
    config.MINIMAL_MODE = True
    config.USE_LOCAL_LLM = True
    return config

@pytest.fixture
def mock_env_setup():
    """Set up environment variables for testing."""
    original_env = os.environ.copy()
    os.environ["DEBUG"] = "true"
    os.environ["TEST_MODE"] = "true"
    os.environ["USE_MOCK_LLM"] = "true"
    os.environ["MINIMAL_MODE"] = "true"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="session")
def test_env() -> Dict[str, str]:
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    return {
        "TESTING": "true",
        "LOG_LEVEL": "DEBUG",
    }

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide access to test data directory."""
    return DATA_DIR

@pytest.fixture(scope="session")
def test_fixtures_dir() -> Path:
    """Provide access to test fixtures directory."""
    return FIXTURES_DIR

@pytest.fixture(scope="function")
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration for testing."""
    return {
        "app_name": "test_app",
        "debug": True,
        "log_level": "DEBUG",
    }

@pytest.fixture(scope="function")
def temp_storage(tmp_path) -> Path:
    """Provide a temporary storage directory for testing."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    return storage_dir

# ----------------------
# Mock Data Fixtures
# ----------------------

@pytest.fixture
def sample_user_query():
    """Returns a sample user query for testing."""
    return "What is LangGraph and how does it work?"

@pytest.fixture
def sample_context():
    """Returns a sample context dictionary for testing."""
    return {
        "query": "What is LangGraph and how does it work?",
        "memory": {
            "short_term": ["User asked about LangGraph workflow"],
            "mid_term": [],
            "long_term": []
        },
        "documents": [
            {"source": "documentation", "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs."}
        ],
        "knowledge_graph": {
            "nodes": [
                {"id": "langchain", "type": "technology", "properties": {"description": "Framework for LLM applications"}},
                {"id": "langgraph", "type": "technology", "properties": {"description": "Library for building multi-actor workflows"}}
            ],
            "edges": [
                {"source": "langgraph", "target": "langchain", "type": "built_on", "properties": {}}
            ]
        },
        "vector_results": [
            {"text": "LangGraph enables stateful workflows using LangChain", "score": 0.92}
        ]
    }

@pytest.fixture
def sample_state():
    """Returns a sample LangGraph state for testing."""
    return {
        "messages": [
            {"role": "user", "content": "What is LangGraph?"},
            {"role": "assistant", "content": "LangGraph is a library for building stateful workflows with LLMs."}
        ],
        "context": {
            "query": "What is LangGraph?",
            "memory": {"short_term": [], "mid_term": [], "long_term": []},
            "documents": [],
            "knowledge_graph": {"nodes": [], "edges": []},
            "vector_results": []
        },
        "status": "awaiting_response",
        "checkpoint_id": "test-checkpoint-123"
    }

@pytest.fixture
def sample_conversation_history():
    """Returns a sample conversation history for testing."""
    return [
        {"role": "user", "content": "What is LangGraph?"},
        {"role": "assistant", "content": "LangGraph is a library for building stateful workflows with LLMs."},
        {"role": "user", "content": "How does it compare to LangChain?"},
        {"role": "assistant", "content": "LangGraph is built on top of LangChain and focuses on stateful, graph-based workflows."}
    ]

@pytest.fixture
def sample_document():
    """Returns a sample document for testing."""
    return {
        "id": "doc-123",
        "title": "LangGraph Documentation",
        "content": "LangGraph is a library for orchestrating LLM applications using a stateful workflow system.",
        "metadata": {
            "source": "documentation",
            "created_at": "2023-01-01T00:00:00Z",
            "format": "text"
        }
    }

# ----------------------
# Mock Component Fixtures
# ----------------------

@pytest.fixture
def mock_model_manager():
    """Mock model manager for testing."""
    mock = MagicMock()
    mock.generate_response = AsyncMock(return_value="This is a mock response from the model.")
    mock.stream_response = AsyncMock(return_value=["This ", "is ", "a ", "streamed ", "response."])
    mock.get_embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    mock.analyze_emotion = AsyncMock(return_value="neutral")
    mock.analyze_content = AsyncMock(return_value={"topics": ["technology"], "complexity": "medium"})
    mock.apply_style = AsyncMock(return_value="This is a styled response.")
    mock.get_model_info = MagicMock(return_value={"name": "test-model", "type": "mock"})
    return mock

@pytest.fixture
def mock_memory_manager():
    """Mock memory manager for testing."""
    mock = MagicMock()
    mock.add_to_memory = AsyncMock()
    mock.get_from_memory = AsyncMock(return_value={
        "short_term": ["Recent message about LangGraph"],
        "mid_term": [],
        "long_term": []
    })
    mock.clear_memory = AsyncMock()
    mock.save_conversation = AsyncMock()
    mock.get_conversation_history = AsyncMock(return_value=[])
    return mock

@pytest.fixture
def mock_vector_search():
    """Mock vector search for testing."""
    mock = MagicMock()
    mock.search = AsyncMock(return_value=[
        {"text": "LangGraph is a library for building workflows", "score": 0.95},
        {"text": "LangGraph manages state between steps", "score": 0.87}
    ])
    mock.add_document = AsyncMock()
    mock.delete_document = AsyncMock()
    mock.update_document = AsyncMock()
    return mock

@pytest.fixture
def mock_document_handler():
    """Mock document handler for testing."""
    mock = MagicMock()
    mock.process_document = AsyncMock(return_value={"chunks": ["Document chunk 1", "Document chunk 2"]})
    mock.extract_text = MagicMock(return_value="Extracted text from document")
    mock.load_document = AsyncMock(return_value={"id": "doc-123", "content": "Document content"})
    mock.save_document = AsyncMock()
    return mock

@pytest.fixture
def mock_web_search():
    """Mock web search for testing."""
    mock = MagicMock()
    mock.search = AsyncMock(return_value=[
        {"title": "LangGraph Documentation", "snippet": "Official docs for LangGraph", "url": "https://example.com/langgraph"},
        {"title": "LangGraph Tutorial", "snippet": "Learn to use LangGraph", "url": "https://example.com/tutorial"}
    ])
    return mock

@pytest.fixture
def mock_function_caller():
    """Mock function caller for testing."""
    mock = MagicMock()
    mock.execute_function = AsyncMock(return_value={"status": "success", "result": "Function output"})
    mock.list_available_functions = MagicMock(return_value=["calculate", "format_text", "search_data"])
    mock.get_function_metadata = MagicMock(return_value={"description": "Test function", "parameters": {}})
    return mock

@pytest.fixture
def mock_context_handler():
    """Mock context handler for testing."""
    mock = MagicMock()
    mock.gather_context = AsyncMock(return_value={
        "memory": {"short_term": [], "mid_term": [], "long_term": []},
        "documents": [],
        "knowledge_graph": {"nodes": [], "edges": []},
        "vector_results": []
    })
    mock.evaluate_context_sufficiency = AsyncMock(return_value=True)
    mock.process_document_references = AsyncMock(return_value=[])
    mock.extract_knowledge_graph = AsyncMock(return_value={"nodes": [], "edges": []})
    return mock

@pytest.fixture
def mock_task_execution():
    """Mock task execution for testing."""
    mock = MagicMock()
    mock.execute_task = AsyncMock(return_value={"status": "success", "data": "Task result"})
    mock.determine_task_type = AsyncMock(return_value={"task_type": "web_search"})
    return mock

@pytest.fixture
def mock_response_generation():
    """Mock response generation for testing."""
    mock = MagicMock()
    mock.generate_response = AsyncMock(return_value="This is a test response.")
    mock.stream_response = AsyncMock(return_value=["This ", "is ", "a ", "streamed ", "response."])
    mock.format_prompt = MagicMock(return_value="Formatted prompt text")
    mock.apply_response_style = AsyncMock(return_value="Styled response")
    return mock

@pytest.fixture
def mock_graph_workflow():
    """Mock graph workflow for testing."""
    mock = MagicMock()
    mock.start = AsyncMock(return_value={"status": "success", "response": "Workflow response"})
    mock.continue_workflow = AsyncMock(return_value={"status": "success", "response": "Continued workflow response"})
    mock.create_workflow = MagicMock()
    mock.process_context = AsyncMock()
    mock.execute_task = AsyncMock()
    mock.generate_response = AsyncMock()
    mock.update_memory = AsyncMock()
    mock.router = MagicMock(return_value="next_step")
    return mock

@pytest.fixture
def mock_graph_search():
    """Mock graph search for testing."""
    mock = MagicMock()
    mock.search = AsyncMock(return_value={"nodes": [], "edges": []})
    mock.add_node = AsyncMock()
    mock.add_edge = AsyncMock()
    mock.update_node = AsyncMock()
    mock.delete_node = AsyncMock()
    return mock

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = MagicMock()
    mock.invoke = MagicMock(return_value="This is a mock LLM response")
    mock.stream = AsyncMock(return_value=["This ", "is ", "a ", "mock ", "streamed ", "response"])
    return mock

# ----------------------
# Test File Fixtures
# ----------------------

@pytest.fixture
def temp_test_file():
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write("This is test content for file operations.")
        file_path = temp_file.name
    
    yield file_path
    
    # Clean up
    if os.path.exists(file_path):
        os.unlink(file_path)

@pytest.fixture
def temp_test_dir():
    """Create a temporary test directory."""
    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    # Clean up
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# ----------------------
# Async Test Helpers
# ----------------------

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Export commonly used test fixtures
__all__ = [
    'WorkflowManager',
    'WorkflowConfig',
    'WorkflowState',
    'ContextHandler',
    'ResponseGenerator',
    'MemoryManager',
    'VectorSearch'
]

@pytest.fixture
def mock_context_handler():
    """Create a mock context handler."""
    handler = MagicMock()
    handler.memory_manager = MagicMock()
    handler.memory_manager.get_short_term_memory = AsyncMock(return_value=[])
    handler.memory_manager.get_mid_term_memory = AsyncMock(return_value=[])
    handler.memory_manager.get_interactions = AsyncMock(return_value=[])
    return handler

@pytest.fixture
def mock_task_execution():
    """Create a mock task execution."""
    executor = MagicMock()
    executor.execute = AsyncMock(return_value={"execution_result": "Task executed"})
    return executor

@pytest.fixture
def mock_response_generation():
    """Create a mock response generator."""
    generator = MagicMock()
    generator.generate = AsyncMock(return_value="Test response")
    return generator

@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager."""
    manager = MagicMock()
    manager.save_interaction = AsyncMock()
    return manager

@pytest.fixture
def sample_user_query():
    """Create a sample user query."""
    return "What is LangGraph?"