"""
Test fixtures for the NeuralFlow system.
Tests the creation and functionality of test fixtures.
"""
import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from neuralflow.core.config import SystemConfig
from neuralflow.core.workflow import Workflow
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

class TestFixtures:
    """Test suite for test fixtures."""
    
    def test_system_config_fixture(self):
        """Test system configuration fixture."""
        config = SystemConfig(
            max_workers=4,
            cache_size=1000,
            max_memory_mb=1024,
            timeout_seconds=30,
            retry_attempts=3,
            batch_size=32,
            enable_metrics=True
        )
        
        assert config.max_workers == 4
        assert config.cache_size == 1000
        assert config.max_memory_mb == 1024
        assert config.timeout_seconds == 30
        assert config.retry_attempts == 3
        assert config.batch_size == 32
        assert config.enable_metrics is True
    
    def test_workflow_fixture(self):
        """Test workflow fixture."""
        # Create nodes
        input_node = InputNode("input", "input", {})
        llm_node = LLMNode("llm", "llm", {"model": "test-model"})
        memory_node = MemoryNode("memory", "memory", {})
        context_node = ContextNode("context", "context", {})
        reasoning_node = ReasoningNode("reasoning", "reasoning", {})
        output_node = OutputNode("output", "output", {})
        
        # Create edges
        edges = [
            Edge("edge1", "input", "llm", "prompt", {}),
            Edge("edge2", "llm", "memory", "response", {}),
            Edge("edge3", "memory", "context", "memory_data", {}),
            Edge("edge4", "context", "reasoning", "context_data", {}),
            Edge("edge5", "reasoning", "output", "reasoning_results", {})
        ]
        
        # Create workflow
        workflow = Workflow(
            "test_workflow",
            "A test workflow",
            [input_node, llm_node, memory_node, context_node, reasoning_node, output_node],
            edges,
            {}
        )
        
        assert workflow.name == "test_workflow"
        assert workflow.description == "A test workflow"
        assert len(workflow.nodes) == 6
        assert len(workflow.edges) == 5
    
    def test_model_fixtures(self):
        """Test model fixtures."""
        config = SystemConfig()
        
        # Test LLM model fixture
        llm_model = LLMModel(config=config)
        assert llm_model is not None
        assert llm_model.config == config
        
        # Test embedding model fixture
        embedding_model = EmbeddingModel(config=config)
        assert embedding_model is not None
        assert embedding_model.config == config
        
        # Test classifier model fixture
        classifier_model = ClassifierModel(config=config)
        assert classifier_model is not None
        assert classifier_model.config == config
    
    def test_storage_fixtures(self):
        """Test storage fixtures."""
        config = SystemConfig()
        
        # Test vector store fixture
        vector_store = VectorStore(config=config)
        assert vector_store is not None
        assert vector_store.config == config
        
        # Test cache fixture
        cache = Cache(config=config)
        assert cache is not None
        assert cache.config == config
        
        # Test database fixture
        database = Database(config=config)
        assert database is not None
        assert database.config == config
    
    def test_ui_fixtures(self):
        """Test UI fixtures."""
        config = SystemConfig()
        
        # Test interface fixture
        interface = Interface(config=config)
        assert interface is not None
        assert interface.config == config
    
    def test_test_data_fixtures(self):
        """Test test data fixtures."""
        # Test sample query fixture
        sample_query = "Test query"
        assert isinstance(sample_query, str)
        assert len(sample_query) > 0
        
        # Test sample context fixture
        sample_context = {
            "user_id": "test_user",
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat()
        }
        assert isinstance(sample_context, dict)
        assert "user_id" in sample_context
        assert "session_id" in sample_context
        assert "timestamp" in sample_context
        
        # Test sample state fixture
        sample_state = {
            "current_node": "input",
            "data": {"input": "Test input"},
            "metadata": {"source": "test"}
        }
        assert isinstance(sample_state, dict)
        assert "current_node" in sample_state
        assert "data" in sample_state
        assert "metadata" in sample_state
    
    def test_mock_fixtures(self):
        """Test mock fixtures."""
        # Test mock LLM response fixture
        mock_llm_response = "Test LLM response"
        assert isinstance(mock_llm_response, str)
        assert len(mock_llm_response) > 0
        
        # Test mock embedding fixture
        mock_embedding = [0.1] * 128
        assert isinstance(mock_embedding, list)
        assert len(mock_embedding) == 128
        assert all(isinstance(x, float) for x in mock_embedding)
        
        # Test mock classification fixture
        mock_classification = {
            "class": "test_class",
            "confidence": 0.95
        }
        assert isinstance(mock_classification, dict)
        assert "class" in mock_classification
        assert "confidence" in mock_classification
    
    def test_error_fixtures(self):
        """Test error fixtures."""
        # Test error response fixture
        error_response = {
            "status": "error",
            "message": "Test error message",
            "code": 400
        }
        assert isinstance(error_response, dict)
        assert error_response["status"] == "error"
        assert "message" in error_response
        assert "code" in error_response
        
        # Test validation error fixture
        validation_error = {
            "field": "test_field",
            "message": "Invalid value",
            "value": None
        }
        assert isinstance(validation_error, dict)
        assert "field" in validation_error
        assert "message" in validation_error
        assert "value" in validation_error
    
    def test_metrics_fixtures(self):
        """Test metrics fixtures."""
        # Test performance metrics fixture
        performance_metrics = {
            "response_time": 0.5,
            "memory_usage": 100 * 1024 * 1024,
            "cpu_usage": 50.0
        }
        assert isinstance(performance_metrics, dict)
        assert "response_time" in performance_metrics
        assert "memory_usage" in performance_metrics
        assert "cpu_usage" in performance_metrics
        
        # Test workflow metrics fixture
        workflow_metrics = {
            "total_executions": 100,
            "success_rate": 0.95,
            "average_response_time": 0.5
        }
        assert isinstance(workflow_metrics, dict)
        assert "total_executions" in workflow_metrics
        assert "success_rate" in workflow_metrics
        assert "average_response_time" in workflow_metrics 