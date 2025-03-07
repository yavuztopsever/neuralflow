"""
API tests for the NeuralFlow system.
Tests the REST API endpoints and their functionality.
"""
import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI

from neuralflow.api.app import create_app
from neuralflow.api.routes import (
    workflow_router,
    model_router,
    storage_router,
    ui_router
)
from neuralflow.core.config import SystemConfig

class TestAPI:
    """Test suite for API endpoints."""
    
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
    def app(self, system_config):
        """Create FastAPI application for testing."""
        app = create_app(system_config)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_workflow_endpoints(self, client):
        """Test workflow-related endpoints."""
        # Test workflow creation
        workflow_data = {
            "name": "test_workflow",
            "description": "Test workflow",
            "nodes": [
                {
                    "id": "input",
                    "type": "input",
                    "config": {}
                },
                {
                    "id": "llm",
                    "type": "llm",
                    "config": {
                        "model": "test-model"
                    }
                },
                {
                    "id": "output",
                    "type": "output",
                    "config": {}
                }
            ],
            "edges": [
                {
                    "id": "edge1",
                    "source": "input",
                    "target": "llm",
                    "data": {}
                },
                {
                    "id": "edge2",
                    "source": "llm",
                    "target": "output",
                    "data": {}
                }
            ]
        }
        
        response = client.post("/api/workflows", json=workflow_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        workflow_id = response.json()["workflow_id"]
        
        # Test workflow retrieval
        response = client.get(f"/api/workflows/{workflow_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "test_workflow"
        
        # Test workflow execution
        execution_data = {
            "input": "Test query"
        }
        response = client.post(
            f"/api/workflows/{workflow_id}/execute",
            json=execution_data
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "output" in response.json()
        
        # Test workflow deletion
        response = client.delete(f"/api/workflows/{workflow_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
    
    def test_model_endpoints(self, client):
        """Test model-related endpoints."""
        # Test LLM generation
        llm_data = {
            "prompt": "Test prompt",
            "temperature": 0.7,
            "max_tokens": 100
        }
        response = client.post("/api/models/llm/generate", json=llm_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "text" in response.json()
        
        # Test embedding generation
        embedding_data = {
            "text": "Test text for embedding"
        }
        response = client.post("/api/models/embedding/generate", json=embedding_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "embedding" in response.json()
        
        # Test classification
        classification_data = {
            "text": "Test text for classification"
        }
        response = client.post("/api/models/classifier/classify", json=classification_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "class" in response.json()
        assert "confidence" in response.json()
    
    def test_storage_endpoints(self, client):
        """Test storage-related endpoints."""
        # Test vector store operations
        vector_data = {
            "id": "test_vector",
            "vector": [0.1] * 128,
            "metadata": {
                "created_at": datetime.now().isoformat()
            }
        }
        response = client.post("/api/storage/vector/store", json=vector_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        response = client.get(f"/api/storage/vector/{vector_data['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "vector" in response.json()
        
        # Test cache operations
        cache_data = {
            "key": "test_key",
            "value": "test_value",
            "ttl": 3600
        }
        response = client.post("/api/storage/cache/set", json=cache_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        response = client.get(f"/api/storage/cache/{cache_data['key']}")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["value"] == "test_value"
        
        # Test database operations
        db_data = {
            "collection": "test_collection",
            "document": {
                "id": "test_doc",
                "data": "test_data",
                "metadata": {
                    "created_at": datetime.now().isoformat()
                }
            }
        }
        response = client.post("/api/storage/database/store", json=db_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        response = client.get(
            f"/api/storage/database/{db_data['collection']}/{db_data['document']['id']}"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["data"] == "test_data"
    
    def test_ui_endpoints(self, client):
        """Test UI-related endpoints."""
        # Test user input handling
        input_data = {
            "input": "Test user input",
            "session_id": "test_session"
        }
        response = client.post("/api/ui/input", json=input_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "response" in response.json()
        
        # Test action handling
        action_data = {
            "type": "click",
            "target": "button1",
            "session_id": "test_session"
        }
        response = client.post("/api/ui/action", json=action_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "result" in response.json()
        
        # Test event handling
        event_data = {
            "type": "hover",
            "target": "component1",
            "session_id": "test_session"
        }
        response = client.post("/api/ui/event", json=event_data)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "handled" in response.json()
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test invalid workflow
        response = client.post("/api/workflows", json={})
        assert response.status_code == 400
        assert response.json()["status"] == "error"
        
        # Test invalid model request
        response = client.post("/api/models/llm/generate", json={})
        assert response.status_code == 400
        assert response.json()["status"] == "error"
        
        # Test invalid storage request
        response = client.post("/api/storage/vector/store", json={})
        assert response.status_code == 400
        assert response.json()["status"] == "error"
        
        # Test invalid UI request
        response = client.post("/api/ui/input", json={})
        assert response.status_code == 400
        assert response.json()["status"] == "error"
    
    def test_metrics_endpoints(self, client):
        """Test metrics-related endpoints."""
        # Test system metrics
        response = client.get("/api/metrics/system")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "cpu_usage" in response.json()
        assert "memory_usage" in response.json()
        
        # Test workflow metrics
        response = client.get("/api/metrics/workflow")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "total_executions" in response.json()
        assert "average_response_time" in response.json()
        
        # Test model metrics
        response = client.get("/api/metrics/model")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "total_tokens" in response.json()
        assert "success_rate" in response.json()
        
        # Test storage metrics
        response = client.get("/api/metrics/storage")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "total_documents" in response.json()
        assert "cache_hit_rate" in response.json() 