"""
Service tests for the NeuralFlow system.
Tests system services and their interactions.
"""
import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from neuralflow.services.workflow_service import WorkflowService
from neuralflow.services.model_service import ModelService
from neuralflow.services.storage_service import StorageService
from neuralflow.services.ui_service import UIService
from neuralflow.services.auth_service import AuthService
from neuralflow.services.metrics_service import MetricsService
from neuralflow.services.notification_service import NotificationService
from neuralflow.services.scheduling_service import SchedulingService
from neuralflow.core.config import SystemConfig

class TestServices:
    """Test suite for system services."""
    
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
    async def service_components(self, system_config):
        """Create service components for testing."""
        workflow_service = WorkflowService(config=system_config)
        model_service = ModelService(config=system_config)
        storage_service = StorageService(config=system_config)
        ui_service = UIService(config=system_config)
        auth_service = AuthService(config=system_config)
        metrics_service = MetricsService(config=system_config)
        notification_service = NotificationService(config=system_config)
        scheduling_service = SchedulingService(config=system_config)
        
        return {
            "config": system_config,
            "workflow_service": workflow_service,
            "model_service": model_service,
            "storage_service": storage_service,
            "ui_service": ui_service,
            "auth_service": auth_service,
            "metrics_service": metrics_service,
            "notification_service": notification_service,
            "scheduling_service": scheduling_service
        }
    
    @pytest.mark.asyncio
    async def test_workflow_service(self, service_components):
        """Test workflow service functionality."""
        # Test service initialization
        assert service_components["workflow_service"] is not None
        assert service_components["workflow_service"].is_initialized
        
        # Test workflow creation
        workflow_data = {
            "name": "test_workflow",
            "description": "Test workflow",
            "nodes": [
                {"id": "input", "type": "input", "config": {}},
                {"id": "llm", "type": "llm", "config": {"model": "test-model"}},
                {"id": "output", "type": "output", "config": {}}
            ],
            "edges": [
                {"id": "edge1", "source": "input", "target": "llm", "data": {}},
                {"id": "edge2", "source": "llm", "target": "output", "data": {}}
            ]
        }
        
        workflow = await service_components["workflow_service"].create_workflow(workflow_data)
        assert workflow is not None
        assert workflow.name == "test_workflow"
        
        # Test workflow execution
        result = await service_components["workflow_service"].execute_workflow(
            workflow.id,
            {"input": "Test query"}
        )
        assert result is not None
        assert "output" in result
        
        # Test workflow management
        workflows = await service_components["workflow_service"].list_workflows()
        assert workflows is not None
        assert len(workflows) > 0
        
        await service_components["workflow_service"].delete_workflow(workflow.id)
        workflows = await service_components["workflow_service"].list_workflows()
        assert len(workflows) == 0
    
    @pytest.mark.asyncio
    async def test_model_service(self, service_components):
        """Test model service functionality."""
        # Test service initialization
        assert service_components["model_service"] is not None
        assert service_components["model_service"].is_initialized
        
        # Test model operations
        llm_result = await service_components["model_service"].generate_text(
            "Test prompt",
            model="test-model"
        )
        assert llm_result is not None
        assert "text" in llm_result
        
        # Test embedding generation
        text = "Test text for embedding"
        embedding = await service_components["model_service"].generate_embedding(
            text,
            model="test-embedding-model"
        )
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        
        # Test batch embedding generation
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await service_components["model_service"].generate_batch_embeddings(
            texts,
            model="test-embedding-model"
        )
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Test embedding similarity
        similarity = await service_components["model_service"].calculate_similarity(
            embedding,
            embedding
        )
        assert similarity == 1.0
        
        # Test classification
        classification = await service_components["model_service"].classify_text(
            "Test text",
            model="test-classifier"
        )
        assert classification is not None
        assert "class" in classification
        assert "confidence" in classification
        
        # Test model management
        models = await service_components["model_service"].list_models()
        assert models is not None
        assert len(models) > 0
    
    @pytest.mark.asyncio
    async def test_storage_service(self, service_components):
        """Test storage service functionality."""
        # Test service initialization
        assert service_components["storage_service"] is not None
        assert service_components["storage_service"].is_initialized
        
        # Test storage operations
        await service_components["storage_service"].store_data(
            "test_collection",
            "test_doc",
            {"data": "test_value"}
        )
        
        stored_data = await service_components["storage_service"].retrieve_data(
            "test_collection",
            "test_doc"
        )
        assert stored_data is not None
        assert stored_data["data"] == "test_value"
        
        # Test vector storage
        await service_components["storage_service"].store_vector(
            "test_vector",
            [0.1] * 128,
            {"metadata": "test"}
        )
        
        vector_data = await service_components["storage_service"].retrieve_vector(
            "test_vector"
        )
        assert vector_data is not None
        assert len(vector_data["vector"]) == 128
        
        # Test cache operations
        await service_components["storage_service"].cache_data(
            "test_key",
            "test_value",
            ttl=3600
        )
        
        cached_value = await service_components["storage_service"].get_cached_data(
            "test_key"
        )
        assert cached_value == "test_value"
    
    @pytest.mark.asyncio
    async def test_ui_service(self, service_components):
        """Test UI service functionality."""
        # Test service initialization
        assert service_components["ui_service"] is not None
        assert service_components["ui_service"].is_initialized
        
        # Test UI operations
        response = await service_components["ui_service"].handle_input(
            "test_user",
            "Test input"
        )
        assert response is not None
        assert "status" in response
        assert "result" in response
        
        # Test UI state management
        await service_components["ui_service"].update_state(
            "test_user",
            {"key": "value"}
        )
        
        state = await service_components["ui_service"].get_state("test_user")
        assert state is not None
        assert state["key"] == "value"
        
        # Test UI event handling
        await service_components["ui_service"].handle_event(
            "test_user",
            "click",
            {"target": "button1"}
        )
        
        events = await service_components["ui_service"].get_user_events("test_user")
        assert events is not None
        assert len(events) > 0
    
    @pytest.mark.asyncio
    async def test_auth_service(self, service_components):
        """Test authentication service functionality."""
        # Test service initialization
        assert service_components["auth_service"] is not None
        assert service_components["auth_service"].is_initialized
        
        # Test authentication
        auth_result = await service_components["auth_service"].authenticate(
            "test_user",
            "test_password"
        )
        assert auth_result is not None
        assert "token" in auth_result
        assert "expires_at" in auth_result
        
        # Test authorization
        authz_result = await service_components["auth_service"].authorize(
            "test_user",
            "test_resource"
        )
        assert authz_result is not None
        assert "allowed" in authz_result
        assert "permissions" in authz_result
        
        # Test session management
        session = await service_components["auth_service"].create_session(
            "test_user"
        )
        assert session is not None
        assert "session_id" in session
        
        await service_components["auth_service"].end_session(session["session_id"])
        active_sessions = await service_components["auth_service"].get_active_sessions(
            "test_user"
        )
        assert len(active_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_metrics_service(self, service_components):
        """Test metrics service functionality."""
        # Test service initialization
        assert service_components["metrics_service"] is not None
        assert service_components["metrics_service"].is_initialized
        
        # Test metrics collection
        await service_components["metrics_service"].record_metric(
            "test_metric",
            {"value": 1.0, "timestamp": datetime.now().isoformat()}
        )
        
        metrics = await service_components["metrics_service"].get_metrics()
        assert metrics is not None
        assert "test_metric" in metrics
        
        # Test metrics aggregation
        aggregated = await service_components["metrics_service"].aggregate_metrics()
        assert aggregated is not None
        assert "test_metric" in aggregated
        assert "count" in aggregated["test_metric"]
        assert "average" in aggregated["test_metric"]
        
        # Test metrics persistence
        await service_components["metrics_service"].persist_metrics()
        assert await service_components["metrics_service"].are_metrics_persisted()
    
    @pytest.mark.asyncio
    async def test_notification_service(self, service_components):
        """Test notification service functionality."""
        # Test service initialization
        assert service_components["notification_service"] is not None
        assert service_components["notification_service"].is_initialized
        
        # Test notification sending
        await service_components["notification_service"].send_notification(
            "test_user",
            "Test notification",
            "info"
        )
        
        notifications = await service_components["notification_service"].get_notifications(
            "test_user"
        )
        assert notifications is not None
        assert len(notifications) > 0
        
        # Test notification management
        await service_components["notification_service"].mark_as_read(
            "test_user",
            notifications[0]["id"]
        )
        
        unread = await service_components["notification_service"].get_unread_notifications(
            "test_user"
        )
        assert len(unread) == 0
    
    @pytest.mark.asyncio
    async def test_scheduling_service(self, service_components):
        """Test scheduling service functionality."""
        # Test service initialization
        assert service_components["scheduling_service"] is not None
        assert service_components["scheduling_service"].is_initialized
        
        # Test task scheduling
        task = await service_components["scheduling_service"].schedule_task(
            "test_task",
            "*/5 * * * *",  # Every 5 minutes
            {"data": "test"}
        )
        assert task is not None
        assert "task_id" in task
        
        # Test task management
        tasks = await service_components["scheduling_service"].list_tasks()
        assert tasks is not None
        assert len(tasks) > 0
        
        await service_components["scheduling_service"].cancel_task(task["task_id"])
        tasks = await service_components["scheduling_service"].list_tasks()
        assert len(tasks) == 0
        
        # Test task execution
        await service_components["scheduling_service"].execute_task(task["task_id"])
        executions = await service_components["scheduling_service"].get_task_executions(
            task["task_id"]
        )
        assert executions is not None
        assert len(executions) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, service_components):
        """Test error handling in services."""
        # Test workflow service error handling
        with pytest.raises(ValueError):
            await service_components["workflow_service"].create_workflow(None)
        
        # Test model service error handling
        with pytest.raises(ValueError):
            await service_components["model_service"].generate_text(None)
        
        # Test storage service error handling
        with pytest.raises(ValueError):
            await service_components["storage_service"].store_data(None, None, None)
        
        # Test UI service error handling
        with pytest.raises(ValueError):
            await service_components["ui_service"].handle_input(None, None)
        
        # Test auth service error handling
        with pytest.raises(ValueError):
            await service_components["auth_service"].authenticate(None, None)
        
        # Test metrics service error handling
        with pytest.raises(ValueError):
            await service_components["metrics_service"].record_metric(None, None)
        
        # Test notification service error handling
        with pytest.raises(ValueError):
            await service_components["notification_service"].send_notification(None, None, None)
        
        # Test scheduling service error handling
        with pytest.raises(ValueError):
            await service_components["scheduling_service"].schedule_task(None, None, None)
        
        # Test invalid model name
        with pytest.raises(ValueError):
            await service_components["model_service"].generate_text(
                "Test prompt",
                model="nonexistent-model"
            )
        
        # Test invalid embedding input
        with pytest.raises(ValueError):
            await service_components["model_service"].generate_embedding(
                None,
                model="test-embedding-model"
            )
        
        # Test invalid batch embedding input
        with pytest.raises(ValueError):
            await service_components["model_service"].generate_batch_embeddings(
                None,
                model="test-embedding-model"
            )
        
        # Test invalid similarity calculation
        with pytest.raises(ValueError):
            await service_components["model_service"].calculate_similarity(
                None,
                [0.1] * 128
            )
        
        # Test invalid classification input
        with pytest.raises(ValueError):
            await service_components["model_service"].classify_text(
                None,
                model="test-classifier"
            ) 