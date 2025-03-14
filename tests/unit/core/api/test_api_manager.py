"""
Unit tests for API manager functionality.
"""
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException

from src.core.api import APIManager, APIRoute, APIMethod, APIResponse, APIError

class TestAPIManager:
    """Test suite for API manager functionality."""
    
    @pytest.fixture
    def api_manager(self):
        """Create an API manager for testing."""
        return APIManager(
            title="Test API",
            version="1.0.0",
            description="Test API for unit testing"
        )
    
    @pytest.fixture
    def sample_route(self):
        """Create a sample API route for testing."""
        async def handler(request: Dict[str, Any]) -> APIResponse:
            return APIResponse(
                status_code=200,
                data={"message": "Success"}
            )
        
        return APIRoute(
            path="/test",
            method=APIMethod.GET,
            handler=handler,
            description="Test endpoint",
            parameters={
                "query": {
                    "name": "test_param",
                    "type": "string",
                    "required": False
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_api_manager_initialization(self, api_manager):
        """Test API manager initialization."""
        assert api_manager.title == "Test API"
        assert api_manager.version == "1.0.0"
        assert api_manager.description == "Test API for unit testing"
        assert isinstance(api_manager.app, FastAPI)
        assert api_manager._routes == {}
    
    @pytest.mark.asyncio
    async def test_route_management(self, api_manager, sample_route):
        """Test route management operations."""
        # Register route
        await api_manager.register_route(sample_route)
        assert sample_route.path in api_manager._routes
        stored_route = api_manager._routes[sample_route.path]
        assert stored_route.method == sample_route.method
        assert stored_route.handler == sample_route.handler
        
        # Get route
        retrieved_route = await api_manager.get_route(sample_route.path)
        assert retrieved_route is not None
        assert retrieved_route.path == sample_route.path
        
        # Update route
        async def updated_handler(request: Dict[str, Any]) -> APIResponse:
            return APIResponse(
                status_code=200,
                data={"message": "Updated"}
            )
        
        updated_route = APIRoute(
            path=sample_route.path,
            method=sample_route.method,
            handler=updated_handler,
            description=sample_route.description,
            parameters=sample_route.parameters
        )
        await api_manager.update_route(updated_route)
        retrieved_route = await api_manager.get_route(sample_route.path)
        assert retrieved_route.handler == updated_handler
        
        # Delete route
        await api_manager.delete_route(sample_route.path)
        assert sample_route.path not in api_manager._routes
    
    @pytest.mark.asyncio
    async def test_route_validation(self, api_manager):
        """Test route validation."""
        # Test invalid path
        with pytest.raises(ValueError):
            await api_manager.register_route(APIRoute(
                path="invalid_path",  # Missing leading slash
                method=APIMethod.GET,
                handler=lambda x: None,
                description="Test"
            ))
        
        # Test invalid method
        with pytest.raises(ValueError):
            await api_manager.register_route(APIRoute(
                path="/test",
                method="INVALID",  # type: ignore
                handler=lambda x: None,
                description="Test"
            ))
        
        # Test invalid handler
        with pytest.raises(ValueError):
            await api_manager.register_route(APIRoute(
                path="/test",
                method=APIMethod.GET,
                handler=None,
                description="Test"
            ))
    
    @pytest.mark.asyncio
    async def test_request_handling(self, api_manager, sample_route):
        """Test request handling functionality."""
        await api_manager.register_route(sample_route)
        
        # Test successful request
        request = {"query": {"test_param": "value"}}
        response = await api_manager.handle_request(sample_route.path, sample_route.method, request)
        assert response.status_code == 200
        assert response.data["message"] == "Success"
        
        # Test request with missing required parameter
        request = {"query": {}}
        with pytest.raises(HTTPException) as exc_info:
            await api_manager.handle_request(sample_route.path, sample_route.method, request)
        assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_error_handling(self, api_manager):
        """Test error handling functionality."""
        async def error_handler(request: Dict[str, Any]) -> APIResponse:
            raise APIError(
                status_code=500,
                message="Internal server error"
            )
        
        route = APIRoute(
            path="/error",
            method=APIMethod.GET,
            handler=error_handler,
            description="Error test endpoint"
        )
        await api_manager.register_route(route)
        
        # Test error handling
        with pytest.raises(HTTPException) as exc_info:
            await api_manager.handle_request(route.path, route.method, {})
        assert exc_info.value.status_code == 500
        assert str(exc_info.value.detail) == "Internal server error"
    
    @pytest.mark.asyncio
    async def test_route_metrics(self, api_manager, sample_route):
        """Test route metrics collection."""
        await api_manager.register_route(sample_route)
        
        # Make some requests
        for _ in range(3):
            await api_manager.handle_request(sample_route.path, sample_route.method, {})
        
        # Get metrics
        metrics = await api_manager.get_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "total_routes" in metrics
        assert "total_requests" in metrics
        assert "request_methods" in metrics
    
    @pytest.mark.asyncio
    async def test_route_error_handling(self, api_manager):
        """Test route error handling."""
        # Test invalid route
        with pytest.raises(ValueError):
            await api_manager.register_route(None)
        
        # Test invalid route path
        with pytest.raises(ValueError):
            await api_manager.get_route(None)
        
        # Test getting non-existent route
        route = await api_manager.get_route("/non_existent")
        assert route is None
        
        # Test updating non-existent route
        with pytest.raises(KeyError):
            await api_manager.update_route(APIRoute(
                path="/non_existent",
                method=APIMethod.GET,
                handler=lambda x: None,
                description="Test"
            ))
        
        # Test deleting non-existent route
        with pytest.raises(KeyError):
            await api_manager.delete_route("/non_existent")
    
    @pytest.mark.asyncio
    async def test_route_method_validation(self, api_manager):
        """Test route method validation."""
        # Test different HTTP methods
        methods = [APIMethod.GET, APIMethod.POST, APIMethod.PUT, APIMethod.DELETE]
        for method in methods:
            route = APIRoute(
                path=f"/test_{method.value.lower()}",
                method=method,
                handler=lambda x: APIResponse(status_code=200, data={}),
                description=f"Test {method.value} endpoint"
            )
            await api_manager.register_route(route)
            assert route.path in api_manager._routes
            assert api_manager._routes[route.path].method == method
        
        # Test handling request with wrong method
        with pytest.raises(HTTPException) as exc_info:
            await api_manager.handle_request("/test_get", APIMethod.POST, {})
        assert exc_info.value.status_code == 405
    
    @pytest.mark.asyncio
    async def test_route_parameter_validation(self, api_manager):
        """Test route parameter validation."""
        async def handler(request: Dict[str, Any]) -> APIResponse:
            return APIResponse(status_code=200, data=request)
        
        route = APIRoute(
            path="/test_params",
            method=APIMethod.POST,
            handler=handler,
            description="Test parameter validation",
            parameters={
                "query": {
                    "required_param": {
                        "type": "string",
                        "required": True
                    },
                    "optional_param": {
                        "type": "integer",
                        "required": False
                    }
                },
                "body": {
                    "data": {
                        "type": "object",
                        "required": True,
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        }
                    }
                }
            }
        )
        await api_manager.register_route(route)
        
        # Test valid request
        request = {
            "query": {"required_param": "value", "optional_param": "42"},
            "body": {"data": {"name": "test", "age": 25}}
        }
        response = await api_manager.handle_request(route.path, route.method, request)
        assert response.status_code == 200
        
        # Test missing required parameter
        request = {
            "query": {"optional_param": "42"},
            "body": {"data": {"name": "test", "age": 25}}
        }
        with pytest.raises(HTTPException) as exc_info:
            await api_manager.handle_request(route.path, route.method, request)
        assert exc_info.value.status_code == 400
        
        # Test invalid parameter type
        request = {
            "query": {"required_param": "value", "optional_param": "invalid"},
            "body": {"data": {"name": "test", "age": 25}}
        }
        with pytest.raises(HTTPException) as exc_info:
            await api_manager.handle_request(route.path, route.method, request)
        assert exc_info.value.status_code == 400 