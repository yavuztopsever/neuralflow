"""
API utilities for the LangGraph application.
This module provides functionality for HTTP endpoints and request/response management.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager
from utils.logging.manager import LogManager
from core.security import SecurityManager, User
from core.workflow import WorkflowManager
from core.state import StateManager
from core.events import EventManager
from core.metrics import MetricsCollector, MetricsReporter
from core.validation import InputValidator

logger = logging.getLogger(__name__)

class APIManager:
    """Manages API endpoints and request/response handling."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the API manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self.security_manager = SecurityManager(self.config)
        self.workflow_manager = WorkflowManager(self.config)
        self.state_manager = StateManager(self.config)
        self.event_manager = EventManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        self.metrics_reporter = MetricsReporter(self.metrics_collector, self.config)
        self.input_validator = InputValidator(self.config)
        
        self.app = FastAPI(
            title="LangGraph API",
            description="API for the LangGraph application",
            version="1.0.0"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup API middleware."""
        try:
            # CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
            
            # Request logging middleware
            @self.app.middleware("http")
            async def log_requests(request: Request, call_next):
                start_time = datetime.now()
                response = await call_next(request)
                duration = (datetime.now() - start_time).total_seconds()
                
                logger.info(
                    f"Request: {request.method} {request.url.path} "
                    f"Status: {response.status_code} Duration: {duration:.2f}s"
                )
                
                return response
            
            logger.info("Setup API middleware")
        except Exception as e:
            logger.error(f"Failed to setup API middleware: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes."""
        try:
            # Authentication routes
            self.app.post("/auth/login")(self._handle_login)
            self.app.post("/auth/refresh")(self._handle_refresh_token)
            
            # Workflow routes
            self.app.post("/workflows")(self._handle_create_workflow)
            self.app.get("/workflows")(self._handle_list_workflows)
            self.app.get("/workflows/{workflow_id}")(self._handle_get_workflow)
            self.app.put("/workflows/{workflow_id}")(self._handle_update_workflow)
            self.app.delete("/workflows/{workflow_id}")(self._handle_delete_workflow)
            self.app.post("/workflows/{workflow_id}/execute")(self._handle_execute_workflow)
            
            # State routes
            self.app.get("/states")(self._handle_list_states)
            self.app.get("/states/{state_id}")(self._handle_get_state)
            self.app.delete("/states/{state_id}")(self._handle_delete_state)
            
            # Event routes
            self.app.get("/events")(self._handle_list_events)
            self.app.get("/events/{event_id}")(self._handle_get_event)
            
            # Metrics routes
            self.app.get("/metrics")(self._handle_get_metrics)
            self.app.get("/metrics/reports")(self._handle_get_reports)
            
            # User routes
            self.app.post("/users")(self._handle_create_user)
            self.app.get("/users")(self._handle_list_users)
            self.app.get("/users/{user_id}")(self._handle_get_user)
            self.app.put("/users/{user_id}")(self._handle_update_user)
            self.app.delete("/users/{user_id}")(self._handle_delete_user)
            
            # Static files
            self.app.mount("/static", StaticFiles(directory="static"), name="static")
            
            logger.info("Setup API routes")
        except Exception as e:
            logger.error(f"Failed to setup API routes: {e}")
            raise
    
    async def _get_current_user(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="auth/login"))) -> User:
        """Get current user from token.
        
        Args:
            token: JWT token
            
        Returns:
            Current user
            
        Raises:
            HTTPException: If token is invalid
        """
        user = self.security_manager.verify_token(token)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        return user
    
    async def _handle_login(self, request: Request) -> Dict[str, Any]:
        """Handle login request.
        
        Args:
            request: HTTP request
            
        Returns:
            Login response
            
        Raises:
            HTTPException: If login fails
        """
        try:
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                raise HTTPException(
                    status_code=400,
                    detail="Username and password required"
                )
            
            token = self.security_manager.authenticate(username, password)
            if not token:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid username or password"
                )
            
            return {'token': token}
        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_refresh_token(self, request: Request) -> Dict[str, Any]:
        """Handle token refresh request.
        
        Args:
            request: HTTP request
            
        Returns:
            Token refresh response
            
        Raises:
            HTTPException: If refresh fails
        """
        try:
            data = await request.json()
            token = data.get('token')
            
            if not token:
                raise HTTPException(
                    status_code=400,
                    detail="Token required"
                )
            
            user = self.security_manager.verify_token(token)
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token"
                )
            
            new_token = self.security_manager._generate_token(user)
            return {'token': new_token}
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_create_workflow(self, request: Request,
                                    current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle workflow creation request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            Workflow creation response
            
        Raises:
            HTTPException: If creation fails
        """
        try:
            data = await request.json()
            
            # Validate input
            self.input_validator.validate_workflow(data)
            
            # Create workflow
            workflow = self.workflow_manager.create_workflow(
                data['id'],
                data['steps']
            )
            
            return workflow.to_dict()
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_list_workflows(self, request: Request,
                                   current_user: User = Depends(_get_current_user)) -> List[Dict[str, Any]]:
        """Handle workflow listing request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            List of workflows
        """
        try:
            workflows = self.workflow_manager.get_workflows()
            return [w.to_dict() for w in workflows]
        except Exception as e:
            logger.error(f"Workflow listing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_get_workflow(self, request: Request,
                                 workflow_id: str,
                                 current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle workflow retrieval request.
        
        Args:
            request: HTTP request
            workflow_id: Workflow ID
            current_user: Current user
            
        Returns:
            Workflow data
            
        Raises:
            HTTPException: If retrieval fails
        """
        try:
            workflow = self.workflow_manager.get_workflow(workflow_id)
            if not workflow:
                raise HTTPException(
                    status_code=404,
                    detail=f"Workflow {workflow_id} not found"
                )
            
            return workflow.to_dict()
        except Exception as e:
            logger.error(f"Workflow retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_update_workflow(self, request: Request,
                                    workflow_id: str,
                                    current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle workflow update request.
        
        Args:
            request: HTTP request
            workflow_id: Workflow ID
            current_user: Current user
            
        Returns:
            Updated workflow data
            
        Raises:
            HTTPException: If update fails
        """
        try:
            data = await request.json()
            
            # Validate input
            self.input_validator.validate_workflow(data)
            
            # Update workflow
            success = self.workflow_manager.update_workflow(
                workflow_id,
                data.get('steps')
            )
            
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Workflow {workflow_id} not found"
                )
            
            workflow = self.workflow_manager.get_workflow(workflow_id)
            return workflow.to_dict()
        except Exception as e:
            logger.error(f"Workflow update failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_delete_workflow(self, request: Request,
                                    workflow_id: str,
                                    current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle workflow deletion request.
        
        Args:
            request: HTTP request
            workflow_id: Workflow ID
            current_user: Current user
            
        Returns:
            Deletion response
            
        Raises:
            HTTPException: If deletion fails
        """
        try:
            success = self.workflow_manager.delete_workflow(workflow_id)
            
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Workflow {workflow_id} not found"
                )
            
            return {'message': f"Workflow {workflow_id} deleted"}
        except Exception as e:
            logger.error(f"Workflow deletion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_execute_workflow(self, request: Request,
                                     workflow_id: str,
                                     current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle workflow execution request.
        
        Args:
            request: HTTP request
            workflow_id: Workflow ID
            current_user: Current user
            
        Returns:
            Execution results
            
        Raises:
            HTTPException: If execution fails
        """
        try:
            data = await request.json()
            context = data.get('context', {})
            
            # Create state
            state = self.state_manager.create_state(
                workflow_id,
                context,
                {'user_id': current_user.id}
            )
            
            # Emit event
            self.event_manager.emit_event(
                'workflow_started',
                workflow_id,
                {'state_id': state.state_id}
            )
            
            try:
                # Execute workflow
                results = self.workflow_manager.execute_workflow(
                    workflow_id,
                    context
                )
                
                # Update state
                self.state_manager.update_state(
                    state.state_id,
                    results,
                    'completed'
                )
                
                # Emit event
                self.event_manager.emit_event(
                    'workflow_completed',
                    workflow_id,
                    {'state_id': state.state_id, 'results': results}
                )
                
                # Record metrics
                self.metrics_collector.record_metric(
                    'workflow_duration',
                    workflow_id,
                    (datetime.now() - datetime.fromisoformat(state.created)).total_seconds()
                )
                
                return results
            except Exception as e:
                # Update state
                self.state_manager.update_state(
                    state.state_id,
                    {},
                    'failed',
                    str(e)
                )
                
                # Emit event
                self.event_manager.emit_event(
                    'workflow_failed',
                    workflow_id,
                    {'state_id': state.state_id, 'error': str(e)}
                )
                
                raise
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_list_states(self, request: Request,
                                current_user: User = Depends(_get_current_user)) -> List[Dict[str, Any]]:
        """Handle state listing request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            List of states
        """
        try:
            states = self.state_manager.get_states()
            return [s.to_dict() for s in states]
        except Exception as e:
            logger.error(f"State listing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_get_state(self, request: Request,
                              state_id: str,
                              current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle state retrieval request.
        
        Args:
            request: HTTP request
            state_id: State ID
            current_user: Current user
            
        Returns:
            State data
            
        Raises:
            HTTPException: If retrieval fails
        """
        try:
            state = self.state_manager.get_state(state_id)
            if not state:
                raise HTTPException(
                    status_code=404,
                    detail=f"State {state_id} not found"
                )
            
            return state.to_dict()
        except Exception as e:
            logger.error(f"State retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_delete_state(self, request: Request,
                                 state_id: str,
                                 current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle state deletion request.
        
        Args:
            request: HTTP request
            state_id: State ID
            current_user: Current user
            
        Returns:
            Deletion response
            
        Raises:
            HTTPException: If deletion fails
        """
        try:
            success = self.state_manager.delete_state(state_id)
            
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"State {state_id} not found"
                )
            
            return {'message': f"State {state_id} deleted"}
        except Exception as e:
            logger.error(f"State deletion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_list_events(self, request: Request,
                                current_user: User = Depends(_get_current_user)) -> List[Dict[str, Any]]:
        """Handle event listing request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            List of events
        """
        try:
            events = self.event_manager.get_events()
            return [e.to_dict() for e in events]
        except Exception as e:
            logger.error(f"Event listing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_get_event(self, request: Request,
                              event_id: str,
                              current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle event retrieval request.
        
        Args:
            request: HTTP request
            event_id: Event ID
            current_user: Current user
            
        Returns:
            Event data
            
        Raises:
            HTTPException: If retrieval fails
        """
        try:
            events = self.event_manager.get_events(event_id=event_id)
            if not events:
                raise HTTPException(
                    status_code=404,
                    detail=f"Event {event_id} not found"
                )
            
            return events[0].to_dict()
        except Exception as e:
            logger.error(f"Event retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_get_metrics(self, request: Request,
                                current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle metrics retrieval request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            Metrics data
        """
        try:
            return self.metrics_collector.get_metric_stats()
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_get_reports(self, request: Request,
                                current_user: User = Depends(_get_current_user)) -> List[Dict[str, Any]]:
        """Handle metrics reports retrieval request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            List of reports
        """
        try:
            return self.metrics_reporter.get_report_history()
        except Exception as e:
            logger.error(f"Reports retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_create_user(self, request: Request,
                                current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle user creation request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            User creation response
            
        Raises:
            HTTPException: If creation fails
        """
        try:
            # Check authorization
            if not self.security_manager.authorize(current_user, ['admin']):
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized"
                )
            
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            roles = data.get('roles', [])
            metadata = data.get('metadata')
            
            if not username or not password:
                raise HTTPException(
                    status_code=400,
                    detail="Username and password required"
                )
            
            user = self.security_manager.create_user(
                username,
                password,
                roles,
                metadata
            )
            
            return user.to_dict()
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_list_users(self, request: Request,
                               current_user: User = Depends(_get_current_user)) -> List[Dict[str, Any]]:
        """Handle user listing request.
        
        Args:
            request: HTTP request
            current_user: Current user
            
        Returns:
            List of users
        """
        try:
            # Check authorization
            if not self.security_manager.authorize(current_user, ['admin']):
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized"
                )
            
            users = self.security_manager.get_users()
            return [u.to_dict() for u in users]
        except Exception as e:
            logger.error(f"User listing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_get_user(self, request: Request,
                             user_id: str,
                             current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle user retrieval request.
        
        Args:
            request: HTTP request
            user_id: User ID
            current_user: Current user
            
        Returns:
            User data
            
        Raises:
            HTTPException: If retrieval fails
        """
        try:
            # Check authorization
            if not self.security_manager.authorize(current_user, ['admin']) and current_user.id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized"
                )
            
            user = self.security_manager.get_user(user_id)
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail=f"User {user_id} not found"
                )
            
            return user.to_dict()
        except Exception as e:
            logger.error(f"User retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_update_user(self, request: Request,
                                user_id: str,
                                current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle user update request.
        
        Args:
            request: HTTP request
            user_id: User ID
            current_user: Current user
            
        Returns:
            Updated user data
            
        Raises:
            HTTPException: If update fails
        """
        try:
            # Check authorization
            if not self.security_manager.authorize(current_user, ['admin']) and current_user.id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized"
                )
            
            data = await request.json()
            password = data.get('password')
            roles = data.get('roles')
            metadata = data.get('metadata')
            
            success = self.security_manager.update_user(
                user_id,
                password,
                roles,
                metadata
            )
            
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"User {user_id} not found"
                )
            
            user = self.security_manager.get_user(user_id)
            return user.to_dict()
        except Exception as e:
            logger.error(f"User update failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def _handle_delete_user(self, request: Request,
                                user_id: str,
                                current_user: User = Depends(_get_current_user)) -> Dict[str, Any]:
        """Handle user deletion request.
        
        Args:
            request: HTTP request
            user_id: User ID
            current_user: Current user
            
        Returns:
            Deletion response
            
        Raises:
            HTTPException: If deletion fails
        """
        try:
            # Check authorization
            if not self.security_manager.authorize(current_user, ['admin']):
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized"
                )
            
            success = self.security_manager.delete_user(user_id)
            
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"User {user_id} not found"
                )
            
            return {'message': f"User {user_id} deleted"}
        except Exception as e:
            logger.error(f"User deletion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            ) 