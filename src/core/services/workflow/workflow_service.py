"""
Workflow service for the LangGraph project.
This service provides workflow management capabilities.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

from ..workflow.workflow_manager import WorkflowManager, WorkflowConfig, WorkflowState
from .graph_service import GraphService

class WorkflowService:
    """Service for managing workflows in the LangGraph system."""
    
    def __init__(self):
        """Initialize the workflow service."""
        self.workflows: Dict[str, WorkflowManager] = {}
        self.graph_service = GraphService()
        self.history: List[Dict[str, Any]] = []
    
    def create_workflow(self, name: str, config: Optional[WorkflowConfig] = None) -> WorkflowManager:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            config: Optional workflow configuration
            
        Returns:
            WorkflowManager: Created workflow manager
        """
        workflow = WorkflowManager(config=config)
        self.workflows[name] = workflow
        return workflow
    
    def get_workflow(self, name: str) -> Optional[WorkflowManager]:
        """
        Get a workflow by name.
        
        Args:
            name: Workflow name
            
        Returns:
            Optional[WorkflowManager]: Workflow if found, None otherwise
        """
        return self.workflows.get(name)
    
    async def run_workflow(
        self,
        name: str,
        user_query: str,
        **kwargs
    ) -> str:
        """
        Run a workflow.
        
        Args:
            name: Workflow name
            user_query: User query to process
            **kwargs: Additional arguments
            
        Returns:
            str: Workflow result
        """
        workflow = self.get_workflow(name)
        if not workflow:
            raise ValueError(f"Workflow not found: {name}")
        
        try:
            result = await workflow.run(user_query, **kwargs)
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'workflow': name,
                'query': user_query,
                'result': result,
                'kwargs': kwargs
            })
            
            return result
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'workflow': name,
                'query': user_query,
                'error': str(e),
                'kwargs': kwargs
            })
            raise
    
    async def run_workflow_with_progress(
        self,
        name: str,
        user_query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run a workflow with progress updates.
        
        Args:
            name: Workflow name
            user_query: User query to process
            **kwargs: Additional arguments
            
        Returns:
            List[Dict[str, Any]]: Progress updates
        """
        workflow = self.get_workflow(name)
        if not workflow:
            raise ValueError(f"Workflow not found: {name}")
        
        try:
            updates = []
            async for progress, message in workflow.run_with_progress(user_query, **kwargs):
                updates.append({
                    'timestamp': datetime.now(),
                    'progress': progress,
                    'message': message
                })
            
            # Record in history
            self.history.append({
                'timestamp': datetime.now(),
                'workflow': name,
                'query': user_query,
                'updates': updates,
                'kwargs': kwargs
            })
            
            return updates
            
        except Exception as e:
            # Record error in history
            self.history.append({
                'timestamp': datetime.now(),
                'workflow': name,
                'query': user_query,
                'error': str(e),
                'kwargs': kwargs
            })
            raise
    
    def get_workflow_state(self, name: str) -> Dict[str, Any]:
        """
        Get the current state of a workflow.
        
        Args:
            name: Workflow name
            
        Returns:
            Dict[str, Any]: Workflow state
        """
        workflow = self.get_workflow(name)
        if not workflow:
            raise ValueError(f"Workflow not found: {name}")
        return workflow.get_state()
    
    def set_workflow_state(self, name: str, state: Dict[str, Any]) -> None:
        """
        Set the state of a workflow.
        
        Args:
            name: Workflow name
            state: New state
        """
        workflow = self.get_workflow(name)
        if not workflow:
            raise ValueError(f"Workflow not found: {name}")
        workflow.set_state(state)
    
    def save_workflow_checkpoint(self, name: str, checkpoint_id: str) -> None:
        """
        Save a workflow checkpoint.
        
        Args:
            name: Workflow name
            checkpoint_id: Checkpoint ID
        """
        workflow = self.get_workflow(name)
        if not workflow:
            raise ValueError(f"Workflow not found: {name}")
        workflow.save_checkpoint(checkpoint_id)
    
    def load_workflow_checkpoint(self, name: str, checkpoint_id: str) -> None:
        """
        Load a workflow checkpoint.
        
        Args:
            name: Workflow name
            checkpoint_id: Checkpoint ID
        """
        workflow = self.get_workflow(name)
        if not workflow:
            raise ValueError(f"Workflow not found: {name}")
        workflow.load_checkpoint(checkpoint_id)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the workflow execution history.
        
        Returns:
            List[Dict[str, Any]]: Execution history
        """
        return self.history
    
    def clear_history(self) -> None:
        """Clear the workflow execution history."""
        self.history = []
    
    def reset(self) -> None:
        """Reset the workflow service to its initial state."""
        self.workflows = {}
        self.graph_service.reset()
        self.history = []

__all__ = ['WorkflowService'] 