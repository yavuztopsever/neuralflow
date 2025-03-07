from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
from pathlib import Path

from ..workflow.base import Workflow, WorkflowNode
from ..models.management.workflow_manager import WorkflowManager

logger = logging.getLogger(__name__)

class WorkflowExecutionService:
    """Service for executing and monitoring workflows."""
    
    def __init__(self, workflow_manager: WorkflowManager):
        self.workflow_manager = workflow_manager
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow and return its results."""
        workflow = await self.workflow_manager.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if workflow_id in self.active_executions:
            raise RuntimeError(f"Workflow {workflow_id} is already running")
        
        # Create execution task
        execution_task = asyncio.create_task(self._execute_workflow_task(workflow))
        self.active_executions[workflow_id] = execution_task
        
        try:
            # Wait for execution to complete
            results = await execution_task
            return results
        finally:
            # Clean up
            if workflow_id in self.active_executions:
                del self.active_executions[workflow_id]
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id not in self.active_executions:
            return False
        
        execution_task = self.active_executions[workflow_id]
        execution_task.cancel()
        
        try:
            await execution_task
        except asyncio.CancelledError:
            pass
        
        # Update workflow status
        workflow = await self.workflow_manager.get_workflow(workflow_id)
        if workflow:
            workflow.status = "cancelled"
            await self.workflow_manager.update_workflow(workflow_id, {})
        
        return True
    
    async def get_execution_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the current execution status of a workflow."""
        workflow = await self.workflow_manager.get_workflow(workflow_id)
        if not workflow:
            return None
        
        status = {
            "workflow_id": workflow_id,
            "status": workflow.status,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "name": node.name,
                    "status": node.status,
                    "inputs": node.inputs,
                    "outputs": node.outputs
                }
                for node in workflow.nodes.values()
            ]
        }
        
        # Add execution history if available
        if workflow_id in self.execution_history:
            status["execution_history"] = self.execution_history[workflow_id]
        
        return status
    
    async def _execute_workflow_task(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute a workflow task and handle its lifecycle."""
        execution_id = f"{workflow.workflow_id}_{datetime.now().timestamp()}"
        execution_record = {
            "execution_id": execution_id,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            # Initialize execution history
            if workflow.workflow_id not in self.execution_history:
                self.execution_history[workflow.workflow_id] = []
            
            # Execute workflow
            results = await workflow.execute()
            
            # Record successful execution
            execution_record.update({
                "end_time": datetime.now().isoformat(),
                "status": "completed",
                "results": results
            })
            
            return results
            
        except Exception as e:
            # Record failed execution
            execution_record.update({
                "end_time": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
            
        finally:
            # Update execution history
            self.execution_history[workflow.workflow_id].append(execution_record)
            
            # Keep only the last 10 executions
            if len(self.execution_history[workflow.workflow_id]) > 10:
                self.execution_history[workflow.workflow_id] = self.execution_history[workflow.workflow_id][-10:]
    
    async def cleanup_old_executions(self, max_age_days: int = 30) -> None:
        """Clean up old execution records."""
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        for workflow_id in list(self.execution_history.keys()):
            self.execution_history[workflow_id] = [
                record for record in self.execution_history[workflow_id]
                if datetime.fromisoformat(record["start_time"]).timestamp() > cutoff_date
            ]
            
            if not self.execution_history[workflow_id]:
                del self.execution_history[workflow_id] 