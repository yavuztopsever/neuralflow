from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid
from pathlib import Path

from ...workflow.base import Workflow, WorkflowNode, WorkflowEdge
from ...workflow.implementations import SequentialWorkflow, ParallelWorkflow

class WorkflowManager:
    """Manages workflow lifecycle and persistence."""
    
    def __init__(self, storage_path: str = "storage/workflows"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_workflows: Dict[str, Workflow] = {}
    
    async def create_workflow(self, name: str, workflow_type: str = "sequential") -> Workflow:
        """Create a new workflow."""
        workflow_id = str(uuid.uuid4())
        
        if workflow_type == "sequential":
            workflow = SequentialWorkflow(workflow_id, name)
        elif workflow_type == "parallel":
            workflow = ParallelWorkflow(workflow_id, name)
        else:
            raise ValueError(f"Unsupported workflow type: {workflow_type}")
        
        self.active_workflows[workflow_id] = workflow
        await self._save_workflow(workflow)
        
        return workflow
    
    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        
        # Try to load from storage
        workflow = await self._load_workflow(workflow_id)
        if workflow:
            self.active_workflows[workflow_id] = workflow
        
        return workflow
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        workflows = []
        
        # Get active workflows
        for workflow in self.active_workflows.values():
            workflows.append(self._workflow_to_dict(workflow))
        
        # Get stored workflows
        for workflow_file in self.storage_path.glob("*.json"):
            workflow_id = workflow_file.stem
            if workflow_id not in self.active_workflows:
                workflow = await self._load_workflow(workflow_id)
                if workflow:
                    workflows.append(self._workflow_to_dict(workflow))
        
        return workflows
    
    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Optional[Workflow]:
        """Update a workflow's properties."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        # Update workflow properties
        for key, value in updates.items():
            if hasattr(workflow, key):
                setattr(workflow, key, value)
        
        workflow.updated_at = datetime.now()
        await self._save_workflow(workflow)
        
        return workflow
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        # Remove from active workflows
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
        
        # Delete from storage
        workflow_file = self.storage_path / f"{workflow_id}.json"
        if workflow_file.exists():
            workflow_file.unlink()
            return True
        
        return False
    
    async def add_node(self, workflow_id: str, node: WorkflowNode) -> Optional[Workflow]:
        """Add a node to a workflow."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        workflow.add_node(node)
        workflow.updated_at = datetime.now()
        await self._save_workflow(workflow)
        
        return workflow
    
    async def add_edge(self, workflow_id: str, edge: WorkflowEdge) -> Optional[Workflow]:
        """Add an edge to a workflow."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        workflow.add_edge(edge)
        workflow.updated_at = datetime.now()
        await self._save_workflow(workflow)
        
        return workflow
    
    async def _save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow to storage."""
        workflow_dict = self._workflow_to_dict(workflow)
        workflow_file = self.storage_path / f"{workflow.workflow_id}.json"
        
        with open(workflow_file, "w") as f:
            json.dump(workflow_dict, f, indent=2, default=str)
    
    async def _load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Load a workflow from storage."""
        workflow_file = self.storage_path / f"{workflow_id}.json"
        if not workflow_file.exists():
            return None
        
        with open(workflow_file, "r") as f:
            workflow_dict = json.load(f)
        
        # Create workflow instance
        workflow_type = workflow_dict["type"]
        if workflow_type == "sequential":
            workflow = SequentialWorkflow(workflow_id, workflow_dict["name"])
        elif workflow_type == "parallel":
            workflow = ParallelWorkflow(workflow_id, workflow_dict["name"])
        else:
            return None
        
        # Restore workflow state
        workflow.status = workflow_dict["status"]
        workflow.created_at = datetime.fromisoformat(workflow_dict["created_at"])
        workflow.updated_at = datetime.fromisoformat(workflow_dict["updated_at"])
        
        # Restore nodes and edges
        for node_dict in workflow_dict["nodes"]:
            node = WorkflowNode(node_dict["node_id"], node_dict["name"])
            node.inputs = node_dict["inputs"]
            node.outputs = node_dict["outputs"]
            node.status = node_dict["status"]
            workflow.add_node(node)
        
        for edge_dict in workflow_dict["edges"]:
            edge = WorkflowEdge(
                edge_dict["source_id"],
                edge_dict["target_id"],
                edge_dict["edge_type"]
            )
            edge.conditions = edge_dict["conditions"]
            workflow.add_edge(edge)
        
        return workflow
    
    def _workflow_to_dict(self, workflow: Workflow) -> Dict[str, Any]:
        """Convert a workflow to a dictionary."""
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "type": workflow.__class__.__name__.lower(),
            "status": workflow.status,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "name": node.name,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "status": node.status
                }
                for node in workflow.nodes.values()
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type,
                    "conditions": edge.conditions
                }
                for edge in workflow.edges
            ]
        } 