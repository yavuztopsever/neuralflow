from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime

T = TypeVar('T')

class WorkflowNode(ABC, Generic[T]):
    """Base class for workflow nodes."""
    
    def __init__(self, node_id: str, name: str):
        self.node_id = node_id
        self.name = name
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.status: str = "pending"
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the node's logic."""
        pass
    
    @abstractmethod
    async def validate(self) -> bool:
        """Validate the node's inputs and configuration."""
        pass

class WorkflowEdge:
    """Represents a connection between workflow nodes."""
    
    def __init__(self, source_id: str, target_id: str, edge_type: str = "default"):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.conditions: List[Dict[str, Any]] = []
    
    def add_condition(self, condition: Dict[str, Any]) -> None:
        """Add a condition to the edge."""
        self.conditions.append(condition)

class Workflow(ABC):
    """Base class for workflows."""
    
    def __init__(self, workflow_id: str, name: str):
        self.workflow_id = workflow_id
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self.status: str = "pending"
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
    
    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the workflow."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: WorkflowEdge) -> None:
        """Add an edge to the workflow."""
        self.edges.append(edge)
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow."""
        pass
    
    @abstractmethod
    async def validate(self) -> bool:
        """Validate the workflow configuration."""
        pass
    
    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get the dependencies for a specific node."""
        return [edge.source_id for edge in self.edges if edge.target_id == node_id]
    
    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get the dependents for a specific node."""
        return [edge.target_id for edge in self.edges if edge.source_id == node_id]

class WorkflowExecutor:
    """Handles the execution of workflows."""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow by its ID."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        if not await workflow.validate():
            raise ValueError(f"Workflow {workflow_id} validation failed")
        
        return await workflow.execute()
    
    def register_workflow(self, workflow: Workflow) -> None:
        """Register a new workflow."""
        self.workflows[workflow.workflow_id] = workflow 