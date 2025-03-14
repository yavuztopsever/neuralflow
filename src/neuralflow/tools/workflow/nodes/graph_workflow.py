from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from .workflow_implementation import WorkflowImplementation, WorkflowState

class WorkflowStatus(Enum):
    INITIAL = "initial"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class WorkflowContext:
    status: WorkflowStatus
    data: Dict[str, Any]
    error: Optional[str] = None
    current_node: Optional[str] = None
    completed_nodes: List[str] = None

    def __post_init__(self):
        if self.completed_nodes is None:
            self.completed_nodes = []

class GraphWorkflow:
    def __init__(self):
        self.context = WorkflowContext(status=WorkflowStatus.INITIAL, data={})
        self.workflow = WorkflowImplementation()
        self.parallel_tasks: List[asyncio.Task] = []

    async def execute_parallel_nodes(self, node_ids: List[str]) -> Dict[str, Any]:
        """Execute multiple nodes in parallel."""
        tasks = []
        for node_id in node_ids:
            task = asyncio.create_task(self.workflow.execute_node(node_id))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {node_id: result for node_id, result in zip(node_ids, results)}

    async def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire workflow with the given input data."""
        try:
            self.context.status = WorkflowStatus.PROCESSING
            self.context.data = input_data.copy()
            
            # Execute the workflow
            result = await self.workflow.execute_workflow(input_data)
            
            self.context.status = WorkflowStatus.COMPLETED
            self.context.data.update(result)
            return result
            
        except Exception as e:
            self.context.status = WorkflowStatus.ERROR
            self.context.error = str(e)
            raise

    def get_workflow_state(self) -> WorkflowState:
        """Get the current state of the workflow."""
        return self.workflow.get_workflow_state()

    def reset_workflow(self):
        """Reset the workflow to its initial state."""
        self.context = WorkflowContext(status=WorkflowStatus.INITIAL, data={})
        self.workflow.reset_workflow()

def create_workflow_graph() -> GraphWorkflow:
    """Create a new workflow graph instance."""
    return GraphWorkflow()

async def run_agent(workflow: GraphWorkflow, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the workflow with the given input data."""
    try:
        return await workflow.execute_workflow(input_data)
    except Exception as e:
        workflow.context.error = str(e)
        raise 