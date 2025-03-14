"""
Task execution engine for LangGraph.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from ..graph.nodes import Node
from ..graph.edges import Edge
from ..graph.workflows import Workflow

@dataclass
class ExecutionContext:
    """Context for task execution."""
    workflow: Workflow
    inputs: Dict[str, Any]
    state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class Executor:
    """Base class for task execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute a workflow with the given context."""
        raise NotImplementedError
    
    async def execute_node(self, node: Node, context: ExecutionContext) -> Dict[str, Any]:
        """Execute a single node with the given context."""
        raise NotImplementedError

class SequentialExecutor(Executor):
    """Executor that runs tasks sequentially."""
    pass

class ParallelExecutor(Executor):
    """Executor that can run tasks in parallel."""
    pass

__all__ = ['ExecutionContext', 'Executor', 'SequentialExecutor', 'ParallelExecutor'] 