"""
Base workflow system for NeuralFlow.
Provides unified workflow and graph management capabilities.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum

from ..storage.base import BaseStorage, StorageConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class WorkflowStatus(Enum):
    """Workflow execution status."""
    INITIAL = "initial"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class NodeType(Enum):
    """Types of workflow nodes."""
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    INPUT_PROCESSOR = "input_processor"
    MEMORY_MANAGER = "memory_manager"
    CONTEXT_ORCHESTRATOR = "context_orchestrator"
    REASONING_ENGINE = "reasoning_engine"
    MODEL_MANAGER = "model_manager"
    RESPONSE_ASSEMBLER = "response_assembler"
    RESPONSE_MANAGER = "response_manager"
    METRICS_LOGGING = "metrics_logging"
    TRAIN_MODULE = "train_module"

class WorkflowConfig:
    """Configuration for workflow systems."""
    
    def __init__(self,
                 workflow_id: str,
                 workflow_type: str,
                 storage_config: Optional[StorageConfig] = None,
                 max_nodes: Optional[int] = None,
                 max_parallel_tasks: Optional[int] = None,
                 **kwargs):
        """Initialize workflow configuration.
        
        Args:
            workflow_id: Unique identifier for the workflow
            workflow_type: Type of workflow
            storage_config: Optional storage configuration
            max_nodes: Optional maximum number of nodes
            max_parallel_tasks: Optional maximum parallel tasks
            **kwargs: Additional configuration parameters
        """
        self.id = workflow_id
        self.type = workflow_type
        self.storage_config = storage_config
        self.max_nodes = max_nodes
        self.max_parallel_tasks = max_parallel_tasks
        self.parameters = kwargs
        self.metadata = {}

class Node(BaseModel):
    """Base node model."""
    id: str
    type: NodeType
    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class Edge(BaseModel):
    """Base edge model."""
    id: str
    source_id: str
    target_id: str
    data_key: str
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkflowState(BaseModel):
    """Workflow state model."""
    status: WorkflowStatus = WorkflowStatus.INITIAL
    current_node: Optional[str] = None
    completed_nodes: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseWorkflow(ABC, Generic[T]):
    """Base class for all workflow implementations."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize the workflow.
        
        Args:
            config: Workflow configuration
        """
        self.config = config
        self.created = datetime.now().isoformat()
        self.modified = self.created
        self._storage = self._initialize_storage()
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        self._state = WorkflowState()
        self._parallel_tasks: List[asyncio.Task] = []
    
    def _initialize_storage(self) -> Optional[BaseStorage]:
        """Initialize storage if configured."""
        if self.config.storage_config:
            return BaseStorage(self.config.storage_config)
        return None
    
    def add_node(self, node: Node) -> None:
        """Add a node to the workflow.
        
        Args:
            node: Node to add
        """
        if self.config.max_nodes and len(self._nodes) >= self.config.max_nodes:
            raise ValueError("Maximum number of nodes reached")
        self._nodes[node.id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the workflow.
        
        Args:
            edge: Edge to add
        """
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} not found")
        self._edges[edge.id] = edge
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node if found, None otherwise
        """
        return self._nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID.
        
        Args:
            edge_id: Edge ID
            
        Returns:
            Edge if found, None otherwise
        """
        return self._edges.get(edge_id)
    
    def get_node_edges(self, node_id: str) -> List[Edge]:
        """Get all edges connected to a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of connected edges
        """
        return [
            edge for edge in self._edges.values()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
    
    def get_node_inputs(self, node_id: str) -> List[Edge]:
        """Get input edges for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of input edges
        """
        return [
            edge for edge in self._edges.values()
            if edge.target_id == node_id
        ]
    
    def get_node_outputs(self, node_id: str) -> List[Edge]:
        """Get output edges for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of output edges
        """
        return [
            edge for edge in self._edges.values()
            if edge.source_id == node_id
        ]
    
    async def execute_node(self, node_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a node.
        
        Args:
            node_id: Node ID
            input_data: Input data for the node
            
        Returns:
            Node execution results
        """
        try:
            node = self.get_node(node_id)
            if not node:
                raise ValueError(f"Node {node_id} not found")
            
            # Update state
            self._state.current_node = node_id
            self._state.data.update(input_data)
            
            # Execute node implementation
            result = await self._execute_node_impl(node, input_data)
            
            # Update state
            self._state.completed_nodes.append(node_id)
            self._state.data.update(result)
            
            return result
        except Exception as e:
            logger.error(f"Failed to execute node {node_id}: {e}")
            self._state.error = str(e)
            raise
    
    async def execute_parallel_nodes(self, node_ids: List[str], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple nodes in parallel.
        
        Args:
            node_ids: List of node IDs
            input_data: Input data for the nodes
            
        Returns:
            Combined execution results
        """
        try:
            if self.config.max_parallel_tasks:
                if len(node_ids) > self.config.max_parallel_tasks:
                    raise ValueError("Too many parallel tasks")
            
            tasks = []
            for node_id in node_ids:
                task = asyncio.create_task(self.execute_node(node_id, input_data))
                tasks.append(task)
                self._parallel_tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Clean up completed tasks
            self._parallel_tasks = [
                task for task in self._parallel_tasks
                if not task.done()
            ]
            
            # Combine results
            combined_results = {}
            for node_id, result in zip(node_ids, results):
                if isinstance(result, Exception):
                    logger.error(f"Node {node_id} failed: {result}")
                    continue
                combined_results[node_id] = result
            
            return combined_results
        except Exception as e:
            logger.error(f"Failed to execute parallel nodes: {e}")
            self._state.error = str(e)
            raise
    
    async def execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire workflow.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Workflow execution results
        """
        try:
            self._state = WorkflowState(status=WorkflowStatus.PROCESSING)
            self._state.data = input_data.copy()
            
            # Execute workflow implementation
            result = await self._execute_workflow_impl(input_data)
            
            # Update state
            self._state.status = WorkflowStatus.COMPLETED
            self._state.data.update(result)
            
            return result
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            self._state.status = WorkflowStatus.ERROR
            self._state.error = str(e)
            raise
    
    @abstractmethod
    async def _execute_node_impl(self, node: Node, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node implementation.
        
        Args:
            node: Node to execute
            input_data: Input data for the node
            
        Returns:
            Node execution results
        """
        pass
    
    @abstractmethod
    async def _execute_workflow_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow implementation.
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Workflow execution results
        """
        pass
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get workflow information.
        
        Returns:
            Dictionary containing workflow information
        """
        return {
            'id': self.config.id,
            'type': self.config.type,
            'created': self.created,
            'modified': self.modified,
            'parameters': self.config.parameters,
            'metadata': self.config.metadata,
            'nodes': len(self._nodes),
            'edges': len(self._edges),
            'state': self._state.dict(),
            'storage': self._storage.get_storage_info() if self._storage else None
        }
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics.
        
        Returns:
            Dictionary containing workflow statistics
        """
        try:
            stats = {
                'nodes': len(self._nodes),
                'edges': len(self._edges),
                'completed_nodes': len(self._state.completed_nodes),
                'parallel_tasks': len(self._parallel_tasks),
                'last_modified': self.modified
            }
            
            if self._storage:
                stats['storage'] = self._storage.get_storage_stats()
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get workflow stats: {e}")
            return {} 