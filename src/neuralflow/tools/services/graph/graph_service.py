"""
Graph service for the LangGraph project.
This service provides graph management capabilities for workflows.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

class GraphService:
    """Service for managing graph operations in the LangGraph system."""
    
    def __init__(self):
        """Initialize the graph service."""
        self.graphs: Dict[str, StateGraph] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
    
    def create_graph(self, name: str, state_type: type) -> StateGraph:
        """
        Create a new graph.
        
        Args:
            name: Graph name
            state_type: Type of state to use
            
        Returns:
            StateGraph: Created graph
        """
        graph = StateGraph(state_type)
        self.graphs[name] = graph
        return graph
    
    def get_graph(self, name: str) -> Optional[StateGraph]:
        """
        Get a graph by name.
        
        Args:
            name: Graph name
            
        Returns:
            Optional[StateGraph]: Graph if found, None otherwise
        """
        return self.graphs.get(name)
    
    def add_node(self, graph_name: str, node_name: str, node_func: callable) -> None:
        """
        Add a node to a graph.
        
        Args:
            graph_name: Graph name
            node_name: Node name
            node_func: Node function
        """
        graph = self.get_graph(graph_name)
        if graph:
            graph.add_node(node_name, node_func)
    
    def add_edge(self, graph_name: str, from_node: str, to_node: str) -> None:
        """
        Add an edge to a graph.
        
        Args:
            graph_name: Graph name
            from_node: Source node
            to_node: Target node
        """
        graph = self.get_graph(graph_name)
        if graph:
            graph.add_edge(from_node, to_node)
    
    def save_checkpoint(self, graph_name: str, checkpoint_id: str, state: Dict[str, Any]) -> None:
        """
        Save a graph state checkpoint.
        
        Args:
            graph_name: Graph name
            checkpoint_id: Checkpoint ID
            state: State to save
        """
        if graph_name not in self.checkpoints:
            self.checkpoints[graph_name] = {}
        self.checkpoints[graph_name][checkpoint_id] = state
    
    def load_checkpoint(self, graph_name: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a graph state checkpoint.
        
        Args:
            graph_name: Graph name
            checkpoint_id: Checkpoint ID
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint state if found, None otherwise
        """
        return self.checkpoints.get(graph_name, {}).get(checkpoint_id)
    
    def clear_checkpoints(self, graph_name: Optional[str] = None) -> None:
        """
        Clear graph checkpoints.
        
        Args:
            graph_name: Optional graph name to clear checkpoints for
        """
        if graph_name:
            self.checkpoints[graph_name] = {}
        else:
            self.checkpoints = {}
    
    def reset(self) -> None:
        """Reset the graph service to its initial state."""
        self.graphs = {}
        self.checkpoints = {}

__all__ = ['GraphService'] 