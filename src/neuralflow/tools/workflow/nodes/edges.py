"""
Graph edge definitions for the workflow system.
"""

from typing import Dict, Any, Optional

class Edge:
    """Represents an edge in the workflow graph."""
    
    def __init__(
        self,
        id: str,
        source: str,
        target: str,
        data_key: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize an edge.
        
        Args:
            id: Unique identifier for the edge
            source: ID of the source node
            target: ID of the target node
            data_key: Key for data transfer between nodes
            config: Optional configuration for the edge
        """
        self.id = id
        self.source = source
        self.target = target
        self.data_key = data_key
        self.config = config or {}
        
    def __str__(self) -> str:
        """String representation of the edge."""
        return f"Edge(id='{self.id}', source='{self.source}', target='{self.target}')"
        
    def __repr__(self) -> str:
        """Detailed string representation of the edge."""
        return f"Edge(id='{self.id}', source='{self.source}', target='{self.target}', data_key='{self.data_key}', config={self.config})"

class DataEdge(Edge):
    """Edge representing data flow between nodes."""
    pass

class ControlEdge(Edge):
    """Edge representing control flow between nodes."""
    pass

class ConditionalEdge(Edge):
    """Edge representing conditional branching."""
    pass

__all__ = ['Edge', 'DataEdge', 'ControlEdge', 'ConditionalEdge'] 