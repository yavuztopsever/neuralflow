"""
Graph node definitions for the workflow system.
"""

from typing import Dict, Any, Optional

class Node:
    """Represents a node in the workflow graph."""
    
    def __init__(
        self,
        id: str,
        type: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a node.
        
        Args:
            id: Unique identifier for the node
            type: Type of the node (e.g., 'llm', 'cache', 'input', 'output')
            config: Configuration parameters for the node
            metadata: Optional metadata about the node
        """
        self.id = id
        self.type = type
        self.config = config
        self.metadata = metadata or {}
        
    def __str__(self) -> str:
        """String representation of the node."""
        return f"Node(id='{self.id}', type='{self.type}')"
        
    def __repr__(self) -> str:
        """Detailed string representation of the node."""
        return f"Node(id='{self.id}', type='{self.type}', config={self.config}, metadata={self.metadata})"

class TaskNode(Node):
    """Node representing a task in the workflow."""
    pass

class DecisionNode(Node):
    """Node representing a decision point in the workflow."""
    pass

class InputNode(Node):
    """Node representing input data in the workflow."""
    pass

class OutputNode(Node):
    """Node representing output data in the workflow."""
    pass

__all__ = ['Node', 'TaskNode', 'DecisionNode', 'InputNode', 'OutputNode'] 