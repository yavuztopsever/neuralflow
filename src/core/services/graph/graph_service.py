"""
Graph management utilities for the LangGraph application.
This module provides functionality for managing the workflow graph and its operations.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime
from utils.common.text import TextProcessor
from utils.error.handlers import ErrorHandler
from config.manager import ConfigManager
from utils.logging.manager import LogManager

logger = logging.getLogger(__name__)

class GraphNode:
    """Represents a node in the workflow graph."""
    
    def __init__(self, node_id: str, 
                 handler: Callable,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a graph node.
        
        Args:
            node_id: Unique identifier for the node
            handler: Function to handle node operations
            metadata: Optional metadata for the node
        """
        self.id = node_id
        self.handler = handler
        self.metadata = metadata or {}
        self.inputs = []
        self.outputs = []
        self.created = datetime.now().isoformat()
        self.modified = self.created
    
    def add_input(self, node: 'GraphNode') -> None:
        """Add an input node.
        
        Args:
            node: Input node to add
        """
        if node not in self.inputs:
            self.inputs.append(node)
            self.modified = datetime.now().isoformat()
    
    def add_output(self, node: 'GraphNode') -> None:
        """Add an output node.
        
        Args:
            node: Output node to add
        """
        if node not in self.outputs:
            self.outputs.append(node)
            self.modified = datetime.now().isoformat()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node's handler.
        
        Args:
            context: Execution context
            
        Returns:
            Execution results
        """
        try:
            return self.handler(context)
        except Exception as e:
            logger.error(f"Failed to execute node {self.id}: {e}")
            raise

class GraphManager:
    """Manages the workflow graph and its operations."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the graph manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self._nodes = {}
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize the workflow graph."""
        try:
            # Create default nodes
            self.add_node('input', self._handle_input)
            self.add_node('process', self._handle_process)
            self.add_node('output', self._handle_output)
            
            # Connect nodes
            self.connect_nodes('input', 'process')
            self.connect_nodes('process', 'output')
            
            logger.info("Initialized workflow graph")
        except Exception as e:
            logger.error(f"Failed to initialize graph: {e}")
            raise
    
    def add_node(self, node_id: str, 
                handler: Callable,
                metadata: Optional[Dict[str, Any]] = None) -> GraphNode:
        """Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            handler: Function to handle node operations
            metadata: Optional metadata for the node
            
        Returns:
            Created node
            
        Raises:
            ValueError: If node_id already exists
        """
        try:
            if node_id in self._nodes:
                raise ValueError(f"Node {node_id} already exists")
            
            node = GraphNode(node_id, handler, metadata)
            self._nodes[node_id] = node
            logger.info(f"Added node {node_id}")
            return node
        except Exception as e:
            logger.error(f"Failed to add node {node_id}: {e}")
            raise
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node instance or None if not found
        """
        return self._nodes.get(node_id)
    
    def connect_nodes(self, from_id: str, to_id: str) -> None:
        """Connect two nodes in the graph.
        
        Args:
            from_id: ID of the source node
            to_id: ID of the target node
            
        Raises:
            ValueError: If either node doesn't exist
        """
        try:
            from_node = self.get_node(from_id)
            to_node = self.get_node(to_id)
            
            if not from_node or not to_node:
                raise ValueError(f"Node {from_id} or {to_id} not found")
            
            from_node.add_output(to_node)
            to_node.add_input(from_node)
            logger.info(f"Connected nodes {from_id} -> {to_id}")
        except Exception as e:
            logger.error(f"Failed to connect nodes {from_id} -> {to_id}: {e}")
            raise
    
    def disconnect_nodes(self, from_id: str, to_id: str) -> None:
        """Disconnect two nodes in the graph.
        
        Args:
            from_id: ID of the source node
            to_id: ID of the target node
            
        Raises:
            ValueError: If either node doesn't exist
        """
        try:
            from_node = self.get_node(from_id)
            to_node = self.get_node(to_id)
            
            if not from_node or not to_node:
                raise ValueError(f"Node {from_id} or {to_id} not found")
            
            if to_node in from_node.outputs:
                from_node.outputs.remove(to_node)
            if from_node in to_node.inputs:
                to_node.inputs.remove(from_node)
            
            logger.info(f"Disconnected nodes {from_id} -> {to_id}")
        except Exception as e:
            logger.error(f"Failed to disconnect nodes {from_id} -> {to_id}: {e}")
            raise
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if the node was deleted, False otherwise
        """
        try:
            node = self.get_node(node_id)
            if not node:
                return False
            
            # Disconnect from all nodes
            for input_node in node.inputs:
                self.disconnect_nodes(input_node.id, node_id)
            for output_node in node.outputs:
                self.disconnect_nodes(node_id, output_node.id)
            
            # Remove node
            del self._nodes[node_id]
            logger.info(f"Deleted node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    def execute_graph(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow graph.
        
        Args:
            context: Initial execution context
            
        Returns:
            Final execution results
        """
        try:
            # Find start nodes (nodes with no inputs)
            start_nodes = [
                node for node in self._nodes.values()
                if not node.inputs
            ]
            
            if not start_nodes:
                raise ValueError("No start nodes found in graph")
            
            # Execute graph starting from each start node
            results = {}
            for node in start_nodes:
                results[node.id] = self._execute_node(node, context)
            
            return results
        except Exception as e:
            logger.error(f"Failed to execute graph: {e}")
            raise
    
    def _execute_node(self, node: GraphNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a node and its outputs.
        
        Args:
            node: Node to execute
            context: Execution context
            
        Returns:
            Node execution results
        """
        try:
            # Execute node
            result = node.execute(context)
            
            # Update context with result
            context[node.id] = result
            
            # Execute output nodes
            output_results = {}
            for output_node in node.outputs:
                output_results[output_node.id] = self._execute_node(output_node, context)
            
            return {
                'result': result,
                'outputs': output_results
            }
        except Exception as e:
            logger.error(f"Failed to execute node {node.id}: {e}")
            raise
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        try:
            return {
                'nodes': len(self._nodes),
                'edges': sum(len(node.outputs) for node in self._nodes.values()),
                'start_nodes': len([node for node in self._nodes.values() if not node.inputs]),
                'end_nodes': len([node for node in self._nodes.values() if not node.outputs]),
                'node_types': {
                    node_id: type(node.handler).__name__
                    for node_id, node in self._nodes.items()
                }
            }
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {}
    
    def _handle_input(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle input node operations.
        
        Args:
            context: Execution context
            
        Returns:
            Input processing results
        """
        try:
            # Process input data
            input_data = context.get('input', {})
            return {
                'processed_input': input_data,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to handle input: {e}")
            raise
    
    def _handle_process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle process node operations.
        
        Args:
            context: Execution context
            
        Returns:
            Processing results
        """
        try:
            # Get input data
            input_data = context.get('input', {}).get('processed_input', {})
            
            # Process data
            processed_data = {
                'processed': True,
                'timestamp': datetime.now().isoformat(),
                'data': input_data
            }
            
            return processed_data
        except Exception as e:
            logger.error(f"Failed to handle process: {e}")
            raise
    
    def _handle_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle output node operations.
        
        Args:
            context: Execution context
            
        Returns:
            Output processing results
        """
        try:
            # Get processed data
            process_data = context.get('process', {}).get('result', {})
            
            # Prepare output
            output_data = {
                'output': process_data,
                'timestamp': datetime.now().isoformat()
            }
            
            return output_data
        except Exception as e:
            logger.error(f"Failed to handle output: {e}")
            raise 