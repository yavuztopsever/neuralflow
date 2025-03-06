"""
Workflow management utilities for the LangGraph application.
This module provides functionality for defining and executing workflows.
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
from core.graph import GraphManager, GraphNode

logger = logging.getLogger(__name__)

class WorkflowStep:
    """Represents a step in a workflow."""
    
    def __init__(self, step_id: str,
                 handler: Callable,
                 inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a workflow step.
        
        Args:
            step_id: Unique identifier for the step
            handler: Function to handle step operations
            inputs: List of input step IDs
            outputs: List of output step IDs
            metadata: Optional metadata for the step
        """
        self.id = step_id
        self.handler = handler
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.metadata = metadata or {}
        self.created = datetime.now().isoformat()
        self.modified = self.created
    
    def add_input(self, step_id: str) -> None:
        """Add an input step.
        
        Args:
            step_id: ID of the input step
        """
        if step_id not in self.inputs:
            self.inputs.append(step_id)
            self.modified = datetime.now().isoformat()
    
    def add_output(self, step_id: str) -> None:
        """Add an output step.
        
        Args:
            step_id: ID of the output step
        """
        if step_id not in self.outputs:
            self.outputs.append(step_id)
            self.modified = datetime.now().isoformat()
    
    def remove_input(self, step_id: str) -> None:
        """Remove an input step.
        
        Args:
            step_id: ID of the input step to remove
        """
        if step_id in self.inputs:
            self.inputs.remove(step_id)
            self.modified = datetime.now().isoformat()
    
    def remove_output(self, step_id: str) -> None:
        """Remove an output step.
        
        Args:
            step_id: ID of the output step to remove
        """
        if step_id in self.outputs:
            self.outputs.remove(step_id)
            self.modified = datetime.now().isoformat()

class WorkflowManager:
    """Manages workflow definitions and execution."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the workflow manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.log_manager = LogManager(self.config)
        self.text_processor = TextProcessor()
        self.graph_manager = GraphManager(self.config)
        self._workflows = {}
        self._initialize_workflows()
    
    def _initialize_workflows(self):
        """Initialize default workflows."""
        try:
            # Create default workflow
            self.create_workflow('default', [
                WorkflowStep('input', self._handle_input),
                WorkflowStep('process', self._handle_process),
                WorkflowStep('output', self._handle_output)
            ])
            
            logger.info("Initialized default workflows")
        except Exception as e:
            logger.error(f"Failed to initialize workflows: {e}")
            raise
    
    def create_workflow(self, workflow_id: str,
                       steps: List[WorkflowStep]) -> None:
        """Create a new workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow
            steps: List of workflow steps
            
        Raises:
            ValueError: If workflow_id already exists
        """
        try:
            if workflow_id in self._workflows:
                raise ValueError(f"Workflow {workflow_id} already exists")
            
            # Create workflow
            workflow = {
                'steps': steps,
                'created': datetime.now().isoformat(),
                'modified': datetime.now().isoformat()
            }
            
            # Add to workflows
            self._workflows[workflow_id] = workflow
            logger.info(f"Created workflow {workflow_id}")
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_id}: {e}")
            raise
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by ID.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow definition or None if not found
        """
        return self._workflows.get(workflow_id)
    
    def update_workflow(self, workflow_id: str,
                       steps: Optional[List[WorkflowStep]] = None) -> bool:
        """Update a workflow.
        
        Args:
            workflow_id: Workflow ID
            steps: Optional new list of steps
            
        Returns:
            True if workflow was updated, False otherwise
        """
        try:
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                return False
            
            if steps is not None:
                workflow['steps'] = steps
            workflow['modified'] = datetime.now().isoformat()
            
            logger.info(f"Updated workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            return False
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if workflow was deleted, False otherwise
        """
        try:
            if workflow_id not in self._workflows:
                return False
            
            del self._workflows[workflow_id]
            logger.info(f"Deleted workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False
    
    def execute_workflow(self, workflow_id: str,
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow_id: Workflow ID
            context: Execution context
            
        Returns:
            Workflow execution results
            
        Raises:
            ValueError: If workflow not found
        """
        try:
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create graph from workflow
            self._create_graph_from_workflow(workflow_id)
            
            # Execute graph
            results = self.graph_manager.execute_graph(context)
            
            return results
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            raise
    
    def _create_graph_from_workflow(self, workflow_id: str) -> None:
        """Create a graph from a workflow definition.
        
        Args:
            workflow_id: Workflow ID
            
        Raises:
            ValueError: If workflow not found
        """
        try:
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Clear existing graph
            self.graph_manager = GraphManager(self.config)
            
            # Add nodes
            for step in workflow['steps']:
                self.graph_manager.add_node(
                    step.id,
                    step.handler,
                    step.metadata
                )
            
            # Connect nodes
            for step in workflow['steps']:
                for input_id in step.inputs:
                    self.graph_manager.connect_nodes(input_id, step.id)
                for output_id in step.outputs:
                    self.graph_manager.connect_nodes(step.id, output_id)
            
            logger.info(f"Created graph for workflow {workflow_id}")
        except Exception as e:
            logger.error(f"Failed to create graph for workflow {workflow_id}: {e}")
            raise
    
    def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get statistics about a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Dictionary containing workflow statistics
        """
        try:
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                return {}
            
            steps = workflow['steps']
            return {
                'steps': len(steps),
                'connections': sum(len(step.outputs) for step in steps),
                'input_steps': len([step for step in steps if not step.inputs]),
                'output_steps': len([step for step in steps if not step.outputs]),
                'step_types': {
                    step.id: type(step.handler).__name__
                    for step in steps
                }
            }
        except Exception as e:
            logger.error(f"Failed to get workflow stats: {e}")
            return {}
    
    def _handle_input(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle input step operations.
        
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
        """Handle process step operations.
        
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
        """Handle output step operations.
        
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