# NeuralFlow Workflow System

The NeuralFlow Workflow System provides a flexible and powerful framework for building and executing AI workflows. It supports both sequential and parallel execution patterns, with built-in persistence and monitoring capabilities.

## Components

### Core Components

1. **WorkflowNode**: Base class for workflow nodes
   - Handles input/output management
   - Provides execution and validation interfaces
   - Tracks node status and timestamps

2. **WorkflowEdge**: Represents connections between nodes
   - Supports conditional execution
   - Handles data flow between nodes
   - Enables workflow graph construction

3. **Workflow**: Abstract base class for workflows
   - Manages node and edge collections
   - Provides workflow lifecycle management
   - Supports workflow validation

### Implementations

1. **SequentialWorkflow**
   - Executes nodes in a sequential order
   - Ensures proper dependency resolution
   - Maintains execution order

2. **ParallelWorkflow**
   - Executes independent nodes in parallel
   - Optimizes execution time
   - Handles concurrent execution safely

### Management

1. **WorkflowManager**
   - Manages workflow lifecycle
   - Handles workflow persistence
   - Provides workflow CRUD operations

2. **WorkflowExecutionService**
   - Executes workflows
   - Monitors execution status
   - Manages execution history
   - Supports workflow cancellation

## Usage

### Creating a Workflow

```python
from neuralflow.core.workflow.implementations import SequentialWorkflow
from neuralflow.core.models.management.workflow_manager import WorkflowManager
from neuralflow.core.services.workflow_executor import WorkflowExecutionService

# Create workflow manager
workflow_manager = WorkflowManager()

# Create workflow
workflow = await workflow_manager.create_workflow("My Workflow", "sequential")

# Add nodes
node1 = MyCustomNode("node1", "Node 1")
node2 = MyCustomNode("node2", "Node 2")
await workflow_manager.add_node(workflow.workflow_id, node1)
await workflow_manager.add_node(workflow.workflow_id, node2)

# Add edges
edge = WorkflowEdge("node1", "node2")
await workflow_manager.add_edge(workflow.workflow_id, edge)

# Create executor
executor = WorkflowExecutionService(workflow_manager)

# Execute workflow
results = await executor.execute_workflow(workflow.workflow_id)
```

### Creating Custom Nodes

```python
from neuralflow.core.workflow.base import WorkflowNode

class MyCustomNode(WorkflowNode):
    async def execute(self) -> dict:
        # Implement node logic
        return {"result": "Node executed"}
    
    async def validate(self) -> bool:
        # Implement validation logic
        return True
```

### Monitoring Workflow Status

```python
# Get workflow status
status = await executor.get_execution_status(workflow.workflow_id)

# Check node status
for node in status["nodes"]:
    print(f"Node {node['name']}: {node['status']}")

# View execution history
for entry in status["execution_history"]:
    print(f"Execution {entry['execution_id']}: {entry['status']}")
```

## Features

1. **Flexible Execution Patterns**
   - Sequential execution
   - Parallel execution
   - Mixed execution patterns

2. **Persistence**
   - Automatic workflow saving
   - Workflow state restoration
   - Execution history tracking

3. **Monitoring**
   - Real-time status updates
   - Execution history
   - Node-level monitoring

4. **Error Handling**
   - Graceful error recovery
   - Workflow cancellation
   - Status tracking

5. **Extensibility**
   - Custom node implementations
   - Custom edge conditions
   - Custom workflow types

## Best Practices

1. **Node Implementation**
   - Keep nodes focused and single-purpose
   - Implement proper validation
   - Handle errors gracefully

2. **Workflow Design**
   - Use appropriate workflow type
   - Minimize dependencies
   - Consider parallelization opportunities

3. **Error Handling**
   - Implement proper validation
   - Handle node failures
   - Monitor execution status

4. **Performance**
   - Use parallel workflows when possible
   - Optimize node execution
   - Clean up old execution history

## Testing

The workflow system includes comprehensive tests:

```bash
pytest tests/test_workflow.py
```

Tests cover:
- Sequential workflow execution
- Parallel workflow execution
- Workflow persistence
- Workflow cancellation
- Execution history
- Error handling
- Edge cases

## Contributing

When contributing to the workflow system:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Consider performance implications
5. Handle edge cases

## License

This component is part of the NeuralFlow framework and is licensed under the MIT License. 