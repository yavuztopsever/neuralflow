# State Management

This module provides state management functionality for the LangGraph framework.

## Overview

The state management module handles the persistence and management of workflow states. It provides:

- State representation and validation
- State persistence to storage
- State management operations
- State history tracking
- State statistics

## Components

### WorkflowState

The `WorkflowState` class represents the state of a workflow execution:

```python
from langgraph.core.state import WorkflowState

# Create a new state
state = WorkflowState(
    workflow_id="workflow1",
    state_id="state1",
    context={"input": "test"},
    metadata={"user_id": "user123"}
)

# Update state
state.update(
    results={"output": "processed"},
    status="completed"
)

# Check state status
if state.is_completed():
    print("Workflow completed successfully")
```

### StateManager

The `StateManager` class manages workflow states:

```python
from langgraph.core.state import StateManager
from pathlib import Path

# Initialize manager
manager = StateManager(storage_dir=Path("storage/states"))

# Create state
state = manager.create_state(
    workflow_id="workflow1",
    context={"input": "test"}
)

# Update state
manager.update_state(
    state.state_id,
    results={"output": "processed"},
    status="completed"
)

# Get state
state = manager.get_state(state.state_id)

# Get workflow states
workflow_states = manager.get_workflow_states("workflow1")

# Get statistics
stats = manager.get_state_stats()
```

### StateValidator

The `StateValidator` class validates workflow states:

```python
from langgraph.core.state import StateValidator, WorkflowState

validator = StateValidator()
state = WorkflowState(...)

# Validate state
validator.validate_state(state)
```

### StatePersistence

The `StatePersistence` class handles state storage:

```python
from langgraph.core.state import StatePersistence
from pathlib import Path

persistence = StatePersistence(storage_dir=Path("storage/states"))

# Save state
persistence.save(state)

# Load state
state = persistence.load(state_id)

# Load all states
states = persistence.load_all()

# Delete state
persistence.delete(state_id)
```

## Features

1. **State Management**
   - Create, read, update, and delete states
   - Track state history
   - Manage workflow states
   - Generate state statistics

2. **Validation**
   - Required field validation
   - Status validation
   - Context validation
   - Metadata validation

3. **Persistence**
   - JSON file storage
   - Automatic file management
   - Error handling
   - Corrupted file recovery

4. **History Tracking**
   - State update history
   - Status changes
   - Result tracking
   - Error tracking

## Best Practices

1. **State Creation**
   - Always provide required fields
   - Use meaningful IDs
   - Include relevant context
   - Add helpful metadata

2. **State Updates**
   - Update status appropriately
   - Include meaningful results
   - Handle errors properly
   - Maintain history

3. **State Management**
   - Use state manager for operations
   - Validate states before saving
   - Clean up unused states
   - Monitor state statistics

4. **Error Handling**
   - Handle validation errors
   - Handle storage errors
   - Log errors appropriately
   - Provide meaningful error messages

## Examples

### Basic Usage

```python
from langgraph.core.state import StateManager, WorkflowState

# Initialize
manager = StateManager()

# Create workflow
state = manager.create_state(
    workflow_id="text_processing",
    context={"text": "Hello, World!"}
)

# Process workflow
try:
    # Process the text
    result = process_text(state.context["text"])
    
    # Update state
    manager.update_state(
        state.state_id,
        results={"processed_text": result},
        status="completed"
    )
except Exception as e:
    # Handle error
    manager.update_state(
        state.state_id,
        results={},
        status="failed",
        error=str(e)
    )

# Get workflow status
state = manager.get_state(state.state_id)
if state.is_completed():
    print(f"Processed text: {state.results['processed_text']}")
else:
    print(f"Error: {state.error}")
```

### Advanced Usage

```python
from langgraph.core.state import StateManager, WorkflowState
from pathlib import Path

# Initialize with custom storage
manager = StateManager(storage_dir=Path("custom/storage"))

# Create workflow with metadata
state = manager.create_state(
    workflow_id="image_processing",
    context={"image_path": "input.jpg"},
    metadata={
        "user_id": "user123",
        "priority": "high",
        "format": "jpg"
    }
)

# Process workflow with history
def process_image(state_id):
    state = manager.get_state(state_id)
    
    # Update status
    manager.update_state(state_id, {}, status="running")
    
    try:
        # Process image
        result = process_image_file(state.context["image_path"])
        
        # Update with success
        manager.update_state(
            state_id,
            results={"processed_image": result},
            status="completed"
        )
    except Exception as e:
        # Update with error
        manager.update_state(
            state_id,
            results={},
            status="failed",
            error=str(e)
        )

# Get workflow statistics
stats = manager.get_state_stats()
print(f"Total workflows: {stats['total_states']}")
print(f"Completed: {stats['status_counts']['completed']}")
print(f"Failed: {stats['status_counts']['failed']}")
```

## Testing

The module includes comprehensive tests:

```bash
# Run all state management tests
pytest tests/core/state/

# Run specific test file
pytest tests/core/state/test_workflow_state.py
pytest tests/core/state/test_state_manager.py
pytest tests/core/state/test_state_validator.py
pytest tests/core/state/test_state_persistence.py
```

## Contributing

When contributing to the state management module:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Handle errors appropriately
5. Maintain backward compatibility 