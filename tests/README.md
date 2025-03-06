# Testing

This directory contains all tests for the LangGraph framework.

## Directory Structure

```
tests/
├── api/           # API tests
├── core/          # Core functionality tests
├── infrastructure/# Infrastructure tests
├── services/      # Service tests
├── utils/         # Utility tests
└── ui/            # UI tests
```

## Testing Guidelines

1. **Test Organization**
   - Mirror the src directory structure
   - Use descriptive test names
   - Group related tests using test classes
   - Use fixtures for common setup

2. **Test Types**
   - Unit tests: Test individual components
   - Integration tests: Test component interactions
   - End-to-End tests: Test complete workflows

3. **Best Practices**
   - Use pytest fixtures for setup and teardown
   - Mock external dependencies
   - Test edge cases and error conditions
   - Maintain test isolation
   - Use parameterized tests where appropriate

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_graph.py

# Run tests with coverage
pytest --cov=src tests/

# Run tests in parallel
pytest -n auto
```

## Test Dependencies

- pytest: Testing framework
- pytest-cov: Coverage reporting
- pytest-mock: Mocking utilities
- pytest-asyncio: Async test support
- pytest-xdist: Parallel test execution

## Example Test

```python
import pytest
from langgraph.core.graph import Graph, Node

def test_graph_creation():
    # Arrange
    graph = Graph()
    
    # Act
    node = Node("test")
    graph.add_node(node)
    
    # Assert
    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "test"

@pytest.fixture
def sample_graph():
    graph = Graph()
    graph.add_node(Node("input"))
    graph.add_node(Node("process"))
    return graph

def test_graph_operations(sample_graph):
    assert len(sample_graph.nodes) == 2
    assert any(node.id == "input" for node in sample_graph.nodes)