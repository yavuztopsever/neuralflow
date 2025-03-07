# NeuralFlow Testing Guide

This directory contains all tests for the NeuralFlow framework. This guide provides comprehensive information about our testing strategy, organization, and best practices.

## Directory Structure

```
tests/
├── unit/                    # Unit tests
│   ├── api/                # API endpoint tests
│   ├── core/               # Core functionality tests
│   │   ├── workflow/      # Workflow engine tests
│   │   ├── state/         # State management tests
│   │   ├── graph/         # Graph processing tests
│   │   ├── context/       # Context management tests
│   │   ├── events/        # Event system tests
│   │   ├── tools/         # Tool management tests
│   │   └── services/      # Service layer tests
│   ├── data/              # Data processing tests
│   ├── graph_store/       # Graph storage tests
│   ├── memory/            # Memory management tests
│   ├── models/            # Data model tests
│   ├── services/          # Business service tests
│   ├── storage/           # Storage system tests
│   ├── ui/                # UI component tests
│   ├── utils/             # Utility function tests
│   └── vector_store/      # Vector storage tests
├── integration/           # Integration tests
│   ├── api/              # API integration tests
│   ├── workflow/         # Workflow integration tests
│   ├── services/         # Service integration tests
│   └── storage/          # Storage integration tests
├── e2e/                  # End-to-end tests
│   ├── workflows/        # Complete workflow tests
│   └── ui/              # UI end-to-end tests
├── performance/          # Performance tests
│   ├── load/            # Load testing
│   └── stress/          # Stress testing
├── acceptance/          # Acceptance tests
├── fixtures/            # Test fixtures
└── conftest.py         # Pytest configuration and shared fixtures
```

## Testing Strategy

### 1. Test Types

#### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Fast execution
- High coverage
- Located in `tests/unit/`

#### Integration Tests
- Test component interactions
- Use real dependencies where appropriate
- Test system boundaries
- Located in `tests/integration/`

#### End-to-End Tests
- Test complete workflows
- Use real system components
- Test user scenarios
- Located in `tests/e2e/`

#### Performance Tests
- Test system performance
- Load and stress testing
- Resource usage monitoring
- Located in `tests/performance/`

#### Acceptance Tests
- Test business requirements
- User story validation
- Located in `tests/acceptance/`

### 2. Test Organization

- Mirror the src directory structure
- Use descriptive test names
- Group related tests using test classes
- Use fixtures for common setup
- Follow AAA pattern (Arrange, Act, Assert)

### 3. Best Practices

- Use pytest fixtures for setup and teardown
- Mock external dependencies
- Test edge cases and error conditions
- Maintain test isolation
- Use parameterized tests where appropriate
- Write clear test descriptions
- Keep tests focused and atomic

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/core/workflow/test_workflow_manager.py

# Run tests with coverage
pytest --cov=src tests/

# Run tests in parallel
pytest -n auto

# Run specific test types
pytest tests/unit/  # Unit tests only
pytest tests/integration/  # Integration tests only
pytest tests/e2e/  # End-to-end tests only
pytest tests/performance/  # Performance tests only

# Run with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run integration tests
```

## Test Dependencies

- pytest: Testing framework
- pytest-cov: Coverage reporting
- pytest-mock: Mocking utilities
- pytest-asyncio: Async test support
- pytest-xdist: Parallel test execution
- pytest-benchmark: Performance testing
- pytest-selenium: UI testing
- pytest-aiohttp: API testing

## Example Tests

### Unit Test Example
```python
import pytest
from src.core.workflow import WorkflowManager

def test_workflow_creation():
    # Arrange
    config = WorkflowConfig()
    
    # Act
    manager = WorkflowManager(config)
    
    # Assert
    assert manager.config == config
    assert manager.state_manager is not None

@pytest.mark.asyncio
async def test_workflow_execution():
    # Arrange
    config = WorkflowConfig()
    manager = WorkflowManager(config)
    
    # Act
    result = await manager.process_user_input("test query")
    
    # Assert
    assert result.final_response is not None
    assert result.error is None
```

### Integration Test Example
```python
@pytest.mark.integration
async def test_workflow_with_memory():
    # Arrange
    config = WorkflowConfig()
    manager = WorkflowManager(config)
    
    # Act
    result1 = await manager.process_user_input("First query")
    result2 = await manager.process_user_input("Follow-up query")
    
    # Assert
    assert result2.context_processed
    assert len(result2.memory) > 0
```

### E2E Test Example
```python
@pytest.mark.e2e
async def test_complete_workflow():
    # Arrange
    client = TestClient(app)
    
    # Act
    response = await client.post("/api/workflow", json={
        "query": "Test query",
        "options": {
            "include_sources": True
        }
    })
    
    # Assert
    assert response.status_code == 200
    assert "result" in response.json()
    assert "sources" in response.json()
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def workflow_config():
    return WorkflowConfig(
        max_context_items=5,
        max_parallel_tasks=3
    )

@pytest.fixture
def workflow_manager(workflow_config):
    return WorkflowManager(workflow_config)

@pytest.fixture
async def test_client():
    app = create_test_app()
    async with TestClient(app) as client:
        yield client
```

## Continuous Integration

Tests are automatically run in CI/CD pipeline:
- Unit tests on every commit
- Integration tests on pull requests
- E2E tests on merge to develop
- Performance tests on release candidates

## Coverage Requirements

- Minimum 80% code coverage
- 100% coverage for critical paths
- Coverage reports generated on CI
- Coverage thresholds enforced

## Performance Testing

Performance tests ensure:
- Response time within SLA
- Resource usage within limits
- Scalability under load
- Stability under stress

## Security Testing

Security tests verify:
- Input validation
- Authentication
- Authorization
- Data protection
- API security

## Maintenance

Regular test maintenance includes:
- Updating test data
- Refactoring for clarity
- Removing obsolete tests
- Adding new test cases
- Updating dependencies