# NeuralFlow Development Guide

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Docker (optional)
- Virtual environment (recommended)

### Development Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yavuztopsever/neuralflow.git
cd neuralflow
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Project Structure

```
neuralflow/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   ├── config/            # Configuration management
│   ├── core/              # Core functionality
│   │   ├── workflow/      # Workflow engine
│   │   ├── state/         # State management
│   │   ├── graph/         # Graph processing
│   │   ├── context/       # Context management
│   │   ├── events/        # Event system
│   │   ├── tools/         # Tool management
│   │   └── services/      # Service layer
│   ├── data/              # Data processing
│   ├── graph_store/       # Graph storage
│   ├── memory/            # Memory management
│   ├── models/            # Data models
│   ├── services/          # Business services
│   ├── storage/           # Storage system
│   ├── ui/                # User interface
│   ├── utils/             # Utility functions
│   └── vector_store/      # Vector storage
├── tests/                 # Test files
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── e2e/             # End-to-end tests
│   ├── performance/     # Performance tests
│   ├── acceptance/      # Acceptance tests
│   └── fixtures/        # Test fixtures
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

## Development Workflow

### 1. Branch Management

- `main`: Production-ready code
- `develop`: Development branch
- Feature branches: `feature/feature-name`
- Bug fix branches: `fix/bug-name`
- Release branches: `release/version`

### 2. Code Style

We follow PEP 8 guidelines with some modifications:

```python
# Good
def process_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process input data and return processed results.
    
    Args:
        data: Input data dictionary
        
    Returns:
        List of processed data dictionaries
    """
    results = []
    for item in data:
        processed = transform(item)
        results.append(processed)
    return results

# Bad
def process_data(data):
    results=[]
    for item in data:processed=transform(item);results.append(processed)
    return results
```

### 3. Testing

#### Test Types
- Unit tests: Test individual components
- Integration tests: Test component interactions
- End-to-end tests: Test complete workflows
- Performance tests: Test system performance
- Acceptance tests: Test business requirements

#### Running Tests
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
```

#### Writing Tests

```python
import pytest
from neuralflow.core.workflow import WorkflowManager

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

#### Test Best Practices
- Use pytest fixtures for setup and teardown
- Mock external dependencies
- Test edge cases and error conditions
- Maintain test isolation
- Use parameterized tests where appropriate
- Write clear test descriptions
- Keep tests focused and atomic

### 4. Documentation

#### Code Documentation
- Use docstrings for all public functions and classes
- Follow Google style docstrings
- Include type hints

```python
def process_workflow(
    workflow_id: str,
    input_data: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a workflow with given input data.
    
    Args:
        workflow_id: Unique identifier for the workflow
        input_data: Input data for workflow processing
        options: Optional configuration parameters
        
    Returns:
        Processed workflow results
        
    Raises:
        WorkflowNotFoundError: If workflow doesn't exist
        ValidationError: If input data is invalid
    """
    pass
```

#### API Documentation
- Update API documentation for new endpoints
- Include request/response examples
- Document error cases

### 5. Code Review Process

1. Create pull request
2. Ensure CI checks pass
3. Address review comments
4. Merge after approval

### 6. Performance Guidelines

1. **Memory Management**
   - Use LangChain's memory system effectively
   - Implement proper cleanup
   - Monitor memory usage

2. **API Design**
   - Implement pagination
   - Use caching where appropriate
   - Optimize response size

3. **Resource Management**
   - Implement proper cleanup
   - Monitor memory usage
   - Handle connection pooling

## Debugging

### 1. Logging

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data):
    logger.debug("Processing data: %s", data)
    try:
        result = transform(data)
        logger.info("Data processed successfully")
        return result
    except Exception as e:
        logger.error("Error processing data: %s", str(e))
        raise
```

### 2. Debug Tools

- Use `pdb` for debugging
- Implement proper error handling
- Use logging effectively

## Deployment

### 1. Local Development

```bash
# Start development server
python src/main.py --dev

# Run with Docker
docker-compose up
```

### 2. Production Deployment

1. Build Docker image:
```bash
docker build -t neuralflow .
```

2. Deploy to production:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Contributing

Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details. 