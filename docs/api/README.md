# NeuralFlow API Documentation

## Overview

The NeuralFlow API provides a comprehensive interface for building and managing AI workflows. This documentation covers all available endpoints, their usage, and examples.

## Documentation Structure

### Core Components
- [Context Management](core/context.md) - Managing workflow contexts and state
- [Graph Processing](core/graph.md) - Working with workflow graphs
- [Memory Management](core/memory.md) - Managing conversation and workflow memory
- [Response Handling](core/response.md) - Understanding API responses

### Security
- [Authentication](security/authentication.md) - API authentication and authorization
- [Rate Limiting](security/rate_limiting.md) - API rate limits and quotas
- [Security Best Practices](security/security.md) - Security guidelines and best practices

### Integration
- [LangChain Integration](integration/langchain.md) - Using NeuralFlow with LangChain
- [Vector Store Integration](integration/vector_store.md) - Working with vector stores
- [Web Search Integration](integration/web_search.md) - Integrating web search capabilities

## Quick Start

### Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### Basic Usage

```python
from neuralflow import NeuralFlow

# Initialize the client
client = NeuralFlow(api_key="your_api_key")

# Create a workflow
workflow = client.workflows.create(
    name="My Workflow",
    description="A sample workflow",
    nodes=[
        {
            "type": "llm",
            "config": {
                "model": "gpt-4",
                "temperature": 0.7
            }
        }
    ]
)

# Execute the workflow
result = workflow.execute(input_data={"prompt": "Generate a story"})
```

## Support

For API support:
- Email: support@neuralflow.com
- Documentation: https://docs.neuralflow.com/api
- GitHub Issues: https://github.com/yavuztopsever/neuralflow/issues 