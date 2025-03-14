# API Module

This module contains all API-related code for the LangGraph framework.

## Directory Structure

- `routes/`: Contains all API route definitions and handlers
- `middleware/`: Contains API middleware components (auth, cors, rate limiting)
- `schemas/`: Contains request/response schemas and data models

## Usage

```python
from langgraph.api.routes import api_router
from langgraph.api.middleware import auth_middleware
from langgraph.api.schemas import GraphRequest, GraphResponse

# Example route handler
@api_router.post("/graph")
async def create_graph(request: GraphRequest) -> GraphResponse:
    # Implementation
    pass
```

## Key Components

1. **Routes**: Define API endpoints and their handlers
2. **Middleware**: Handle cross-cutting concerns like authentication and rate limiting
3. **Schemas**: Define data structures for API requests and responses

## Best Practices

- Keep route handlers thin, moving business logic to services
- Use Pydantic models for request/response validation
- Implement proper error handling and status codes
- Document all endpoints using OpenAPI/Swagger 