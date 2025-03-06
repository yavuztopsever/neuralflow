# Graph API Documentation

This document provides detailed information about the Graph API endpoints for managing workflows and graph-based operations.

## Endpoints

### Create Workflow
```http
POST /v1/workflows
```

Creates a new workflow graph.

**Request Body:**
```json
{
  "name": "string",
  "description": "string",
  "nodes": [
    {
      "id": "string",
      "type": "string",
      "config": {}
    }
  ],
  "edges": [
    {
      "from": "string",
      "to": "string",
      "condition": "string"
    }
  ]
}
```

**Response:**
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "nodes": [],
  "edges": [],
  "created_at": "string",
  "updated_at": "string"
}
```

### Get Workflow
```http
GET /v1/workflows/{workflow_id}
```

Retrieves a specific workflow by ID.

**Response:**
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "nodes": [],
  "edges": [],
  "created_at": "string",
  "updated_at": "string"
}
```

### Update Workflow
```http
PUT /v1/workflows/{workflow_id}
```

Updates an existing workflow.

**Request Body:**
```json
{
  "name": "string",
  "description": "string",
  "nodes": [],
  "edges": []
}
```

### Delete Workflow
```http
DELETE /v1/workflows/{workflow_id}
```

Deletes a specific workflow.

### List Workflows
```http
GET /v1/workflows
```

Lists all workflows with pagination support.

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `sort`: Sort field (default: created_at)
- `order`: Sort order (asc/desc)

### Execute Workflow
```http
POST /v1/workflows/{workflow_id}/execute
```

Executes a workflow with the provided input.

**Request Body:**
```json
{
  "input": {},
  "options": {
    "timeout": "number",
    "max_steps": "number"
  }
}
```

**Response:**
```json
{
  "execution_id": "string",
  "status": "string",
  "result": {},
  "steps": [],
  "started_at": "string",
  "completed_at": "string"
}
```

### Get Execution Status
```http
GET /v1/executions/{execution_id}
```

Retrieves the status of a workflow execution.

**Response:**
```json
{
  "execution_id": "string",
  "workflow_id": "string",
  "status": "string",
  "result": {},
  "steps": [],
  "started_at": "string",
  "completed_at": "string"
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameters",
    "details": {}
  }
}
```

### 401 Unauthorized
```json
{
  "error": {
    "code": "unauthorized",
    "message": "Authentication required"
  }
}
```

### 403 Forbidden
```json
{
  "error": {
    "code": "forbidden",
    "message": "Insufficient permissions"
  }
}
```

### 404 Not Found
```json
{
  "error": {
    "code": "not_found",
    "message": "Resource not found"
  }
}
```

### 429 Too Many Requests
```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded"
  }
}
```

## Rate Limiting

The Graph API implements rate limiting based on your subscription tier:

- Free Tier: 100 requests per minute
- Pro Tier: 1000 requests per minute
- Enterprise Tier: Custom limits

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623456789
```

## Authentication

All API requests require authentication using an API key:

```http
Authorization: Bearer your-api-key
```

## Webhooks

You can configure webhooks to receive notifications about workflow events:

```json
{
  "url": "https://your-domain.com/webhook",
  "events": [
    "workflow.created",
    "workflow.updated",
    "workflow.deleted",
    "workflow.executed",
    "workflow.completed",
    "workflow.failed"
  ],
  "secret": "your-webhook-secret"
}
```

## SDK Examples

### Python
```python
from langgraph.api import GraphAPI

api = GraphAPI(api_key="your-api-key")

# Create workflow
workflow = api.create_workflow({
    "name": "My Workflow",
    "description": "A sample workflow",
    "nodes": [...],
    "edges": [...]
})

# Execute workflow
execution = api.execute_workflow(workflow.id, {
    "input": {...},
    "options": {...}
})
```

### JavaScript
```javascript
const { GraphAPI } = require('langgraph');

const api = new GraphAPI('your-api-key');

// Create workflow
const workflow = await api.createWorkflow({
    name: 'My Workflow',
    description: 'A sample workflow',
    nodes: [...],
    edges: [...]
});

// Execute workflow
const execution = await api.executeWorkflow(workflow.id, {
    input: {...},
    options: {...}
});
```
