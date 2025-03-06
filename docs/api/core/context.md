# Context API Documentation

This document provides detailed information about the Context API endpoints for managing context in the system.

## Endpoints

### Create Context
```http
POST /v1/context
```

Creates a new context.

**Request Body:**
```json
{
  "name": "string",
  "description": "string",
  "type": "string",  // "conversation", "document", "session"
  "metadata": {
    "source": "string",
    "tags": ["string"]
  },
  "content": {}
}
```

**Response:**
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "type": "string",
  "metadata": {},
  "content": {},
  "created_at": "string",
  "updated_at": "string"
}
```

### Get Context
```http
GET /v1/context/{context_id}
```

Retrieves a specific context by ID.

### Update Context
```http
PUT /v1/context/{context_id}
```

Updates an existing context.

### Delete Context
```http
DELETE /v1/context/{context_id}
```

Deletes a specific context.

### Add Context Item
```http
POST /v1/context/{context_id}/items
```

Adds a new item to the context.

**Request Body:**
```json
{
  "type": "string",
  "content": {},
  "metadata": {
    "timestamp": "string",
    "source": "string",
    "tags": ["string"]
  }
}
```

### Get Context Item
```http
GET /v1/context/{context_id}/items/{item_id}
```

Retrieves a specific context item.

### Update Context Item
```http
PUT /v1/context/{context_id}/items/{item_id}
```

Updates an existing context item.

### Delete Context Item
```http
DELETE /v1/context/{context_id}/items/{item_id}
```

Deletes a specific context item.

### List Context Items
```http
GET /v1/context/{context_id}/items
```

Lists all items in a context with pagination support.

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `sort`: Sort field (default: timestamp)
- `order`: Sort order (asc/desc)

### Search Context
```http
POST /v1/context/search
```

Searches for contexts using semantic search.

**Request Body:**
```json
{
  "query": "string",
  "filters": {
    "type": "string",
    "tags": ["string"],
    "date_range": {
      "start": "string",
      "end": "string"
    }
  },
  "limit": "number"
}
```

### Summarize Context
```http
POST /v1/context/{context_id}/summarize
```

Generates a summary of the context.

**Request Body:**
```json
{
  "max_length": "number",
  "format": "string"  // "text", "markdown", "html"
}
```

### Link Contexts
```http
POST /v1/context/links
```

Creates relationships between contexts.

**Request Body:**
```json
{
  "source_id": "string",
  "target_id": "string",
  "relation_type": "string",
  "metadata": {}
}
```

## Context Types

### Conversation Context
- Manages conversation history
- Tracks speaker turns
- Maintains dialogue flow
- Supports multi-party conversations

### Document Context
- Handles document content
- Manages document structure
- Supports versioning
- Enables document analysis

### Session Context
- Manages user sessions
- Tracks user state
- Handles preferences
- Supports persistence

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

The Context API implements rate limiting based on your subscription tier:

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

## SDK Examples

### Python
```python
from langgraph.api import ContextAPI

api = ContextAPI(api_key="your-api-key")

# Create context
context = api.create_context({
    "name": "User Session",
    "description": "User session context",
    "type": "session",
    "metadata": {
        "source": "web_app",
        "tags": ["user", "session"]
    },
    "content": {
        "user_id": "123",
        "preferences": {}
    }
})

# Add context item
api.add_context_item(context.id, {
    "type": "preference",
    "content": {
        "theme": "dark",
        "language": "en"
    },
    "metadata": {
        "timestamp": "2024-03-06T12:00:00Z",
        "source": "user_settings"
    }
})
```

### JavaScript
```javascript
const { ContextAPI } = require('langgraph');

const api = new ContextAPI('your-api-key');

// Create context
const context = await api.createContext({
    name: 'User Session',
    description: 'User session context',
    type: 'session',
    metadata: {
        source: 'web_app',
        tags: ['user', 'session']
    },
    content: {
        userId: '123',
        preferences: {}
    }
});

// Add context item
await api.addContextItem(context.id, {
    type: 'preference',
    content: {
        theme: 'dark',
        language: 'en'
    },
    metadata: {
        timestamp: '2024-03-06T12:00:00Z',
        source: 'user_settings'
    }
});
```
