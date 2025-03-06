# Memory API Documentation

This document provides detailed information about the Memory API endpoints for managing different types of memory in the system.

## Endpoints

### Create Memory Store
```http
POST /v1/memory/stores
```

Creates a new memory store.

**Request Body:**
```json
{
  "name": "string",
  "type": "string",  // "short_term", "mid_term", "long_term"
  "config": {
    "max_items": "number",
    "ttl": "number",
    "persistence": "boolean"
  }
}
```

**Response:**
```json
{
  "id": "string",
  "name": "string",
  "type": "string",
  "config": {},
  "created_at": "string",
  "updated_at": "string"
}
```

### Get Memory Store
```http
GET /v1/memory/stores/{store_id}
```

Retrieves a specific memory store by ID.

### Update Memory Store
```http
PUT /v1/memory/stores/{store_id}
```

Updates an existing memory store configuration.

### Delete Memory Store
```http
DELETE /v1/memory/stores/{store_id}
```

Deletes a specific memory store.

### Store Memory
```http
POST /v1/memory/stores/{store_id}/items
```

Stores a new memory item.

**Request Body:**
```json
{
  "key": "string",
  "value": {},
  "metadata": {
    "timestamp": "string",
    "source": "string",
    "tags": ["string"]
  }
}
```

### Retrieve Memory
```http
GET /v1/memory/stores/{store_id}/items/{key}
```

Retrieves a specific memory item by key.

### Search Memory
```http
POST /v1/memory/stores/{store_id}/search
```

Searches for memory items using semantic search.

**Request Body:**
```json
{
  "query": "string",
  "filters": {
    "tags": ["string"],
    "timestamp_range": {
      "start": "string",
      "end": "string"
    }
  },
  "limit": "number"
}
```

### Update Memory
```http
PUT /v1/memory/stores/{store_id}/items/{key}
```

Updates an existing memory item.

### Delete Memory
```http
DELETE /v1/memory/stores/{store_id}/items/{key}
```

Deletes a specific memory item.

### List Memory Items
```http
GET /v1/memory/stores/{store_id}/items
```

Lists all memory items in a store with pagination support.

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `sort`: Sort field (default: timestamp)
- `order`: Sort order (asc/desc)

### Summarize Memory
```http
POST /v1/memory/stores/{store_id}/summarize
```

Generates a summary of memory items.

**Request Body:**
```json
{
  "items": ["string"],
  "max_length": "number"
}
```

### Link Memories
```http
POST /v1/memory/stores/{store_id}/links
```

Creates relationships between memory items.

**Request Body:**
```json
{
  "source_key": "string",
  "target_key": "string",
  "relation_type": "string",
  "metadata": {}
}
```

## Memory Types

### Short-term Memory
- Temporary storage for recent context
- Limited capacity
- Fast access
- Automatic cleanup

### Mid-term Memory
- Session-level storage
- Moderate capacity
- Balanced access speed
- Configurable retention

### Long-term Memory
- Persistent storage
- Large capacity
- Optimized for retrieval
- Manual cleanup

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

The Memory API implements rate limiting based on your subscription tier:

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
from langgraph.api import MemoryAPI

api = MemoryAPI(api_key="your-api-key")

# Create memory store
store = api.create_store({
    "name": "My Store",
    "type": "long_term",
    "config": {
        "max_items": 1000,
        "persistence": True
    }
})

# Store memory
api.store_memory(store.id, {
    "key": "user_preference",
    "value": {"theme": "dark"},
    "metadata": {
        "timestamp": "2024-03-06T12:00:00Z",
        "source": "user_settings"
    }
})

# Retrieve memory
memory = api.get_memory(store.id, "user_preference")
```

### JavaScript
```javascript
const { MemoryAPI } = require('langgraph');

const api = new MemoryAPI('your-api-key');

// Create memory store
const store = await api.createStore({
    name: 'My Store',
    type: 'long_term',
    config: {
        maxItems: 1000,
        persistence: true
    }
});

// Store memory
await api.storeMemory(store.id, {
    key: 'user_preference',
    value: { theme: 'dark' },
    metadata: {
        timestamp: '2024-03-06T12:00:00Z',
        source: 'user_settings'
    }
});

// Retrieve memory
const memory = await api.getMemory(store.id, 'user_preference');
```
