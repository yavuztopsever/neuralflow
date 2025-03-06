# Memory System Documentation

This document provides detailed information about the LangGraph memory system, including its components, management, and integration with workflows.

## Overview

The LangGraph memory system implements a multi-level memory architecture that enables efficient storage, retrieval, and management of information across different time scales and contexts.

## Memory Types

### 1. Short-term Memory
- **Purpose**: Store recent context and session data
- **Characteristics**:
  - Fast access
  - Limited capacity
  - Temporary storage
  - Session-specific

### 2. Mid-term Memory
- **Purpose**: Store session-level information
- **Characteristics**:
  - Moderate access speed
  - Medium capacity
  - Session persistence
  - Context-aware

### 3. Long-term Memory
- **Purpose**: Store historical data and knowledge
- **Characteristics**:
  - Persistent storage
  - Large capacity
  - Semantic search
  - Knowledge base

## Memory Components

### 1. Memory Store
```json
{
  "id": "string",
  "type": "string",
  "name": "string",
  "description": "string",
  "config": {
    "capacity": "number",
    "ttl": "number",
    "persistence": "string",
    "indexing": "string"
  },
  "metadata": {}
}
```

### 2. Memory Item
```json
{
  "id": "string",
  "store_id": "string",
  "content": "string",
  "metadata": {
    "timestamp": "string",
    "type": "string",
    "tags": ["string"],
    "source": "string"
  },
  "embedding": "array",
  "relationships": []
}
```

## Memory Operations

### 1. Storage Operations
- **Add Item**: Store new memory items
- **Update Item**: Modify existing items
- **Delete Item**: Remove items
- **Batch Operations**: Handle multiple items

### 2. Retrieval Operations
- **Get Item**: Retrieve specific items
- **Search Items**: Find items by query
- **List Items**: Enumerate items
- **Filter Items**: Apply filters

### 3. Management Operations
- **Create Store**: Initialize new store
- **Delete Store**: Remove store
- **Update Store**: Modify store settings
- **Optimize Store**: Improve performance

## Memory Integration

### 1. Workflow Integration
```python
# Example workflow with memory
workflow = {
    "nodes": [
        {
            "id": "memory",
            "type": "memory",
            "config": {
                "store_id": "string",
                "operation": "string",
                "options": {}
            }
        }
    ]
}
```

### 2. Context Integration
- **Context Retrieval**: Get relevant context
- **Context Update**: Update with new information
- **Context Filtering**: Filter relevant context
- **Context Summarization**: Generate summaries

## Memory Management

### 1. Storage Management
- **Capacity Management**: Handle storage limits
- **Cleanup**: Remove expired items
- **Optimization**: Improve storage efficiency
- **Backup**: Data backup and recovery

### 2. Performance Management
- **Caching**: Implement caching strategies
- **Indexing**: Optimize search performance
- **Load Balancing**: Distribute load
- **Monitoring**: Track performance metrics

### 3. Security Management
- **Access Control**: Manage permissions
- **Encryption**: Secure data storage
- **Audit Logging**: Track operations
- **Compliance**: Ensure data compliance

## Memory APIs

### 1. Store Management API
```python
# Create store
store = api.create_memory_store({
    "type": "short_term",
    "name": "Session Store",
    "config": {
        "capacity": 1000,
        "ttl": 3600
    }
})

# Add items
api.add_memory_items(store.id, {
    "items": [
        {
            "content": "Sample content",
            "metadata": {
                "type": "text",
                "timestamp": "2024-03-06T12:00:00Z"
            }
        }
    ]
})
```

### 2. Retrieval API
```python
# Search items
results = api.search_memory(store.id, {
    "query": "search query",
    "options": {
        "k": 5,
        "score_threshold": 0.7
    }
})

# Get items
items = api.get_memory_items(store.id, {
    "ids": ["item1", "item2"]
})
```

## Best Practices

### 1. Memory Design
- Choose appropriate memory types
- Set reasonable capacity limits
- Implement proper cleanup
- Use meaningful metadata

### 2. Performance
- Optimize storage usage
- Implement efficient retrieval
- Use caching effectively
- Monitor performance

### 3. Security
- Secure sensitive data
- Implement access controls
- Monitor access patterns
- Regular security audits

### 4. Maintenance
- Regular cleanup
- Performance optimization
- Security updates
- Documentation updates

## SDK Examples

### Python
```python
from langgraph.api import MemoryAPI

api = MemoryAPI(api_key="your-api-key")

# Create memory store
store = api.create_memory_store({
    "type": "long_term",
    "name": "Knowledge Base",
    "config": {
        "capacity": 10000,
        "persistence": "disk",
        "indexing": "semantic"
    }
})

# Add memory items
api.add_memory_items(store.id, {
    "items": [
        {
            "content": "Paris is the capital of France.",
            "metadata": {
                "type": "fact",
                "category": "geography",
                "source": "encyclopedia"
            }
        }
    ]
})

# Search memory
results = api.search_memory(store.id, {
    "query": "What is the capital of France?",
    "options": {
        "k": 1,
        "score_threshold": 0.8
    }
})
```

### JavaScript
```javascript
const { MemoryAPI } = require('langgraph');

const api = new MemoryAPI('your-api-key');

// Create memory store
const store = await api.createMemoryStore({
    type: 'long_term',
    name: 'Knowledge Base',
    config: {
        capacity: 10000,
        persistence: 'disk',
        indexing: 'semantic'
    }
});

// Add memory items
await api.addMemoryItems(store.id, {
    items: [
        {
            content: 'Paris is the capital of France.',
            metadata: {
                type: 'fact',
                category: 'geography',
                source: 'encyclopedia'
            }
        }
    ]
});

// Search memory
const results = await api.searchMemory(store.id, {
    query: 'What is the capital of France?',
    options: {
        k: 1,
        scoreThreshold: 0.8
    }
});
``` 