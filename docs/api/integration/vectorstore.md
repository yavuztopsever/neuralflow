# Vector Store Integration API Documentation

This document provides detailed information about the Vector Store Integration API endpoints for managing vector stores and embeddings in the system.

## Endpoints

### Initialize Vector Store
```http
POST /v1/integration/vectorstore/initialize
```

Initializes a vector store with configuration.

**Request Body:**
```json
{
  "type": "string",  // "chroma", "faiss", "pinecone"
  "config": {
    "persist_directory": "string",
    "collection_name": "string",
    "embedding_function": "string",
    "dimension": "number",
    "metric": "string"
  }
}
```

**Response:**
```json
{
  "status": "string",
  "store_id": "string",
  "config": {},
  "stats": {
    "total_vectors": "number",
    "dimension": "number",
    "last_updated": "string"
  }
}
```

### Add Documents
```http
POST /v1/integration/vectorstore/{store_id}/documents
```

Adds documents to the vector store.

**Request Body:**
```json
{
  "documents": [
    {
      "id": "string",
      "content": "string",
      "metadata": {},
      "embedding": "array"
    }
  ],
  "options": {
    "batch_size": "number",
    "upsert": "boolean"
  }
}
```

### Search Documents
```http
POST /v1/integration/vectorstore/{store_id}/search
```

Searches for similar documents using vector similarity.

**Request Body:**
```json
{
  "query": "string",
  "query_embedding": "array",
  "options": {
    "k": "number",
    "score_threshold": "number",
    "filter": {}
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "string",
      "content": "string",
      "metadata": {},
      "score": "number"
    }
  ],
  "total": "number",
  "took": "number"
}
```

### Update Document
```http
PUT /v1/integration/vectorstore/{store_id}/documents/{document_id}
```

Updates an existing document in the vector store.

**Request Body:**
```json
{
  "content": "string",
  "metadata": {},
  "embedding": "array"
}
```

### Delete Document
```http
DELETE /v1/integration/vectorstore/{store_id}/documents/{document_id}
```

Deletes a document from the vector store.

### Get Document
```http
GET /v1/integration/vectorstore/{store_id}/documents/{document_id}
```

Retrieves a specific document from the vector store.

### List Documents
```http
GET /v1/integration/vectorstore/{store_id}/documents
```

Lists all documents in the vector store with pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `sort`: Sort field (default: created_at)
- `order`: Sort order (asc/desc)

### Get Store Stats
```http
GET /v1/integration/vectorstore/{store_id}/stats
```

Retrieves statistics about the vector store.

**Response:**
```json
{
  "total_documents": "number",
  "total_vectors": "number",
  "dimension": "number",
  "last_updated": "string",
  "index_type": "string",
  "storage_size": "number"
}
```

### Optimize Store
```http
POST /v1/integration/vectorstore/{store_id}/optimize
```

Optimizes the vector store index.

**Request Body:**
```json
{
  "options": {
    "force": "boolean",
    "reindex": "boolean"
  }
}
```

## Supported Vector Stores

### Chroma
- Local persistence
- In-memory operation
- Custom embeddings
- Metadata filtering

### FAISS
- High performance
- GPU acceleration
- Multiple index types
- Custom metrics

### Pinecone
- Cloud hosted
- Serverless scaling
- Real-time updates
- Namespace support

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

The Vector Store Integration API implements rate limiting based on your subscription tier:

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
from langgraph.api import VectorStoreAPI

api = VectorStoreAPI(api_key="your-api-key")

# Initialize vector store
store = api.initialize_store({
    "type": "chroma",
    "config": {
        "persist_directory": "./data",
        "collection_name": "documents",
        "embedding_function": "text-embedding-ada-002"
    }
})

# Add documents
api.add_documents(store.id, {
    "documents": [
        {
            "id": "doc1",
            "content": "Sample document content",
            "metadata": {"source": "example"}
        }
    ]
})

# Search documents
results = api.search_documents(store.id, {
    "query": "sample query",
    "options": {
        "k": 5,
        "score_threshold": 0.7
    }
})
```

### JavaScript
```javascript
const { VectorStoreAPI } = require('langgraph');

const api = new VectorStoreAPI('your-api-key');

// Initialize vector store
const store = await api.initializeStore({
    type: 'chroma',
    config: {
        persistDirectory: './data',
        collectionName: 'documents',
        embeddingFunction: 'text-embedding-ada-002'
    }
});

// Add documents
await api.addDocuments(store.id, {
    documents: [
        {
            id: 'doc1',
            content: 'Sample document content',
            metadata: { source: 'example' }
        }
    ]
});

// Search documents
const results = await api.searchDocuments(store.id, {
    query: 'sample query',
    options: {
        k: 5,
        scoreThreshold: 0.7
    }
});
``` 