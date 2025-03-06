# LangChain Integration API Documentation

This document provides detailed information about the LangChain Integration API endpoints for integrating LangChain components with LangGraph.

## Endpoints

### Initialize LangChain
```http
POST /v1/integration/langchain/initialize
```

Initializes LangChain integration with configuration.

**Request Body:**
```json
{
  "config": {
    "openai_api_key": "string",
    "model_name": "string",
    "temperature": "number",
    "max_tokens": "number",
    "memory_type": "string",
    "vector_store_type": "string",
    "max_memory_items": "number",
    "persist_directory": "string"
  }
}
```

**Response:**
```json
{
  "status": "string",
  "config": {},
  "initialized_components": {
    "llm": "boolean",
    "memory": "boolean",
    "vector_store": "boolean",
    "tools": "boolean"
  }
}
```

### Configure LLM
```http
POST /v1/integration/langchain/llm
```

Configures the Language Model settings.

**Request Body:**
```json
{
  "model_name": "string",
  "temperature": "number",
  "max_tokens": "number",
  "top_p": "number",
  "frequency_penalty": "number",
  "presence_penalty": "number"
}
```

### Configure Memory
```http
POST /v1/integration/langchain/memory
```

Configures the memory system.

**Request Body:**
```json
{
  "type": "string",  // "buffer", "summary", "window"
  "max_items": "number",
  "return_messages": "boolean",
  "memory_key": "string"
}
```

### Configure Vector Store
```http
POST /v1/integration/langchain/vectorstore
```

Configures the vector store settings.

**Request Body:**
```json
{
  "type": "string",  // "chroma", "faiss"
  "persist_directory": "string",
  "collection_name": "string",
  "embedding_function": "string"
}
```

### Configure Tools
```http
POST /v1/integration/langchain/tools
```

Configures available LangChain tools.

**Request Body:**
```json
{
  "tools": [
    {
      "name": "string",
      "type": "string",
      "config": {}
    }
  ]
}
```

### Get Available Tools
```http
GET /v1/integration/langchain/tools
```

Lists all available LangChain tools.

**Response:**
```json
{
  "tools": [
    {
      "name": "string",
      "type": "string",
      "description": "string",
      "parameters": {}
    }
  ]
}
```

### Execute Chain
```http
POST /v1/integration/langchain/execute
```

Executes a LangChain chain.

**Request Body:**
```json
{
  "chain_type": "string",
  "input": {},
  "options": {
    "timeout": "number",
    "max_steps": "number"
  }
}
```

### Get Chain Status
```http
GET /v1/integration/langchain/chains/{chain_id}
```

Retrieves the status of a chain execution.

## Supported Components

### Language Models
- OpenAI GPT-4
- OpenAI GPT-3.5
- Anthropic Claude
- Custom Models

### Memory Types
- ConversationBufferMemory
- ConversationSummaryMemory
- ConversationBufferWindowMemory
- Custom Memory

### Vector Stores
- Chroma
- FAISS
- Pinecone
- Custom Stores

### Tools
- Search
- Calculator
- Python REPL
- Web Browser
- Custom Tools

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

The LangChain Integration API implements rate limiting based on your subscription tier:

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
from langgraph.api import LangChainAPI

api = LangChainAPI(api_key="your-api-key")

# Initialize LangChain
api.initialize_langchain({
    "config": {
        "openai_api_key": "your-key",
        "model_name": "gpt-4",
        "temperature": 0.7,
        "memory_type": "buffer",
        "vector_store_type": "chroma"
    }
})

# Configure LLM
api.configure_llm({
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000
})

# Execute chain
response = api.execute_chain({
    "chain_type": "qa",
    "input": {
        "question": "What is the capital of France?",
        "context": "Paris is the capital of France."
    }
})
```

### JavaScript
```javascript
const { LangChainAPI } = require('langgraph');

const api = new LangChainAPI('your-api-key');

// Initialize LangChain
await api.initializeLangChain({
    config: {
        openaiApiKey: 'your-key',
        modelName: 'gpt-4',
        temperature: 0.7,
        memoryType: 'buffer',
        vectorStoreType: 'chroma'
    }
});

// Configure LLM
await api.configureLLM({
    modelName: 'gpt-4',
    temperature: 0.7,
    maxTokens: 2000
});

// Execute chain
const response = await api.executeChain({
    chainType: 'qa',
    input: {
        question: 'What is the capital of France?',
        context: 'Paris is the capital of France.'
    }
});
```
