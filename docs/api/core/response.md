# Response API Documentation

This document provides detailed information about the Response API endpoints for managing and generating responses in the system.

## Endpoints

### Generate Response
```http
POST /v1/responses/generate
```

Generates a response based on the provided input and context.

**Request Body:**
```json
{
  "input": "string",
  "context": {
    "conversation_id": "string",
    "memory_ids": ["string"],
    "context_ids": ["string"]
  },
  "options": {
    "style": "string",
    "format": "string",
    "max_length": "number",
    "temperature": "number",
    "include_sources": "boolean"
  }
}
```

**Response:**
```json
{
  "id": "string",
  "content": "string",
  "metadata": {
    "sources": [],
    "confidence": "number",
    "processing_time": "number"
  },
  "created_at": "string"
}
```

### Get Response
```http
GET /v1/responses/{response_id}
```

Retrieves a specific response by ID.

### Update Response
```http
PUT /v1/responses/{response_id}
```

Updates an existing response.

### Delete Response
```http
DELETE /v1/responses/{response_id}
```

Deletes a specific response.

### List Responses
```http
GET /v1/responses
```

Lists all responses with pagination support.

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `sort`: Sort field (default: created_at)
- `order`: Sort order (asc/desc)

### Stream Response
```http
POST /v1/responses/stream
```

Streams a response generation in real-time.

**Request Body:**
```json
{
  "input": "string",
  "context": {},
  "options": {}
}
```

### Analyze Response
```http
POST /v1/responses/{response_id}/analyze
```

Analyzes a response for quality and relevance.

**Response:**
```json
{
  "quality_score": "number",
  "relevance_score": "number",
  "sentiment": "string",
  "topics": ["string"],
  "suggestions": ["string"]
}
```

### Format Response
```http
POST /v1/responses/{response_id}/format
```

Formats a response in a specific style.

**Request Body:**
```json
{
  "style": "string",
  "format": "string",
  "options": {}
}
```

### Translate Response
```http
POST /v1/responses/{response_id}/translate
```

Translates a response to a target language.

**Request Body:**
```json
{
  "target_language": "string",
  "preserve_formatting": "boolean"
}
```

## Response Types

### Text Response
- Plain text output
- Markdown support
- HTML formatting
- Custom styling

### Structured Response
- JSON format
- XML format
- Custom schemas
- Data validation

### Rich Response
- Multimedia content
- Interactive elements
- Dynamic components
- Custom widgets

## Response Options

### Style Options
- Formal
- Casual
- Technical
- Creative
- Professional

### Format Options
- Text
- Markdown
- HTML
- JSON
- XML

### Generation Options
- Max length
- Temperature
- Top-p sampling
- Frequency penalty
- Presence penalty

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

The Response API implements rate limiting based on your subscription tier:

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
from langgraph.api import ResponseAPI

api = ResponseAPI(api_key="your-api-key")

# Generate response
response = api.generate_response({
    "input": "What is the capital of France?",
    "context": {
        "conversation_id": "conv_123",
        "memory_ids": ["mem_456"],
        "context_ids": ["ctx_789"]
    },
    "options": {
        "style": "formal",
        "format": "markdown",
        "max_length": 1000,
        "temperature": 0.7,
        "include_sources": True
    }
})

# Stream response
for chunk in api.stream_response({
    "input": "Tell me a story",
    "context": {},
    "options": {}
}):
    print(chunk)
```

### JavaScript
```javascript
const { ResponseAPI } = require('langgraph');

const api = new ResponseAPI('your-api-key');

// Generate response
const response = await api.generateResponse({
    input: 'What is the capital of France?',
    context: {
        conversationId: 'conv_123',
        memoryIds: ['mem_456'],
        contextIds: ['ctx_789']
    },
    options: {
        style: 'formal',
        format: 'markdown',
        maxLength: 1000,
        temperature: 0.7,
        includeSources: true
    }
});

// Stream response
const stream = await api.streamResponse({
    input: 'Tell me a story',
    context: {},
    options: {}
});

for await (const chunk of stream) {
    console.log(chunk);
}
```
