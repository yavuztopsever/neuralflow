# Web Search Integration API Documentation

This document provides detailed information about the Web Search Integration API endpoints for performing web searches and retrieving search results.

## Endpoints

### Initialize Search
```http
POST /v1/integration/websearch/initialize
```

Initializes web search with configuration.

**Request Body:**
```json
{
  "config": {
    "search_engine": "string",  // "google", "bing", "duckduckgo"
    "api_key": "string",
    "max_results": "number",
    "timeout": "number",
    "proxy": "string"
  }
}
```

**Response:**
```json
{
  "status": "string",
  "config": {},
  "capabilities": {
    "search_types": ["string"],
    "filters": ["string"],
    "max_results": "number"
  }
}
```

### Perform Search
```http
POST /v1/integration/websearch/search
```

Performs a web search query.

**Request Body:**
```json
{
  "query": "string",
  "options": {
    "type": "string",  // "web", "news", "images", "videos"
    "language": "string",
    "region": "string",
    "time_range": "string",
    "safe_search": "boolean",
    "max_results": "number"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "string",
      "title": "string",
      "url": "string",
      "snippet": "string",
      "metadata": {
        "source": "string",
        "date": "string",
        "language": "string"
      }
    }
  ],
  "total": "number",
  "took": "number",
  "next_page_token": "string"
}
```

### Get Search Results
```http
GET /v1/integration/websearch/results/{search_id}
```

Retrieves results from a specific search.

### Get Next Page
```http
GET /v1/integration/websearch/results/{search_id}/next
```

Retrieves the next page of search results.

### Filter Results
```http
POST /v1/integration/websearch/results/{search_id}/filter
```

Filters search results based on criteria.

**Request Body:**
```json
{
  "filters": {
    "date_range": {
      "start": "string",
      "end": "string"
    },
    "language": "string",
    "domain": "string",
    "content_type": "string"
  }
}
```

### Extract Content
```http
POST /v1/integration/websearch/extract
```

Extracts content from a webpage.

**Request Body:**
```json
{
  "url": "string",
  "options": {
    "extract_images": "boolean",
    "extract_links": "boolean",
    "extract_metadata": "boolean",
    "timeout": "number"
  }
}
```

**Response:**
```json
{
  "content": "string",
  "title": "string",
  "description": "string",
  "images": [],
  "links": [],
  "metadata": {}
}
```

### Get Search History
```http
GET /v1/integration/websearch/history
```

Retrieves search history with pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `sort`: Sort field (default: timestamp)
- `order`: Sort order (asc/desc)

### Clear Search History
```http
DELETE /v1/integration/websearch/history
```

Clears the search history.

## Supported Search Engines

### Google
- Web search
- News search
- Image search
- Video search
- Custom search

### Bing
- Web search
- News search
- Image search
- Video search
- API integration

### DuckDuckGo
- Web search
- News search
- Image search
- Privacy focused
- No API key required

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

The Web Search Integration API implements rate limiting based on your subscription tier:

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
from langgraph.api import WebSearchAPI

api = WebSearchAPI(api_key="your-api-key")

# Initialize search
api.initialize_search({
    "config": {
        "search_engine": "google",
        "api_key": "your-search-api-key",
        "max_results": 100
    }
})

# Perform search
results = api.search({
    "query": "python programming",
    "options": {
        "type": "web",
        "language": "en",
        "max_results": 10
    }
})

# Extract content
content = api.extract_content({
    "url": "https://example.com",
    "options": {
        "extract_images": True,
        "extract_links": True
    }
})
```

### JavaScript
```javascript
const { WebSearchAPI } = require('langgraph');

const api = new WebSearchAPI('your-api-key');

// Initialize search
await api.initializeSearch({
    config: {
        searchEngine: 'google',
        apiKey: 'your-search-api-key',
        maxResults: 100
    }
});

// Perform search
const results = await api.search({
    query: 'python programming',
    options: {
        type: 'web',
        language: 'en',
        maxResults: 10
    }
});

// Extract content
const content = await api.extractContent({
    url: 'https://example.com',
    options: {
        extractImages: true,
        extractLinks: true
    }
});
``` 