# NeuralFlow API Documentation

This section provides comprehensive documentation for the NeuralFlow API, including all available endpoints, methods, and usage examples.

## Table of Contents

### Core API
- [Graph API](graph.md) - Graph workflow management endpoints
- [Memory API](memory.md) - Memory management endpoints
- [Context API](context.md) - Context management endpoints
- [Response API](response.md) - Response generation endpoints

### Integration APIs
- [LangChain Integration](langchain.md) - LangChain integration endpoints
- [Vector Store Integration](vector_store.md) - Vector store integration endpoints
- [Web Search Integration](web_search.md) - Web search integration endpoints

### Authentication & Security
- [Authentication](auth.md) - Authentication methods and endpoints
- [Rate Limiting](rate_limiting.md) - Rate limiting configuration
- [Security](security.md) - Security best practices

### Examples
- [Basic Usage](examples/basic.md) - Simple API usage examples
- [Advanced Usage](examples/advanced.md) - Complex API patterns
- [Integration Examples](examples/integration.md) - Integration examples

## API Overview

The NeuralFlow API is designed to be RESTful and follows these principles:

- **Resource-based URLs**: All endpoints are resource-based
- **HTTP Methods**: Standard HTTP methods (GET, POST, PUT, DELETE)
- **JSON Format**: Request and response bodies use JSON
- **Versioning**: API versioning through URL prefix
- **Authentication**: Token-based authentication

## Base URL

```
https://api.neuralflow.com/v1
```

## Authentication

All API requests require authentication using an API key. Include your API key in the request header:

```http
Authorization: Bearer your-api-key
```

## Rate Limiting

The API implements rate limiting to ensure fair usage. Rate limits are specified in the response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623456789
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages in the response body:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameters",
    "details": {
      "field": "query",
      "reason": "required"
    }
  }
}
```

## Pagination

List endpoints support pagination using the following parameters:

- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)
- `cursor`: Cursor for cursor-based pagination

## Webhooks

The API supports webhooks for asynchronous operations. Configure webhooks in your account settings:

```json
{
  "url": "https://your-domain.com/webhook",
  "events": ["workflow.completed", "workflow.failed"],
  "secret": "your-webhook-secret"
}
```

## SDK Support

Official SDKs are available for:

- [Python](https://github.com/yavuztopsever/neuralflow-python)
- [JavaScript](https://github.com/yavuztopsever/neuralflow-js)
- [Go](https://github.com/yavuztopsever/neuralflow-go)

## Getting Started

1. [Obtain an API key](auth.md#getting-an-api-key)
2. [Set up your environment](examples/basic.md#setup)
3. [Make your first API call](examples/basic.md#first-call)

## Support

For API support:

1. Check the [FAQ](../guides/faq.md)
2. Review the [Troubleshooting Guide](../guides/troubleshooting.md)
3. Contact support at support@neuralflow.com 