# Rate Limiting API Documentation

This document provides detailed information about the Rate Limiting API endpoints for managing and monitoring API rate limits in the system.

## Endpoints

### Get Rate Limits
```http
GET /v1/rate-limits
```

Retrieves current rate limits for the authenticated user.

**Response:**
```json
{
  "limits": {
    "requests_per_minute": "number",
    "requests_per_hour": "number",
    "requests_per_day": "number",
    "concurrent_requests": "number"
  },
  "usage": {
    "current_minute": "number",
    "current_hour": "number",
    "current_day": "number",
    "concurrent": "number"
  },
  "subscription": {
    "plan": "string",
    "features": ["string"],
    "expires_at": "string"
  }
}
```

### Get Rate Limit Status
```http
GET /v1/rate-limits/status
```

Retrieves detailed rate limit status and usage information.

**Response:**
```json
{
  "status": "string",  // "normal", "warning", "critical"
  "limits": {
    "requests_per_minute": "number",
    "requests_per_hour": "number",
    "requests_per_day": "number",
    "concurrent_requests": "number"
  },
  "usage": {
    "current_minute": "number",
    "current_hour": "number",
    "current_day": "number",
    "concurrent": "number"
  },
  "reset_times": {
    "minute": "string",
    "hour": "string",
    "day": "string"
  }
}
```

### Get Rate Limit History
```http
GET /v1/rate-limits/history
```

Retrieves historical rate limit usage data.

**Query Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `interval`: Time interval (minute, hour, day)
- `limit`: Number of records (default: 100)

**Response:**
```json
{
  "history": [
    {
      "timestamp": "string",
      "requests": "number",
      "limit": "number",
      "status": "string"
    }
  ],
  "total": "number",
  "interval": "string"
}
```

### Get Rate Limit Alerts
```http
GET /v1/rate-limits/alerts
```

Retrieves rate limit alert settings and history.

**Response:**
```json
{
  "alerts": [
    {
      "id": "string",
      "type": "string",
      "threshold": "number",
      "enabled": "boolean",
      "notification_channels": ["string"]
    }
  ],
  "history": [
    {
      "id": "string",
      "type": "string",
      "timestamp": "string",
      "threshold": "number",
      "actual": "number",
      "status": "string"
    }
  ]
}
```

### Configure Rate Limit Alerts
```http
POST /v1/rate-limits/alerts
```

Configures rate limit alert settings.

**Request Body:**
```json
{
  "alerts": [
    {
      "type": "string",  // "warning", "critical"
      "threshold": "number",
      "notification_channels": ["string"]
    }
  ]
}
```

### Get Rate Limit Exceptions
```http
GET /v1/rate-limits/exceptions
```

Retrieves list of rate limit exceptions.

**Response:**
```json
{
  "exceptions": [
    {
      "id": "string",
      "endpoint": "string",
      "limit": "number",
      "reason": "string",
      "expires_at": "string"
    }
  ]
}
```

### Request Rate Limit Exception
```http
POST /v1/rate-limits/exceptions
```

Requests a rate limit exception.

**Request Body:**
```json
{
  "endpoint": "string",
  "requested_limit": "number",
  "reason": "string",
  "duration": "string"  // ISO 8601 duration
}
```

### Get Rate Limit Policies
```http
GET /v1/rate-limits/policies
```

Retrieves rate limit policies for different subscription tiers.

**Response:**
```json
{
  "policies": {
    "free": {
      "requests_per_minute": "number",
      "requests_per_hour": "number",
      "requests_per_day": "number",
      "concurrent_requests": "number",
      "features": ["string"]
    },
    "pro": {
      "requests_per_minute": "number",
      "requests_per_hour": "number",
      "requests_per_day": "number",
      "concurrent_requests": "number",
      "features": ["string"]
    },
    "enterprise": {
      "requests_per_minute": "number",
      "requests_per_hour": "number",
      "requests_per_day": "number",
      "concurrent_requests": "number",
      "features": ["string"]
    }
  }
}
```

## Rate Limit Types

### Request Rate Limits
- Requests per minute
- Requests per hour
- Requests per day
- Concurrent requests

### Feature Rate Limits
- API calls
- Storage usage
- Bandwidth usage
- Custom features

### Subscription Tiers
- Free Tier
- Pro Tier
- Enterprise Tier

## Rate Limit Headers

All API responses include rate limit information in headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623456789
X-RateLimit-Window: 60
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
    "message": "Rate limit exceeded",
    "retry_after": "number"
  }
}
```

## SDK Examples

### Python
```python
from langgraph.api import RateLimitAPI

api = RateLimitAPI(api_key="your-api-key")

# Get rate limits
limits = api.get_rate_limits()

# Get rate limit status
status = api.get_rate_limit_status()

# Configure alerts
api.configure_alerts({
    "alerts": [
        {
            "type": "warning",
            "threshold": 80,
            "notification_channels": ["email"]
        }
    ]
})

# Request exception
api.request_exception({
    "endpoint": "/v1/api/endpoint",
    "requested_limit": 200,
    "reason": "High traffic period",
    "duration": "PT1H"
})
```

### JavaScript
```javascript
const { RateLimitAPI } = require('langgraph');

const api = new RateLimitAPI('your-api-key');

// Get rate limits
const limits = await api.getRateLimits();

// Get rate limit status
const status = await api.getRateLimitStatus();

// Configure alerts
await api.configureAlerts({
    alerts: [
        {
            type: 'warning',
            threshold: 80,
            notificationChannels: ['email']
        }
    ]
});

// Request exception
await api.requestException({
    endpoint: '/v1/api/endpoint',
    requestedLimit: 200,
    reason: 'High traffic period',
    duration: 'PT1H'
});
```
