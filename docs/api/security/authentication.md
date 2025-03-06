# Authentication API Documentation

This document provides detailed information about the Authentication API endpoints for managing user authentication and authorization in the system.

## Endpoints

### Register User
```http
POST /v1/auth/register
```

Registers a new user account.

**Request Body:**
```json
{
  "email": "string",
  "password": "string",
  "name": "string",
  "organization": "string",
  "subscription_plan": "string"
}
```

**Response:**
```json
{
  "user_id": "string",
  "email": "string",
  "name": "string",
  "organization": "string",
  "subscription_plan": "string",
  "created_at": "string"
}
```

### Login
```http
POST /v1/auth/login
```

Authenticates a user and returns access tokens.

**Request Body:**
```json
{
  "email": "string",
  "password": "string",
  "device_info": {
    "device_id": "string",
    "platform": "string",
    "browser": "string"
  }
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "expires_in": "number",
  "token_type": "string",
  "user": {
    "id": "string",
    "email": "string",
    "name": "string",
    "organization": "string",
    "subscription_plan": "string"
  }
}
```

### Refresh Token
```http
POST /v1/auth/refresh
```

Refreshes an expired access token.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "expires_in": "number",
  "token_type": "string"
}
```

### Logout
```http
POST /v1/auth/logout
```

Logs out a user and invalidates their tokens.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

### Change Password
```http
POST /v1/auth/password/change
```

Changes a user's password.

**Request Body:**
```json
{
  "current_password": "string",
  "new_password": "string"
}
```

### Reset Password Request
```http
POST /v1/auth/password/reset-request
```

Requests a password reset email.

**Request Body:**
```json
{
  "email": "string"
}
```

### Reset Password
```http
POST /v1/auth/password/reset
```

Resets a user's password using a reset token.

**Request Body:**
```json
{
  "token": "string",
  "new_password": "string"
}
```

### Verify Email
```http
POST /v1/auth/email/verify
```

Verifies a user's email address.

**Request Body:**
```json
{
  "token": "string"
}
```

### Resend Verification Email
```http
POST /v1/auth/email/resend-verification
```

Resends the email verification link.

**Request Body:**
```json
{
  "email": "string"
}
```

### Get User Profile
```http
GET /v1/auth/profile
```

Retrieves the current user's profile.

**Response:**
```json
{
  "id": "string",
  "email": "string",
  "name": "string",
  "organization": "string",
  "subscription_plan": "string",
  "email_verified": "boolean",
  "created_at": "string",
  "updated_at": "string"
}
```

### Update User Profile
```http
PUT /v1/auth/profile
```

Updates the current user's profile.

**Request Body:**
```json
{
  "name": "string",
  "organization": "string"
}
```

### List Active Sessions
```http
GET /v1/auth/sessions
```

Lists all active sessions for the current user.

**Response:**
```json
{
  "sessions": [
    {
      "id": "string",
      "device_info": {},
      "ip_address": "string",
      "last_active": "string",
      "created_at": "string"
    }
  ]
}
```

### Revoke Session
```http
DELETE /v1/auth/sessions/{session_id}
```

Revokes a specific session.

### Revoke All Sessions
```http
DELETE /v1/auth/sessions
```

Revokes all sessions except the current one.

## Authentication Methods

### API Key Authentication
```http
Authorization: Bearer your-api-key
```

### JWT Authentication
```http
Authorization: Bearer your-jwt-token
```

### OAuth2 Authentication
```http
Authorization: Bearer your-oauth-token
```

## Security Features

### Password Requirements
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

### Token Security
- JWT tokens with RSA-256 signing
- Refresh token rotation
- Token expiration
- Token revocation

### Session Management
- Device tracking
- IP address logging
- Session timeout
- Concurrent session limits

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

The Authentication API implements rate limiting based on your subscription tier:

- Free Tier: 100 requests per minute
- Pro Tier: 1000 requests per minute
- Enterprise Tier: Custom limits

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623456789
```

## SDK Examples

### Python
```python
from langgraph.api import AuthAPI

api = AuthAPI(api_key="your-api-key")

# Register user
user = api.register({
    "email": "user@example.com",
    "password": "secure_password",
    "name": "John Doe",
    "organization": "Example Corp"
})

# Login
auth = api.login({
    "email": "user@example.com",
    "password": "secure_password",
    "device_info": {
        "device_id": "device123",
        "platform": "web",
        "browser": "chrome"
    }
})

# Get profile
profile = api.get_profile()
```

### JavaScript
```javascript
const { AuthAPI } = require('langgraph');

const api = new AuthAPI('your-api-key');

// Register user
const user = await api.register({
    email: 'user@example.com',
    password: 'secure_password',
    name: 'John Doe',
    organization: 'Example Corp'
});

// Login
const auth = await api.login({
    email: 'user@example.com',
    password: 'secure_password',
    device_info: {
        device_id: 'device123',
        platform: 'web',
        browser: 'chrome'
    }
});

// Get profile
const profile = await api.get_profile();
``` 