# Security API Documentation

This document provides detailed information about the Security API endpoints for managing security settings and monitoring security events in the system.

## Endpoints

### Get Security Settings
```http
GET /v1/security/settings
```

Retrieves current security settings for the organization.

**Response:**
```json
{
  "settings": {
    "password_policy": {
      "min_length": "number",
      "require_uppercase": "boolean",
      "require_lowercase": "boolean",
      "require_numbers": "boolean",
      "require_special": "boolean",
      "max_age_days": "number"
    },
    "session_policy": {
      "timeout_minutes": "number",
      "max_concurrent": "number",
      "require_reauth": "boolean"
    },
    "mfa_policy": {
      "enabled": "boolean",
      "methods": ["string"],
      "require_all": "boolean"
    },
    "ip_policy": {
      "whitelist": ["string"],
      "blacklist": ["string"],
      "require_whitelist": "boolean"
    }
  }
}
```

### Update Security Settings
```http
PUT /v1/security/settings
```

Updates security settings for the organization.

**Request Body:**
```json
{
  "settings": {
    "password_policy": {},
    "session_policy": {},
    "mfa_policy": {},
    "ip_policy": {}
  }
}
```

### Get Security Events
```http
GET /v1/security/events
```

Retrieves security event logs.

**Query Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `event_type`: Event type filter
- `severity`: Severity filter
- `limit`: Number of records (default: 100)

**Response:**
```json
{
  "events": [
    {
      "id": "string",
      "timestamp": "string",
      "type": "string",
      "severity": "string",
      "user_id": "string",
      "ip_address": "string",
      "details": {}
    }
  ],
  "total": "number"
}
```

### Get Security Alerts
```http
GET /v1/security/alerts
```

Retrieves active security alerts.

**Response:**
```json
{
  "alerts": [
    {
      "id": "string",
      "type": "string",
      "severity": "string",
      "status": "string",
      "created_at": "string",
      "details": {},
      "actions": ["string"]
    }
  ]
}
```

### Configure Security Alerts
```http
POST /v1/security/alerts
```

Configures security alert settings.

**Request Body:**
```json
{
  "alerts": [
    {
      "type": "string",
      "severity": "string",
      "threshold": "number",
      "notification_channels": ["string"],
      "actions": ["string"]
    }
  ]
}
```

### Get Security Audit Log
```http
GET /v1/security/audit
```

Retrieves security audit logs.

**Query Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `user_id`: User ID filter
- `action`: Action filter
- `limit`: Number of records (default: 100)

**Response:**
```json
{
  "audit_logs": [
    {
      "id": "string",
      "timestamp": "string",
      "user_id": "string",
      "action": "string",
      "resource": "string",
      "ip_address": "string",
      "details": {}
    }
  ],
  "total": "number"
}
```

### Get Security Reports
```http
GET /v1/security/reports
```

Retrieves security reports and analytics.

**Query Parameters:**
- `report_type`: Report type
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `format`: Report format (json, csv, pdf)

**Response:**
```json
{
  "report_id": "string",
  "type": "string",
  "generated_at": "string",
  "period": {
    "start": "string",
    "end": "string"
  },
  "data": {},
  "summary": {}
}
```

### Get Security Compliance
```http
GET /v1/security/compliance
```

Retrieves security compliance status.

**Response:**
```json
{
  "compliance": {
    "status": "string",
    "last_audit": "string",
    "requirements": [
      {
        "id": "string",
        "name": "string",
        "status": "string",
        "details": {}
      }
    ],
    "certifications": [
      {
        "name": "string",
        "status": "string",
        "expires_at": "string"
      }
    ]
  }
}
```

## Security Features

### Authentication Security
- Multi-factor authentication
- Password policies
- Session management
- IP restrictions

### Data Security
- Encryption at rest
- Encryption in transit
- Data backup
- Data retention

### Access Control
- Role-based access
- Resource permissions
- API key management
- IP whitelisting

### Monitoring
- Security events
- Audit logging
- Compliance reporting
- Alert management

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

## SDK Examples

### Python
```python
from langgraph.api import SecurityAPI

api = SecurityAPI(api_key="your-api-key")

# Get security settings
settings = api.get_security_settings()

# Update security settings
api.update_security_settings({
    "settings": {
        "password_policy": {
            "min_length": 12,
            "require_uppercase": True,
            "require_special": True
        },
        "mfa_policy": {
            "enabled": True,
            "methods": ["totp", "sms"]
        }
    }
})

# Get security events
events = api.get_security_events({
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-03-06T23:59:59Z",
    "severity": "high"
})

# Configure alerts
api.configure_alerts({
    "alerts": [
        {
            "type": "failed_login",
            "severity": "high",
            "threshold": 5,
            "notification_channels": ["email", "slack"]
        }
    ]
})
```

### JavaScript
```javascript
const { SecurityAPI } = require('langgraph');

const api = new SecurityAPI('your-api-key');

// Get security settings
const settings = await api.getSecuritySettings();

// Update security settings
await api.updateSecuritySettings({
    settings: {
        passwordPolicy: {
            minLength: 12,
            requireUppercase: true,
            requireSpecial: true
        },
        mfaPolicy: {
            enabled: true,
            methods: ['totp', 'sms']
        }
    }
});

// Get security events
const events = await api.getSecurityEvents({
    startDate: '2024-01-01T00:00:00Z',
    endDate: '2024-03-06T23:59:59Z',
    severity: 'high'
});

// Configure alerts
await api.configureAlerts({
    alerts: [
        {
            type: 'failed_login',
            severity: 'high',
            threshold: 5,
            notificationChannels: ['email', 'slack']
        }
    ]
});
```
