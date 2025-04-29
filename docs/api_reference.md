# Healthcare MCP Server API Reference

## Authentication

### Login
`POST /login`

Authenticate a user and receive an access token.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "token": "string",
  "user": {
    "id": "integer",
    "role": "string",
    "username": "string"
  }
}
```

**Status Codes:**
- 200: Success
- 401: Invalid credentials
- 400: Invalid request format

### Register
`POST /register`

Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "role": "string (patient|doctor)"
}
```

**Response:**
```json
{
  "message": "Registration successful",
  "user_id": "integer"
}
```

## Medical Analysis

### Analyze Symptoms
`POST /api/analyze-symptoms`

Analyze patient symptoms using medical NLP.

**Request Body:**
```json
{
  "symptoms": ["string"],
  "patient_data": {
    "age": "integer",
    "gender": "string",
    "medical_history": "string",
    "medications": ["string"]
  }
}
```

**Response:**
```json
{
  "analysis": {
    "diagnoses": [
      {
        "condition": "string",
        "confidence": "float",
        "urgency": "string"
      }
    ],
    "recommended_tests": ["string"],
    "immediate_actions": ["string"]
  }
}
```

### Analyze Medical Image
`POST /api/analyze-image`

Process and analyze medical images.

**Request Body:**
```json
{
  "image": "binary",
  "type": "string (x-ray|mri|ct|ultrasound)",
  "context": "string"
}
```

**Response:**
```json
{
  "classification": {
    "label": "string",
    "confidence": "float"
  },
  "findings": ["string"],
  "recommendations": ["string"]
}
```

### Get Treatment Plan
`POST /api/treatment-suggestions`

Generate treatment recommendations.

**Request Body:**
```json
{
  "condition": "string",
  "patient_history": "string",
  "current_medications": ["string"]
}
```

**Response:**
```json
{
  "treatment_plan": {
    "primary_treatments": ["string"],
    "alternatives": ["string"],
    "lifestyle_changes": ["string"],
    "monitoring_plan": "string"
  }
}
```

## Health Management

### Create Appointment
`POST /api/appointments`

Schedule a new medical appointment.

**Request Body:**
```json
{
  "patient_id": "integer",
  "doctor_id": "integer",
  "datetime": "string (ISO 8601)",
  "notes": "string"
}
```

**Response:**
```json
{
  "appointment_id": "integer",
  "status": "string",
  "details": {
    "doctor": "string",
    "patient": "string",
    "datetime": "string",
    "location": "string"
  }
}
```

### Get Medical History
`GET /api/medical-history/{patient_id}`

Retrieve patient medical history.

**Response:**
```json
{
  "patient": {
    "id": "integer",
    "name": "string"
  },
  "history": {
    "conditions": ["string"],
    "medications": ["string"],
    "allergies": ["string"],
    "procedures": ["string"]
  }
}
```

## Health Insights

### Generate Health Report
`POST /api/health-insights`

Generate personalized health insights.

**Request Body:**
```json
{
  "patient_data": {
    "metrics": {
      "vital_signs": {
        "blood_pressure": "string",
        "heart_rate": "integer",
        "temperature": "float"
      },
      "lab_results": {}
    },
    "history": "string"
  }
}
```

**Response:**
```json
{
  "insights": {
    "status": "string",
    "risk_factors": ["string"],
    "recommendations": ["string"],
    "monitoring_needs": ["string"]
  }
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "error": "string",
  "message": "string",
  "details": {}
}
```

### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Authentication required"
}
```

### 403 Forbidden
```json
{
  "error": "Forbidden",
  "message": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
  "error": "Not Found",
  "message": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal Server Error",
  "message": "An unexpected error occurred"
}
```

## Rate Limiting

- Rate limit: 100 requests per minute
- Rate limit headers included in responses:
  - X-RateLimit-Limit
  - X-RateLimit-Remaining
  - X-RateLimit-Reset

## Versioning

API version is specified in the URL path:
- Current version: `/api/v1/`
- Legacy support: `/api/v0/` (deprecated)

## Authentication

All API endpoints except `/login` and `/register` require authentication:

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```