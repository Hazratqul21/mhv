# MIYA API Reference

Base URL: `http://localhost:8000`

## Authentication

All endpoints (except `/health`) require a JWT Bearer token in production mode.
In development mode (`MIYA_ENV=development`), authentication is optional.

```
Authorization: Bearer <token>
```

## Endpoints

### Chat

#### POST /api/chat

Send a message and receive an AI response.

**Request:**
```json
{
  "message": "Write a Python function to sort a list",
  "session_id": "sess_abc123",
  "context": {},
  "stream": false
}
```

**Response:**
```json
{
  "response": "Here's a sorting function...",
  "agent_used": "code",
  "tools_used": ["sandbox"],
  "session_id": "sess_abc123",
  "confidence": 0.92,
  "execution_time_ms": 1543
}
```

### WebSocket

#### WS /ws/{session_id}

Real-time bidirectional communication with streaming support.

**Send:**
```json
{"type": "message", "content": "Hello"}
```

**Receive (streaming):**
```json
{"type": "token", "content": "Hello"}
{"type": "token", "content": " there"}
{"type": "done", "agent_used": "chat", "execution_time_ms": 234}
```

### System

#### GET /health

Health check for all services.

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "services": {
    "api": "up",
    "chroma": "up",
    "redis": "up"
  }
}
```

#### GET /api/agents

List available AI agents.

#### GET /api/tools

List available tools with descriptions.

### Sessions

#### GET /api/sessions/{session_id}

Get session information and message count.

#### DELETE /api/sessions/{session_id}

Delete a session and its history.

### Files

#### POST /api/upload

Upload a file (max 10 MB).

**Request:** `multipart/form-data` with `file` field.

### Metrics

#### GET /metrics

Prometheus-compatible metrics endpoint.
