# CodeQuery API Documentation

> **Version**: 1.0.0  
> **Base URL**: `http://localhost:8000/api/v1`  
> **Protocol**: HTTP/1.1 (REST + SSE)  
> **Content-Type**: `application/json`

---

## Table of Contents

1. [Repositories API](#repositories-api)
2. [Chat API](#chat-api)
3. [Data Models](#data-models)
4. [Status Codes](#status-codes)
5. [Examples](#examples)



## Repositories API

### Create Repository

Submit a GitHub URL for indexing.

**Endpoint**: `POST /api/v1/repos`

#### Request

```http
POST /api/v1/repos HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "url": "https://github.com/microsoft/LightGBM"
}
```

**Request Body** (`RepoCreate`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | GitHub repository URL |

#### Success Response (201 Created)

```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "id": "3d6f12e6-5301-4e67-ac61-e9d51a1067e6",
  "url": "https://github.com/microsoft/LightGBM",
  "name": "LightGBM",
  "status": "pending",
  "error_message": null,
  "created_at": "2026-05-16T10:30:00",
  "updated_at": "2026-05-16T10:30:00"
}
```

**Response Fields** (`RepoResponse`):

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | UUID v4, unique identifier |
| `url` | string | Original GitHub URL |
| `name` | string\|null | Repository name (from URL) |
| `status` | string | Current status: `pending`, `cloning`, `parsing`, `embedding`, `ready`, `failed` |
| `error_message` | string\|null | Error description if failed |
| `created_at` | datetime | When indexing started |
| `updated_at` | datetime | Last status update |

### Error Responses
----


**400 - Invalid URL**
```json
{
  "detail": "Invalid GitHub URL format"
}
```

**400 - Missing URL**
```json
{
  "detail": [
    {
      "loc": ["body", "url"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 - Database Error**
```json
{
  "detail": "Failed to create repository record"
}
```

**500 - Worker Error**
```json
{
  "detail": "Failed to start indexing task"
}
```

---

### List Repositories

Get all indexed repositories with their current status.

**Endpoint**: `GET /api/v1/repos`

#### Request

```http
GET /api/v1/repos HTTP/1.1
Host: localhost:8000
```

#### Success Response (200 OK)

```http
HTTP/1.1 200 OK
Content-Type: application/json

[
  {
    "id": "3d6f12e6-5301-4e67-ac61-e9d51a1067e6",
    "url": "https://github.com/microsoft/LightGBM",
    "name": "LightGBM",
    "status": "ready",
    "error_message": null,
    "created_at": "2026-05-16T10:30:00",
    "updated_at": "2026-05-16T10:35:42"
  },
  {
    "id": "8a2b9c4d-1234-5678-90ab-cdef12345678",
    "url": "https://github.com/huggingface/transformers",
    "name": "transformers",
    "status": "embedding",
    "error_message": null,
    "created_at": "2026-05-16T10:40:00",
    "updated_at": "2026-05-16T10:42:15"
  },
  {
    "id": "1f3e5d7c-aaaa-bbbb-cccc-dddd12345678",
    "url": "https://github.com/user/bad-repo",
    "name": "bad-repo",
    "status": "failed",
    "error_message": "Failed to clone repository: Repository not found",
    "created_at": "2026-05-16T10:45:00",
    "updated_at": "2026-05-16T10:45:05"
  }
]
```

#### Error Responses

**500 - Database Error**
```json
{
  "detail": "Failed to fetch repositories"
}
```

---

### Delete Repository

Remove a repository and all its associated data (local clone, Qdrant collection, and database record).

**Endpoint**: `DELETE /api/v1/repos/{repo_id}`

#### Request

```http
DELETE /api/v1/repos/3d6f12e6-5301-4e67-ac61-e9d51a1067e6 HTTP/1.1
Host: localhost:8000
```

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo_id` | string | Yes | Repository UUID |

#### Success Response (200 OK)

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "detail": "Repo deleted"
}
```

#### Error Responses

**404 - Repo Not Found**
```json
{
  "detail": "Repo not found"
}
```

---

## Chat API

### Send Message

Ask a question about an indexed repository. Returns a Server-Sent Events (SSE) stream.

**Endpoint**: `POST /api/v1/chat?repo_id={repo_id}`

#### Request

```http
POST /api/v1/chat?repo_id=3d6f12e6-5301-4e67-ac61-e9d51a1067e6 HTTP/1.1
Host: localhost:8000
Content-Type: application/json
Accept: text/event-stream

{
  "message": "Explain the LGBMRanker class",
  "session_id": null
}
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo_id` | string | Yes | Repository UUID |

**Request Body** (`ChatRequest`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | User's question |
| `session_id` | string\|null | No | Previous thread ID for multi-turn conversations |

#### SSE Response Format

The response is a stream of SSE events. Each event has the format:

```
event: {event_type}
data: {json_payload}

```

**Event Types**:

| Event | Description | Payload |
|-------|-------------|---------|
| `token` | Streaming text chunk | `{"token": "The LGBMRanker "}` |
| `citations` | Source references | `{"citations": [...]}` |
| `error` | Error occurred | `{"detail": "..."}` |
| `done` | Stream complete | `{"content": "...", "thread_id": "..."}` |

#### Success Response (200 OK)

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: token
data: {"token": "The "}

event: token
data: {"token": "LGBMRanker "}

event: token
data: {"token": "class "}

event: token
data: {"token": "is "}

... (more tokens) ...

event: citations
data: {"citations": [{"file": "python-package/lightgbm/sklearn.py", "start_line": 1814, "end_line": 1936}]}

event: done
data: {"content": "The LGBMRanker class is a ranking model...", "thread_id": "5a74e11e-cb84-449d-9ca5-96eb9e240230"}
```

#### Error Responses (SSE Format)

**404 - Repo Not Found**
```
event: error
data: {"detail": "Repo not found"}
```

**400 - Repo Not Ready**
```
event: error
data: {"detail": "Repo is not ready. Status: embedding"}
```

**400 - Rate Limit**
```
event: error
data: {"detail": "Request too large. The context exceeded the model's limit. Try reducing input tokens."}
```

**500 - LLM Error**
```
event: error
data: {"detail": "Error in graph execution: Rate limit exceeded"}
```

**500 - Empty Response**
```
event: error
data: {"detail": "No response generated."}
```

---

## Data Models

### RepoCreate

```python
class RepoCreate(BaseModel):
    url: str  # GitHub repository URL
    
    @field_validator("url", mode="before")
    def prepend_https(cls, v: str) -> str:
        # Auto-prefixes https:// if missing
        if not v.startswith(("http://", "https://")):
            return f"https://{v}"
        return v
```

**Validation Rules**:
- URL must be non-empty
- Auto-prepends `https://` if protocol missing
- Does NOT validate if URL is accessible

### RepoResponse

```python
class RepoResponse(BaseModel):
    id: str                    # UUID v4
    url: str                   # Full GitHub URL
    name: Optional[str]        # Repo name from URL
    status: str                # pending | cloning | parsing | embedding | ready | failed
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
```

### ChatRequest

```python
class ChatRequest(BaseModel):
    message: str                    # User's question
    session_id: Optional[str] = None  # Thread ID for multi-turn
```

**Multi-turn Usage**:
- First message: `session_id: null` → creates new thread
- Follow-up: `session_id: "previous-thread-id"` → continues conversation
- Thread ID returned in `done` event

### Citation

```json
{
  "file": "python-package/lightgbm/sklearn.py",
  "start_line": 1814,
  "end_line": 1936
}
```


### HTTP Status Codes

| Code | Usage |
|------|-------|
| `200` | Success (list repos, chat stream) |
| `201` | Created (new repo) |
| `400` | Bad request, repo not ready |
| `404` | Repo not found |
| `422` | Validation error (Pydantic) |
| `500` | Internal server error |

---



