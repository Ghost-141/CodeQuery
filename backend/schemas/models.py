from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class RepoCreate(BaseModel):
    url: str

    @field_validator('url', mode='before')
    @classmethod
    def prepend_https(cls, v: str) -> str:
        if isinstance(v, str) and not v.startswith(("http://", "https://")):
            return f"https://{v}"
        return v


class RepoResponse(BaseModel):
    id: str
    url: str
    name: Optional[str]
    status: str
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    repo_id: str
    title: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    repo_id: str
    title: Optional[str]
    reasoning_enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True


class MessageItem(BaseModel):
    role: str
    content: str
    reasoning_trace: Optional[list[dict[str, Any]]] = None
    citations: Optional[list[dict[str, Any]]] = None


class ChatRequest(BaseModel):
    message: str


class ChatStreamEvent(BaseModel):
    event: str
    data: Any
