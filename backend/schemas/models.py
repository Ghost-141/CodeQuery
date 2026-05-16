from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class RepoCreate(BaseModel):
    url: str

    @field_validator("url", mode="before")
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


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Reuse thread for multi-turn conversations
