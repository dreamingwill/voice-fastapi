from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UserCreateAndUpdate(BaseModel):
    username: str
    identity: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    identity: Optional[str] = None
    has_voiceprint: bool


class IdentifyResponse(BaseModel):
    matched: str
    similarity: float
    topk: list
    threshold: float


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenPayload(BaseModel):
    id: int
    username: str
    role: str
    display_name: Optional[str] = None


class TokenResponse(BaseModel):
    token: str
    expires_in: int
    user: TokenPayload
    refresh_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    token: str
    expires_in: int
    refresh_token: Optional[str] = None


class LogEntryResponse(BaseModel):
    id: int
    timestamp: str
    type: str
    session_id: Optional[str] = None
    operator: Optional[str] = None
    authorized: Optional[bool] = None
    payload: Optional[Dict[str, Any]] = None
    username: Optional[str] = None
    user_id: Optional[int] = None
    category: Optional[str] = None
    redacted: Optional[bool] = None


class LogsResponse(BaseModel):
    items: List[LogEntryResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: int
    asr_model: Optional[str]
    speaker_model: Optional[str]
    asr_ready: bool
    speaker_ready: bool
    database: str


class MetricsResponse(BaseModel):
    active_sessions: int
    avg_recognition_latency_ms: float
    avg_embedding_similarity: float
    undetermined_speaker_rate: float
    audio_queue_depth: int


__all__ = [
    "UserCreateAndUpdate",
    "UserResponse",
    "IdentifyResponse",
    "LoginRequest",
    "TokenPayload",
    "TokenResponse",
    "RefreshRequest",
    "RefreshResponse",
    "LogEntryResponse",
    "LogsResponse",
    "HealthResponse",
    "MetricsResponse",
]
