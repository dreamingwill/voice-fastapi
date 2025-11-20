from typing import Any, Dict, List, Optional
from typing_extensions import Literal

from pydantic import BaseModel, field_validator


class UserCreateAndUpdate(BaseModel):
    username: str
    identity: Optional[str] = None
    account: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    account: str
    identity: Optional[str] = None
    phone: Optional[str] = None
    status: str
    has_voiceprint: bool


class UsersListResponse(BaseModel):
    items: List[UserResponse]
    total: int
    page: int
    page_size: int


class UserStatusUpdate(BaseModel):
    status: Literal["enabled", "disabled"]


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
    speaker_recognition_enabled: bool


class MetricsResponse(BaseModel):
    active_sessions: int
    avg_recognition_latency_ms: float
    avg_embedding_similarity: float
    undetermined_speaker_rate: float
    audio_queue_depth: int


class TranscriptSegmentResponse(BaseModel):
    id: int
    segment_id: int
    speaker_name: Optional[str] = None
    speaker_user_id: Optional[int] = None
    similarity: Optional[float] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    text: str
    topk: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None


class TranscriptResponse(BaseModel):
    id: int
    session_id: str
    user_id: Optional[int] = None
    username: Optional[str] = None
    dominant_speaker: Optional[str] = None
    speakers: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None
    duration_ms: Optional[int] = None
    similarity_avg: Optional[float] = None
    similarity_max: Optional[float] = None
    segments_count: int
    status: str
    locale: Optional[str] = None
    channel: Optional[str] = None
    operator: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None


class TranscriptDetailResponse(TranscriptResponse):
    segments: List[TranscriptSegmentResponse]


class TranscriptsResponse(BaseModel):
    items: List[TranscriptResponse]
    total: int
    page: int
    page_size: int


class CommandUploadItem(BaseModel):
    text: str
    code: Optional[str] = None


class CommandUploadRequest(BaseModel):
    commands: List[CommandUploadItem]

    @field_validator("commands", mode="before")
    @classmethod
    def _normalize_commands(cls, value):  # type: ignore[override]
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("commands must be a list")
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                normalized.append({"text": item})
            elif isinstance(item, CommandUploadItem):
                normalized.append(item.model_dump())
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                raise ValueError("Invalid command entry")
        return normalized


class CommandToggleRequest(BaseModel):
    enabled: bool
    match_threshold: Optional[float] = None


class CommandItem(BaseModel):
    id: int
    text: str
    code: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CommandListResponse(BaseModel):
    enabled: bool
    match_threshold: Optional[float] = None
    items: List[CommandItem]
    total: int
    page: int
    page_size: int
    updated_at: Optional[str] = None


class CommandSearchResponse(BaseModel):
    items: List[CommandItem]
    total: int
    page: int
    page_size: int


class CommandUpdateRequest(BaseModel):
    text: str
    code: Optional[str] = None


class CommandStatusUpdateRequest(BaseModel):
    status: str


class SystemSettingsResponse(BaseModel):
    enable_speaker_recognition: bool


class SystemSettingsUpdate(BaseModel):
    enable_speaker_recognition: bool


__all__ = [
    "UserCreateAndUpdate",
    "UserResponse",
    "UsersListResponse",
    "UserStatusUpdate",
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
    "TranscriptSegmentResponse",
    "TranscriptResponse",
    "TranscriptDetailResponse",
    "TranscriptsResponse",
    "CommandUploadItem",
    "CommandUploadRequest",
    "CommandToggleRequest",
    "CommandItem",
    "CommandListResponse",
    "CommandSearchResponse",
    "CommandUpdateRequest",
    "CommandStatusUpdateRequest",
    "SystemSettingsResponse",
    "SystemSettingsUpdate",
]
