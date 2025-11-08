import json
from typing import Any, Dict, Optional

from ..database import SessionLocal
from ..models import EventLog
from ..schemas import LogEntryResponse
from ..utils import now_utc, to_iso


def record_event_log(
    *,
    session_id: Optional[str],
    user_id: Optional[int],
    username: Optional[str],
    operator: Optional[str],
    event_type: str,
    category: Optional[str],
    authorized: Optional[bool],
    payload: Optional[Dict[str, Any]],
    timestamp=None,
) -> None:
    ts = timestamp or now_utc()
    entry = EventLog(
        session_id=session_id,
        user_id=user_id,
        username=username,
        operator=operator,
        type=event_type,
        category=category,
        authorized=authorized if authorized is not None else True,
        payload=json.dumps(payload, ensure_ascii=False) if payload is not None else None,
        timestamp=ts,
    )
    with SessionLocal() as db:
        db.add(entry)
        db.commit()


def serialize_event_log(log: EventLog, *, redact_sensitive: bool = True) -> LogEntryResponse:
    payload: Optional[Dict[str, Any]] = None
    redacted = False
    if log.payload:
        try:
            payload = json.loads(log.payload)
        except json.JSONDecodeError:
            payload = None
    if (
        redact_sensitive
        and log.type in {"transcript"}
        and isinstance(payload, dict)
        and "text" in payload
    ):
        payload.pop("text", None)
        payload["redacted"] = True
        redacted = True

    timestamp = log.timestamp or now_utc()
    return LogEntryResponse(
        id=log.id,
        timestamp=to_iso(timestamp),
        type=log.type,
        session_id=log.session_id,
        operator=log.operator,
        authorized=log.authorized,
        payload=payload,
        username=log.username,
        user_id=log.user_id,
        category=log.category,
        redacted=redacted,
    )


__all__ = ["record_event_log", "serialize_event_log"]
