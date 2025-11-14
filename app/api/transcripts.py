import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Transcript, TranscriptSegment
from ..schemas import (
    TranscriptDetailResponse,
    TranscriptResponse,
    TranscriptSegmentResponse,
    TranscriptsResponse,
    TokenPayload,
)
from ..utils import now_utc, parse_iso8601, to_iso

router = APIRouter(prefix="/api", tags=["transcripts"])


@router.get("/transcripts", response_model=TranscriptsResponse)
async def list_transcripts(
    session_id: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    username: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    speaker: Optional[str] = Query(None, description="Filter by dominant speaker"),
    search: Optional[str] = Query(None, description="Full-text search against transcript text"),
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Transcript)

    if session_id:
        query = query.filter(Transcript.session_id == session_id)
    if user_id is not None:
        query = query.filter(Transcript.user_id == user_id)
    if username:
        query = query.filter(Transcript.username == username)
    if status:
        query = query.filter(Transcript.status == status)
    if speaker:
        query = query.filter(Transcript.dominant_speaker == speaker)
    if search:
        like_pattern = f"%{search.lower()}%"
        query = query.filter(func.lower(Transcript.text).like(like_pattern))

    from_dt = parse_iso8601(from_ts)
    to_dt = parse_iso8601(to_ts)
    if from_dt:
        query = query.filter(Transcript.created_at >= from_dt)
    if to_dt:
        query = query.filter(Transcript.created_at <= to_dt)

    total = query.with_entities(func.count(Transcript.id)).scalar() or 0
    items = (
        query.order_by(Transcript.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    payload = [_serialize_transcript(item) for item in items]
    return TranscriptsResponse(items=payload, total=total, page=page, page_size=page_size)


@router.get("/transcripts/{session_id}", response_model=TranscriptDetailResponse)
async def get_transcript(
    session_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    transcript = (
        db.query(Transcript).filter(Transcript.session_id == session_id).first()
    )
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.transcript_id == transcript.id)
        .order_by(TranscriptSegment.segment_id.asc())
        .all()
    )
    data = _serialize_transcript(transcript)
    data["segments"] = [_serialize_segment(segment) for segment in segments]
    return TranscriptDetailResponse(**data)


def _serialize_transcript(model: Transcript) -> Dict[str, Any]:
    created_at = model.created_at or now_utc()
    updated_at = model.updated_at
    speakers = _loads_json(model.speakers)
    return {
        "id": model.id,
        "session_id": model.session_id,
        "user_id": model.user_id,
        "username": model.username,
        "dominant_speaker": model.dominant_speaker,
        "speakers": speakers,
        "text": model.text,
        "duration_ms": model.duration_ms,
        "similarity_avg": model.similarity_avg,
        "similarity_max": model.similarity_max,
        "segments_count": model.segments_count or 0,
        "status": model.status,
        "locale": model.locale,
        "channel": model.channel,
        "operator": model.operator,
        "created_at": to_iso(created_at),
        "updated_at": to_iso(updated_at) if updated_at else None,
    }


def _serialize_segment(model: TranscriptSegment) -> Dict[str, Any]:
    created_at = model.created_at or now_utc()
    return {
        "id": model.id,
        "segment_id": model.segment_id,
        "speaker_name": model.speaker_name,
        "speaker_user_id": model.speaker_user_id,
        "similarity": model.similarity,
        "start_ms": model.start_ms,
        "end_ms": model.end_ms,
        "text": model.text,
        "topk": _loads_json(model.topk),
        "created_at": to_iso(created_at),
    }


def _loads_json(raw: Optional[str]) -> Optional[Any]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


__all__ = ["list_transcripts", "get_transcript"]
