import json
from typing import Any, Dict, List, Optional

from ..database import SessionLocal
from ..models import Transcript, TranscriptSegment
from ..utils import now_utc


def append_transcript_segment(
    *,
    session_id: str,
    segment_id: int,
    text: str,
    speaker_name: Optional[str],
    speaker_user_id: Optional[int],
    similarity: Optional[float],
    start_ms: Optional[int],
    end_ms: Optional[int],
    topk: Optional[List[Dict[str, Any]]] = None,
    locale: Optional[str] = None,
    channel: Optional[str] = None,
    operator: Optional[str] = None,
    status: str = "in_progress",
) -> None:
    """Upsert transcript + segments for a session-final chunk."""
    if not session_id:
        return
    speaker_label = speaker_name or "unknown"
    with SessionLocal() as db:
        transcript = (
            db.query(Transcript)
            .filter(Transcript.session_id == session_id)
            .first()
        )
        if transcript is None:
            transcript = Transcript(
                session_id=session_id,
                status=status,
                locale=locale,
                channel=channel,
                operator=operator,
                user_id=speaker_user_id,
                username=None if speaker_label == "unknown" else speaker_label,
                created_at=now_utc(),
                updated_at=now_utc(),
            )
            db.add(transcript)
            db.flush()

        segment = TranscriptSegment(
            transcript_id=transcript.id,
            segment_id=segment_id,
            speaker_name=speaker_label,
            speaker_user_id=speaker_user_id,
            similarity=similarity,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            topk=_dumps_json(topk),
        )
        db.add(segment)

        _update_transcript_summary(
            transcript=transcript,
            new_text=text,
            speaker=speaker_label,
            speaker_user_id=speaker_user_id,
            similarity=similarity,
            duration_ms=end_ms,
            locale=locale,
            channel=channel,
            operator=operator,
        )

        db.commit()


def finalize_transcript(
    *,
    session_id: str,
    status: str = "completed",
    overwrite: bool = False,
) -> None:
    if not session_id:
        return
    with SessionLocal() as db:
        transcript = (
            db.query(Transcript).filter(Transcript.session_id == session_id).first()
        )
        if transcript is None:
            return
        if not overwrite and transcript.status == "completed":
            return
        transcript.status = status
        transcript.updated_at = now_utc()
        db.commit()


def _dumps_json(payload: Optional[Any]) -> Optional[str]:
    if payload is None:
        return None
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return None


def _update_transcript_summary(
    *,
    transcript: Transcript,
    new_text: str,
    speaker: str,
    speaker_user_id: Optional[int],
    similarity: Optional[float],
    duration_ms: Optional[int],
    locale: Optional[str],
    channel: Optional[str],
    operator: Optional[str],
) -> None:
    prev_count = int(transcript.segments_count or 0)
    new_count = prev_count + 1
    transcript.segments_count = new_count

    clean_text = (new_text or "").strip()
    if clean_text:
        transcript.text = (
            f"{transcript.text} {clean_text}".strip() if transcript.text else clean_text
        )

    if duration_ms is not None:
        transcript.duration_ms = max(int(transcript.duration_ms or 0), int(duration_ms))

    if similarity is not None:
        prev_total = (transcript.similarity_avg or 0.0) * prev_count
        transcript.similarity_avg = (prev_total + float(similarity)) / new_count
        transcript.similarity_max = max(transcript.similarity_max or 0.0, float(similarity))
    elif transcript.similarity_avg is None:
        transcript.similarity_avg = 0.0

    transcript.user_id = transcript.user_id or speaker_user_id
    if speaker and speaker != "unknown" and not transcript.username:
        transcript.username = speaker

    transcript.locale = transcript.locale or locale
    transcript.channel = transcript.channel or channel
    transcript.operator = transcript.operator or operator
    transcript.updated_at = now_utc()

    _update_speakers_field(transcript, speaker, similarity)


def _update_speakers_field(
    transcript: Transcript, speaker: str, similarity: Optional[float]
) -> None:
    payload: List[Dict[str, Any]] = []
    if transcript.speakers:
        try:
            payload = json.loads(transcript.speakers)
        except json.JSONDecodeError:
            payload = []

    speakers_map: Dict[str, Dict[str, Any]] = {
        entry.get("name") or "unknown": dict(entry) for entry in payload if isinstance(entry, dict)
    }
    key = speaker or "unknown"
    entry = speakers_map.get(key, {"name": key, "segments": 0})
    entry["segments"] = int(entry.get("segments", 0)) + 1
    if similarity is not None:
        current = float(entry.get("similarity_max", 0.0))
        entry["similarity_max"] = max(current, float(similarity))
        entry["similarity_last"] = float(similarity)
    speakers_map[key] = entry
    ordered = list(speakers_map.values())
    transcript.speakers = json.dumps(ordered, ensure_ascii=False)
    dominant = max(ordered, key=lambda item: item.get("segments", 0)) if ordered else None
    transcript.dominant_speaker = (dominant or {}).get("name")


__all__ = ["append_transcript_segment", "finalize_transcript"]
