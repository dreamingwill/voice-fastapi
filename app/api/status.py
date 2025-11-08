from argparse import Namespace
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text

from ..auth import require_admin
from ..database import engine
from ..schemas import HealthResponse, MetricsResponse, TokenPayload
from ..utils import now_utc

router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    start_time: Optional[datetime] = getattr(request.app.state, "start_time", None)
    uptime = int((now_utc() - start_time).total_seconds()) if start_time else 0
    asr_ready = getattr(request.app.state, "recognizer", None) is not None
    speaker_ready = getattr(request.app.state, "embedder", None) is not None
    db_status = "ok"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    args = getattr(request.app.state, "args", Namespace())
    asr_model = getattr(args, "encoder", None)
    speaker_model = getattr(args, "model_path", None)

    return HealthResponse(
        status="ok" if all([asr_ready, speaker_ready, db_status == "ok"]) else "degraded",
        uptime_seconds=uptime,
        asr_model=asr_model,
        speaker_model=speaker_model,
        asr_ready=asr_ready,
        speaker_ready=speaker_ready,
        database=db_status,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(
    request: Request,
    _: TokenPayload = Depends(require_admin),
):
    metrics_state = getattr(request.app.state, "session_metrics", None) or {}
    latency_samples: List[float] = metrics_state.get("latency_samples", [])
    similarity_samples: List[float] = metrics_state.get("similarity_samples", [])
    undetermined = metrics_state.get("undetermined_count", 0)
    audio_depth = metrics_state.get("audio_queue_depth", 0)

    avg_latency = float(sum(latency_samples) / len(latency_samples)) if latency_samples else 0.0
    avg_similarity = float(sum(similarity_samples) / len(similarity_samples)) if similarity_samples else 0.0
    active_sessions = int(metrics_state.get("active_sessions", 0))
    undetermined_rate = float(undetermined / len(similarity_samples)) if similarity_samples else 0.0

    return MetricsResponse(
        active_sessions=active_sessions,
        avg_recognition_latency_ms=avg_latency,
        avg_embedding_similarity=avg_similarity,
        undetermined_speaker_rate=undetermined_rate,
        audio_queue_depth=int(audio_depth),
    )
