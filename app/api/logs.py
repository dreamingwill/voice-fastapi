from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import EventLog
from ..schemas import LogsResponse, TokenPayload
from ..services.events import serialize_event_log
from ..utils import parse_iso8601

router = APIRouter(prefix="/api", tags=["logs"])


@router.get("/logs", response_model=LogsResponse)
async def get_logs(
    type: Optional[str] = Query(None, description="Filter by log type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    authorized: Optional[bool] = Query(None),
    user_id: Optional[int] = Query(None),
    username: Optional[str] = Query(None),
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    include_text: bool = Query(False, description="Include raw transcript content"),
    current_user: TokenPayload = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(EventLog)

    if type:
        query = query.filter(EventLog.type == type)
    if category:
        query = query.filter(EventLog.category == category)
    if authorized is not None:
        query = query.filter(EventLog.authorized == authorized)
    if user_id is not None:
        query = query.filter(EventLog.user_id == user_id)
    if username is not None:
        query = query.filter(EventLog.username == username)

    from_dt = parse_iso8601(from_ts)
    to_dt = parse_iso8601(to_ts)
    if from_dt:
        query = query.filter(EventLog.timestamp >= from_dt)
    if to_dt:
        query = query.filter(EventLog.timestamp <= to_dt)

    total = query.with_entities(func.count(EventLog.id)).scalar() or 0
    items = (
        query.order_by(EventLog.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    data = [serialize_event_log(item, redact_sensitive=not include_text) for item in items]
    return LogsResponse(items=data, total=total, page=page, page_size=page_size)
