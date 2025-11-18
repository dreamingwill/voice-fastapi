from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from ..auth import require_admin
from ..database import get_db
from ..schemas import SystemSettingsResponse, SystemSettingsUpdate, TokenPayload
from ..services.events import record_event_log
from ..services.settings import load_system_settings_snapshot, update_system_settings_snapshot

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/system", response_model=SystemSettingsResponse)
async def get_system_settings(
    request: Request,
    db: Session = Depends(get_db),
    _: TokenPayload = Depends(require_admin),
):
    snapshot = load_system_settings_snapshot(db)
    request.app.state.system_settings = snapshot
    return SystemSettingsResponse(enable_speaker_recognition=snapshot.enable_speaker_recognition)


@router.patch("/system", response_model=SystemSettingsResponse)
async def update_system_settings(
    payload: SystemSettingsUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    snapshot = update_system_settings_snapshot(
        db,
        enable_speaker_recognition=payload.enable_speaker_recognition,
    )
    request.app.state.system_settings = snapshot
    record_event_log(
        session_id=None,
        user_id=current_admin.id,
        username=current_admin.username,
        operator=current_admin.display_name or current_admin.username,
        event_type="settings",
        category="update",
        authorized=True,
        payload={"enable_speaker_recognition": snapshot.enable_speaker_recognition},
    )
    return SystemSettingsResponse(enable_speaker_recognition=snapshot.enable_speaker_recognition)
