from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import TokenPayload, require_admin
from ..schemas import (
    CommandListResponse,
    CommandToggleRequest,
    CommandUploadRequest,
)
from ..services.commands import get_command_service

router = APIRouter(prefix="/api/commands", tags=["commands"])


@router.get("", response_model=CommandListResponse, status_code=status.HTTP_200_OK)
async def get_commands(user: TokenPayload = Depends(require_admin)):
    service = get_command_service()
    listing = service.list_commands(user.id)
    return CommandListResponse(**listing)


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_commands(payload: CommandUploadRequest, user: TokenPayload = Depends(require_admin)):
    service = get_command_service()
    try:
        inserted = service.upload_commands(user.id, payload.commands)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"inserted": inserted}


@router.post("/toggle", response_model=CommandListResponse, status_code=status.HTTP_200_OK)
async def toggle_command_matching(
    payload: CommandToggleRequest,
    user: TokenPayload = Depends(require_admin),
):
    service = get_command_service()
    service.update_matching_state(
        user.id,
        enabled=payload.enabled,
        match_threshold=payload.match_threshold,
    )
    listing = service.list_commands(user.id)
    return CommandListResponse(**listing)


__all__ = ["router"]
