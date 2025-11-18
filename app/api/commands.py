from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import TokenPayload, require_admin
from ..schemas import (
    CommandListResponse,
    CommandSearchResponse,
    CommandToggleRequest,
    CommandUploadRequest,
    CommandUpdateRequest,
    CommandItem,
    CommandStatusUpdateRequest,
)
from ..services.commands import get_command_service

router = APIRouter(prefix="/api/commands", tags=["commands"])


@router.get("", response_model=CommandListResponse, status_code=status.HTTP_200_OK)
async def get_commands(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    user: TokenPayload = Depends(require_admin),
):
    service = get_command_service()
    listing = service.list_commands(user.id, page=page, page_size=page_size)
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


@router.get("/search", response_model=CommandSearchResponse, status_code=status.HTTP_200_OK)
async def search_commands(
    q: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    user: TokenPayload = Depends(require_admin),
):
    service = get_command_service()
    payload = service.search_commands(user.id, q, page=page, page_size=page_size)
    return CommandSearchResponse(**payload)


@router.delete("/{command_id}", status_code=status.HTTP_200_OK)
async def delete_command(command_id: int, user: TokenPayload = Depends(require_admin)):
    service = get_command_service()
    success = service.delete_command(user.id, command_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Command not found")
    return {"deleted": True}


@router.patch("/{command_id}/status", response_model=CommandItem, status_code=status.HTTP_200_OK)
async def update_command_status(
    command_id: int,
    payload: CommandStatusUpdateRequest,
    user: TokenPayload = Depends(require_admin),
):
    service = get_command_service()
    try:
        updated = service.update_command_status(user.id, command_id, payload.status)
    except ValueError as exc:
        msg = str(exc)
        if msg == "Command not found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from exc
    return CommandItem(**updated)


@router.put("/{command_id}", response_model=CommandItem, status_code=status.HTTP_200_OK)
async def update_command(
    command_id: int,
    payload: CommandUpdateRequest,
    user: TokenPayload = Depends(require_admin),
):
    service = get_command_service()
    try:
        updated = service.update_command(user.id, command_id, payload.text)
    except ValueError as exc:
        msg = str(exc)
        if msg == "Command not found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from exc
    return CommandItem(**updated)


__all__ = ["router"]
