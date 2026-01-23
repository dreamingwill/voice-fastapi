import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials

from ..auth import TokenPayload, require_admin, security, validate_access_token
from ..config import COMMAND_FORWARD_URL
from ..database import SessionLocal
from ..models import Command, User
from ..schemas import (
    CommandListResponse,
    CommandSearchResponse,
    CommandToggleRequest,
    CommandUploadRequest,
    CommandUpdateRequest,
    CommandItem,
    CommandStatusUpdateRequest,
    CommandForwardRequest,
    CommandForwardResponse,
)
from ..services.command_forwarder import forward_command_manual
from ..services.commands import CommandCreatePayload, get_command_service, GLOBAL_USER_ID

router = APIRouter(prefix="/api/commands", tags=["commands"])
logger = logging.getLogger("api.commands")


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
    entries = [CommandCreatePayload(text=item.text, code=item.code) for item in payload.commands]
    try:
        inserted = service.upload_commands(user.id, entries)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"inserted": inserted}


@router.post("/toggle", response_model=CommandListResponse, status_code=status.HTTP_200_OK)
async def toggle_command_matching(
    payload: CommandToggleRequest,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    # Determine acting user id: prefer authenticated admin; otherwise fall back to builtin admin
    acting_user_id: int
    if credentials and credentials.credentials:
        try:
            user = await validate_access_token(credentials.credentials)
            acting_user_id = int(user.id)
        except HTTPException:
            acting_user_id = _get_builtin_admin_user_id()
    else:
        acting_user_id = _get_builtin_admin_user_id()

    service = get_command_service()
    service.update_matching_state(
        acting_user_id,
        enabled=payload.enabled,
        match_threshold=payload.match_threshold,
    )
    listing = service.list_commands(acting_user_id)
    return CommandListResponse(**listing)


@router.post("/forward", response_model=CommandForwardResponse, status_code=status.HTTP_200_OK)
async def forward_command(payload: CommandForwardRequest):
    if not COMMAND_FORWARD_URL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Command forward URL not configured",
        )

    code = payload.project_code.strip()
    operator_name = payload.operator_name.strip()
    operator_account = payload.operator_account.strip()
    if not code or not operator_name or not operator_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="projectCode, operatorAccount, and operatorName are required",
        )

    with SessionLocal() as db:
        command = (
            db.query(Command)
            .filter(Command.user_id == GLOBAL_USER_ID, Command.code == code)
            .first()
        )
        if command is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="没有该指令",
            )
        operator = (
            db.query(User)
            .filter(
                User.account == operator_account,
                User.username == operator_name,
                User.status != "disabled",
            )
            .first()
        )
        if operator is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="没有该操作员",
            )

    speaker = operator_name
    try:
        forwarded_at = await forward_command_manual(
            code=code,
            speaker=speaker,
            create_time=payload.create_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else "unknown"
        body_snippet = ""
        if exc.response is not None:
            try:
                body_snippet = (exc.response.text or "")[:200]
            except Exception:
                body_snippet = ""
        logger.warning(
            "command.forward downstream status=%s url=%s body=%s",
            status_code,
            COMMAND_FORWARD_URL,
            body_snippet,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Command forward request failed: downstream_status={status_code}",
        ) from exc
    except httpx.RequestError as exc:
        logger.warning(
            "command.forward network error=%s url=%s",
            exc.__class__.__name__,
            COMMAND_FORWARD_URL,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Command forward request failed: network_error={exc.__class__.__name__}",
        ) from exc

    return CommandForwardResponse(
        sent=True,
        project_code=code,
        speaker=speaker,
        forwarded_at=forwarded_at,
    )


def _get_builtin_admin_user_id() -> int:
    """Return the builtin admin's user id for unauthenticated toggles.

    Falls back to the first admin account if the configured one is not found.
    Raises 500 if no admin account exists at all.
    """
    from ..config import ADMIN_USERNAME
    from ..database import SessionLocal
    from ..models import AdminAccount

    with SessionLocal() as db:
        account = (
            db.query(AdminAccount)
            .filter(AdminAccount.username == ADMIN_USERNAME)
            .first()
        )
        if account is None:
            account = db.query(AdminAccount).order_by(AdminAccount.id.asc()).first()
        if account is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No admin account configured")
        return int(account.id)


@router.get("/search", response_model=CommandSearchResponse, status_code=status.HTTP_200_OK)
async def search_commands(
    q: str | None = Query(None, min_length=1),
    code: str | None = Query(None, min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    user: TokenPayload = Depends(require_admin),
):
    service = get_command_service()
    payload = service.search_commands(user.id, q, code=code, page=page, page_size=page_size)
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
    update_code = "code" in payload.model_fields_set
    try:
        updated = service.update_command(
            user.id,
            command_id,
            payload.text,
            code=payload.code,
            update_code=update_code,
        )
    except ValueError as exc:
        msg = str(exc)
        if msg == "Command not found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from exc
    return CommandItem(**updated)


__all__ = ["router"]
