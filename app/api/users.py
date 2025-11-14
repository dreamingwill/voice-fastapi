import io
import json
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile, status
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..auth import get_current_user, require_admin
from ..database import get_db
from ..models import User
from ..schemas import (
    TokenPayload,
    UserCreateAndUpdate,
    UserResponse,
    UserStatusUpdate,
    UsersListResponse,
)
from ..services.events import record_event_log

router = APIRouter(prefix="/api/users", tags=["users"])


def _normalize_status(value: Optional[str]) -> str:
    return "disabled" if (value or "").lower() == "disabled" else "enabled"


def _clean_phone(value: Optional[str]) -> Optional[str]:
    phone = (value or "").strip()
    return phone or None


def _user_to_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        account=user.account or "",
        identity=user.identity,
        phone=user.phone,
        status=user.status or "enabled",
        has_voiceprint=bool(user.embedding),
    )


@router.get("", response_model=UsersListResponse)
async def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: Optional[str] = Query(None, description="Search by username"),
    db: Session = Depends(get_db),
    current_user: TokenPayload = Depends(get_current_user),
):
    query = db.query(User)
    if keyword:
        like_value = f"%{keyword}%"
        query = query.filter(or_(User.username.ilike(like_value), User.account.ilike(like_value)))
    total = query.with_entities(func.count(User.id)).scalar() or 0
    items = (
        query.order_by(User.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    data = [_user_to_response(u) for u in items]
    return UsersListResponse(items=data, total=total, page=page, page_size=page_size)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: TokenPayload = Depends(get_current_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.status == "disabled":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User is disabled")

    return _user_to_response(user)


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreateAndUpdate,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    account = (payload.account or "").strip()
    if not account:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Account is required")
    existing_account = db.query(User).filter(User.account == account).first()
    if existing_account:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Account already exists")

    status_value = _normalize_status(payload.status)
    phone_value = _clean_phone(payload.phone)
    user = User(
        username=payload.username,
        identity=payload.identity,
        account=account,
        phone=phone_value,
        status=status_value,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    record_event_log(
        session_id=None,
        user_id=user.id,
        username=user.username,
        operator=current_admin.display_name or current_admin.username,
        event_type="operator_change",
        category="create",
        authorized=True,
        payload={
            "identity": user.identity,
            "account": user.account,
            "status": user.status,
            "phone": user.phone,
        },
    )
    return _user_to_response(user)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user_by_id(
    user_id: int,
    payload: UserCreateAndUpdate,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    changes: Dict[str, Any] = {}

    if payload.username and payload.username != user.username:
        exists = db.query(User).filter(User.username == payload.username).first()
        if exists:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
        changes["username"] = {"from": user.username, "to": payload.username}
        user.username = payload.username

    if payload.account and payload.account != user.account:
        exists_account = db.query(User).filter(User.account == payload.account).first()
        if exists_account:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Account already exists")
        changes["account"] = {"from": user.account, "to": payload.account}
        user.account = payload.account

    if payload.identity is not None:
        if payload.identity != user.identity:
            changes["identity"] = {"from": user.identity, "to": payload.identity}
            user.identity = payload.identity

    if payload.phone is not None:
        phone_value = _clean_phone(payload.phone)
        if phone_value != user.phone:
            changes["phone"] = {"from": user.phone, "to": phone_value}
            user.phone = phone_value

    if payload.status is not None:
        status_value = _normalize_status(payload.status)
        if status_value != user.status:
            changes["status"] = {"from": user.status, "to": status_value}
            user.status = status_value

    db.commit()
    db.refresh(user)

    if changes:
        record_event_log(
            session_id=None,
            user_id=user.id,
            username=user.username,
            operator=current_admin.display_name or current_admin.username,
            event_type="operator_change",
            category="update",
            authorized=True,
            payload={"changes": changes},
        )

    return _user_to_response(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    payload = {
        "username": user.username,
        "identity": user.identity,
        "account": user.account,
        "phone": user.phone,
        "status": user.status,
    }
    db.delete(user)
    db.commit()
    record_event_log(
        session_id=None,
        user_id=user_id,
        username=payload["username"],
        operator=current_admin.display_name or current_admin.username,
        event_type="operator_change",
        category="delete",
        authorized=True,
        payload=payload,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/by-username/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_by_username(
    username: str,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    payload = {
        "username": user.username,
        "identity": user.identity,
        "account": user.account,
        "phone": user.phone,
        "status": user.status,
    }
    db.delete(user)
    db.commit()
    record_event_log(
        session_id=None,
        user_id=user.id,
        username=user.username,
        operator=current_admin.display_name or current_admin.username,
        event_type="operator_change",
        category="delete",
        authorized=True,
        payload=payload,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{user_id}/status", response_model=UserResponse)
async def update_user_status(
    user_id: int,
    payload: UserStatusUpdate,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_status = _normalize_status(payload.status)
    if user.status != new_status:
        user.status = new_status
        db.commit()
        db.refresh(user)
        record_event_log(
            session_id=None,
            user_id=user.id,
            username=user.username,
            operator=current_admin.display_name or current_admin.username,
            event_type="operator_change",
            category="status_change",
            authorized=True,
            payload={"status": new_status},
        )

    return _user_to_response(user)


@router.post("/{user_id}/voiceprint/aggregate", response_model=UserResponse)
async def aggregate_voiceprint(
    user_id: int,
    files: List[UploadFile],
    request: Request,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if len(files) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 1 sound file to aggregate voiceprint",
        )

    embedder = request.app.state.embedder
    ans = None
    for f in files:
        raw = await f.read()
        if not raw:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File {f.filename} is NULL")

        try:
            with io.BytesIO(raw) as bio:
                data, sample_rate = sf.read(bio, always_2d=True, dtype="float32")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to decode {f.filename}: {e}") from e

        if sample_rate != embedder.sample_rate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sample rate mismatch for {f.filename}: expected {embedder.sample_rate}, got {sample_rate}",
            )

        data = data[:, 0]
        samples = np.ascontiguousarray(data, dtype=np.float32)
        embedding = embedder.embed(samples, sample_rate)

        if ans is None:
            ans = embedding
        else:
            ans += embedding

    if ans is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to compute embeddings")

    ans = ans / len(files)
    user.embedding = json.dumps(ans.tolist())

    db.commit()
    db.refresh(user)

    record_event_log(
        session_id=None,
        user_id=user.id,
        username=user.username,
        operator=current_admin.display_name or current_admin.username,
        event_type="operator_change",
        category="voiceprint_aggregate",
        authorized=True,
        payload={"file_count": len(files)},
    )

    return _user_to_response(user)
