import io
import json
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status
from sqlalchemy.orm import Session

from ..auth import get_current_user, require_admin
from ..database import get_db
from ..models import User
from ..schemas import TokenPayload, UserCreateAndUpdate, UserResponse
from ..services.events import record_event_log

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=list[UserResponse])
async def get_users(
    db: Session = Depends(get_db),
    current_user: TokenPayload = Depends(get_current_user),
):
    users = db.query(User).all()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            identity=u.identity,
            has_voiceprint=bool(u.embedding),
        )
        for u in users
    ]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: TokenPayload = Depends(get_current_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse(
        id=user.id,
        username=user.username,
        identity=user.identity,
        has_voiceprint=bool(user.embedding),
    )


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreateAndUpdate,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    user = User(username=payload.username, identity=payload.identity)
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
        payload={"identity": user.identity},
    )
    return UserResponse(
        id=user.id,
        username=user.username,
        identity=user.identity,
        has_voiceprint=bool(getattr(user, "embedding", None)),
    )


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
    original_username = user.username
    original_identity = user.identity

    if payload.username and payload.username != user.username:
        exists = db.query(User).filter(User.username == payload.username).first()
        if exists:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
        changes["username"] = {"from": user.username, "to": payload.username}
        user.username = payload.username

    if payload.identity is not None:
        if payload.identity != user.identity:
            changes["identity"] = {"from": user.identity, "to": payload.identity}
            user.identity = payload.identity

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
            payload={"changes": changes, "original_username": original_username, "original_identity": original_identity},
        )

    return UserResponse(
        id=user.id,
        username=user.username,
        identity=user.identity,
        has_voiceprint=bool(user.embedding),
    )


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
    payload = {"username": user.username, "identity": user.identity}
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
    if len(files) != 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Need 3 sound files, but actual {len(files)}",
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

    return UserResponse(
        id=user.id,
        username=user.username,
        identity=user.identity,
        has_voiceprint=True,
    )
