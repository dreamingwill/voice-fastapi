import asyncio
import secrets
from datetime import timedelta
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import (
    ACCESS_TOKEN_TTL,
    ADMIN_DISPLAY_NAME,
    ADMIN_PASSWORD,
    ADMIN_ROLE,
    ADMIN_USERNAME,
    REFRESH_TOKEN_TTL,
)
from .schemas import TokenPayload, TokenResponse
from .utils import now_utc


security = HTTPBearer(auto_error=False)
access_tokens: Dict[str, Dict[str, Any]] = {}
refresh_tokens: Dict[str, Dict[str, Any]] = {}
token_lock = asyncio.Lock()


async def issue_tokens(user_payload: TokenPayload) -> TokenResponse:
    access_token = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(40)
    now = now_utc()
    access_exp = now + timedelta(seconds=ACCESS_TOKEN_TTL)
    refresh_exp = now + timedelta(seconds=REFRESH_TOKEN_TTL)

    record = {
        "token": access_token,
        "user": user_payload.model_dump(),
        "expires_at": access_exp,
    }
    refresh_record = {
        "refresh_token": refresh_token,
        "user": user_payload.model_dump(),
        "expires_at": refresh_exp,
    }

    async with token_lock:
        access_tokens[access_token] = record
        refresh_tokens[refresh_token] = refresh_record

    return TokenResponse(
        token=access_token,
        expires_in=ACCESS_TOKEN_TTL,
        user=user_payload,
        refresh_token=refresh_token,
    )


async def refresh_access_token(refresh_token: str) -> TokenResponse:
    now = now_utc()
    async with token_lock:
        record = refresh_tokens.get(refresh_token)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        if record["expires_at"] <= now:
            del refresh_tokens[refresh_token]
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Expired refresh token",
            )

        user_payload = TokenPayload(**record["user"])

    return await issue_tokens(user_payload)


async def validate_access_token(token: str) -> TokenPayload:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing access token",
        )

    now = now_utc()
    async with token_lock:
        record = access_tokens.get(token)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token",
            )
        if record["expires_at"] <= now:
            del access_tokens[token]
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
            )
        return TokenPayload(**record["user"])


def authenticate_user(username: str, password: str) -> Optional[TokenPayload]:
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return TokenPayload(
            id=0,
            username=username,
            role=ADMIN_ROLE,
            display_name=ADMIN_DISPLAY_NAME,
        )
    return None


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> TokenPayload:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )
    return await validate_access_token(credentials.credentials)


async def require_admin(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user


__all__ = [
    "security",
    "issue_tokens",
    "refresh_access_token",
    "validate_access_token",
    "authenticate_user",
    "get_current_user",
    "require_admin",
]
