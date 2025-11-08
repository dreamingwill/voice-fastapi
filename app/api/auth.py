from fastapi import APIRouter, HTTPException, status

from ..auth import authenticate_user, issue_tokens, refresh_access_token
from ..schemas import LoginRequest, RefreshRequest, RefreshResponse, TokenResponse
from ..services.events import record_event_log

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login(payload: LoginRequest):
    user_payload = authenticate_user(payload.username, payload.password)
    if not user_payload:
        record_event_log(
            session_id=None,
            user_id=None,
            username=payload.username,
            operator=None,
            event_type="auth",
            category="login_failed",
            authorized=False,
            payload={"reason": "INVALID_CREDENTIALS"},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")

    token_response = await issue_tokens(user_payload)
    record_event_log(
        session_id=None,
        user_id=user_payload.id,
        username=user_payload.username,
        operator=user_payload.display_name,
        event_type="auth",
        category="login",
        authorized=True,
        payload={"expires_in": token_response.expires_in},
    )
    return token_response


@router.post("/refresh", response_model=RefreshResponse, status_code=status.HTTP_200_OK)
async def refresh_token(payload: RefreshRequest):
    token_response = await refresh_access_token(payload.refresh_token)
    return RefreshResponse(
        token=token_response.token,
        expires_in=token_response.expires_in,
        refresh_token=token_response.refresh_token,
    )
