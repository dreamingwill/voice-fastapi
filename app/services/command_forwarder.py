import logging
from datetime import datetime
from typing import Optional

import httpx

from ..config import COMMAND_FORWARD_TIMEOUT, COMMAND_FORWARD_URL

logger = logging.getLogger("command.forwarder")


async def forward_command_match(
    code: str,
    speaker: Optional[str] = None,
    *,
    created_at: Optional[datetime] = None,
) -> Optional[str]:
    """
    Send a POST notification when a command has been recognized.
    """
    if not COMMAND_FORWARD_URL or not code:
        return "missing_forward_url_or_code"
    timestamp = (created_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "createTime": timestamp,
        "projectCode": code,
        "speaker": speaker or "",
    }

    try:
        async with httpx.AsyncClient(timeout=COMMAND_FORWARD_TIMEOUT) as client:
            response = await client.post(COMMAND_FORWARD_URL, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - defensive logging
        status_code = exc.response.status_code if exc.response else "unknown"
        body_snippet = ""
        if exc.response is not None:
            try:
                body_snippet = (exc.response.text or "")[:200]
            except Exception:
                body_snippet = ""
        logger.warning(
            "command.forward request failed url=%s code=%s status=%s body=%s",
            COMMAND_FORWARD_URL,
            code,
            status_code,
            body_snippet,
        )
        return f"downstream_status={status_code}"
    except httpx.RequestError as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "command.forward request failed url=%s code=%s network_error=%s",
            COMMAND_FORWARD_URL,
            code,
            exc.__class__.__name__,
        )
        return f"network_error={exc.__class__.__name__}"
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("command.forward request failed url=%s code=%s error=%s", COMMAND_FORWARD_URL, code, exc)
        return f"unexpected_error={exc.__class__.__name__}"

    logger.info("command.forward sent url=%s code=%s speaker=%s", COMMAND_FORWARD_URL, code, speaker or "")
    return None


def _resolve_timestamp(*, created_at: Optional[datetime] = None, create_time: Optional[str] = None) -> str:
    if create_time:
        return create_time
    return (created_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


async def forward_command_manual(
    code: str,
    speaker: str,
    *,
    create_time: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> str:
    """
    Send a POST notification for manual command forwarding.
    """
    if not COMMAND_FORWARD_URL or not code:
        raise ValueError("Command forward URL or code is missing")
    timestamp = _resolve_timestamp(created_at=created_at, create_time=create_time)
    payload = {
        "createTime": timestamp,
        "projectCode": code,
        "speaker": speaker or "",
    }

    async with httpx.AsyncClient(timeout=COMMAND_FORWARD_TIMEOUT) as client:
        response = await client.post(COMMAND_FORWARD_URL, json=payload)
        response.raise_for_status()

    logger.info("command.forward manual sent url=%s code=%s speaker=%s", COMMAND_FORWARD_URL, code, speaker or "")
    return timestamp


async def check_forward_target() -> dict:
    """Check if the command forward target is reachable."""
    if not COMMAND_FORWARD_URL:
        return {"configured": False, "reachable": False}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.head(COMMAND_FORWARD_URL)
            # 501 = HEAD not implemented, server is up; treat as reachable
            reachable = response.status_code < 500 or response.status_code == 501
            return {"configured": True, "reachable": reachable, "status_code": response.status_code}
    except Exception:
        return {"configured": True, "reachable": False}


__all__ = ["forward_command_match", "forward_command_manual", "check_forward_target"]
