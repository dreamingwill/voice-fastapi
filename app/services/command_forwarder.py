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
) -> None:
    """
    Send a POST notification when a command has been recognized.
    """
    if not COMMAND_FORWARD_URL or not code:
        return
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
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("command.forward request failed url=%s code=%s error=%s", COMMAND_FORWARD_URL, code, exc)
        return

    logger.info("command.forward sent url=%s code=%s speaker=%s", COMMAND_FORWARD_URL, code, speaker or "")


__all__ = ["forward_command_match"]
