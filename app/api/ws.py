import random

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.events import record_event_log
from ..services.voice import AsrSession
from ..utils import now_utc

router = APIRouter()


@router.websocket("/ws/asr")
async def ws_identify(websocket: WebSocket):
    session_id = f"sess-{int(now_utc().timestamp() * 1000)}-{random.randint(1000, 9999)}"
    await websocket.accept()
    websocket.scope["session_id"] = session_id

    metrics = getattr(websocket.app.state, "session_metrics", None)
    if metrics is not None:
        metrics["active_sessions"] = int(metrics.get("active_sessions", 0)) + 1

    record_event_log(
        session_id=session_id,
        user_id=None,
        username=None,
        operator=None,
        event_type="session",
        category="open",
        authorized=True,
        payload={"session_id": session_id},
    )

    session = AsrSession(websocket, websocket.app)

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if msg.get("bytes") is not None:
                await session.handle_binary_audio(msg["bytes"])
                continue

            txt = msg.get("text")
            if txt is not None and txt.strip().upper() == "DONE":
                await session.handle_done()
                await websocket.send_json({"type": "done"})
                await websocket.close()
                return

    except WebSocketDisconnect:
        record_event_log(
            session_id=session_id,
            user_id=None,
            username=None,
            operator=None,
            event_type="session",
            category="disconnect",
            authorized=True,
            payload={"reason": "client_disconnected"},
        )
        return
    except Exception as e:
        record_event_log(
            session_id=session_id,
            user_id=None,
            username=None,
            operator=None,
            event_type="session",
            category="error",
            authorized=False,
            payload={"error": str(e)},
        )
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass
        raise
    finally:
        if metrics is not None:
            metrics["active_sessions"] = max(0, int(metrics.get("active_sessions", 1)) - 1)
        record_event_log(
            session_id=session_id,
            user_id=None,
            username=None,
            operator=None,
            event_type="session",
            category="close",
            authorized=True,
            payload={"session_id": session_id},
        )
