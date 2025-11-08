import argparse
import asyncio
import json
import queue
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import sounddevice as sd
import websockets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream microphone audio to /ws/asr and print transcripts.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/asr", help="WebSocket endpoint")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate (Hz)")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk size in milliseconds")
    parser.add_argument("--device", type=str, default=None, help="Optional sounddevice input name/index")
    parser.add_argument("--operator", type=str, default="cli-tester", help="Operator label for logging")
    parser.add_argument("--token", type=str, default=None, help="Optional auth token forwarded to backend")
    return parser


class MicStreamer:
    """Capture audio from the default (or chosen) microphone and fan it into a Queue."""

    def __init__(self, sample_rate: int, chunk_ms: int, device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.blocksize = int(sample_rate * chunk_ms / 1000)
        self.device = device
        self.queue: "queue.Queue[bytes]" = queue.Queue(maxsize=50)
        self._stream: Optional[sd.RawInputStream] = None

    def __enter__(self):
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        # unblock queue consumers
        self.queue.put_nowait(b"")

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] status warning: {status}", file=sys.stderr)
        self.queue.put(bytes(indata))


async def send_audio(ws, mic: MicStreamer, stop_event: asyncio.Event):
    loop = asyncio.get_running_loop()
    while not stop_event.is_set():
        chunk = await loop.run_in_executor(None, mic.queue.get)
        if not chunk:
            break
        await ws.send(chunk)
    await ws.send(json.dumps({"type": "audio.stop", "data": {}}))


async def recv_messages(ws, stop_event: asyncio.Event):
    async for raw in ws:
        msg: Dict[str, Any]
        if isinstance(raw, bytes):
            print("[client] received unexpected binary payload, ignoring", file=sys.stderr)
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[client] non-JSON message: {raw}")
            continue
        mtype = msg.get("type")
        if mtype == "partial":
            print(f"\r[partial] {msg.get('text','')}", end="", flush=True)
        elif mtype == "final":
            start = msg.get("start_ms")
            end = msg.get("end_ms")
            print(
                f"\n[final] {msg.get('text','')} "
                f"(speaker={msg.get('speaker')} start={start}ms end={end}ms similarity={msg.get('similarity')})"
            )
        elif mtype == "speaker":
            data = msg.get("data", {})
            print(
                f"\n[speaker] id={data.get('id')} name={data.get('name')} confidence={data.get('confidence')}"
            )
        elif mtype == "meta":
            print(f"\n[meta] {msg.get('data')}")
        elif mtype == "error":
            print(f"\n[error] {msg.get('message') or msg.get('msg')}", file=sys.stderr)
        elif mtype == "done":
            print("\n[server] done")
            stop_event.set()
            break
        else:
            print(f"\n[{mtype}] {msg}")


async def ws_session(args):
    async with websockets.connect(args.url) as ws:
        handshake = {
            "type": "audio.start",
            "data": {
                "sampleRate": args.sample_rate,
                "channels": 1,
                "format": "PCM16",
                "operator": {"name": args.operator},
                "token": args.token,
            },
        }
        await ws.send(json.dumps(handshake))

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        with MicStreamer(args.sample_rate, args.chunk_ms, args.device) as mic:
            sender = asyncio.create_task(send_audio(ws, mic, stop_event))
            receiver = asyncio.create_task(recv_messages(ws, stop_event))

            def handle_stop(*_):
                if not stop_event.is_set():
                    print("\n[client] stopping...")
                    stop_event.set()

            loop.add_signal_handler(signal.SIGINT, handle_stop)
            loop.add_signal_handler(signal.SIGTERM, handle_stop)

            await stop_event.wait()
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(ws_session(args))
    except KeyboardInterrupt:
        print("\n[client] interrupted by user")
    except Exception as exc:
        print(f"[client] fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
