import logging
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from ..config import RECORDINGS_DIR, RECORDINGS_MAX_COUNT

logger = logging.getLogger("asr.recordings")


def open_recording(session_id: str, sample_rate: int) -> Tuple[Optional[wave.Wave_write], Optional[str]]:
    try:
        recordings_path = Path(RECORDINGS_DIR)
        recordings_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = recordings_path / f"recording_{session_id}_{timestamp}.wav"
        writer = wave.open(str(filename), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)  # PCM16 = 2 bytes
        writer.setframerate(sample_rate)
        logger.info("recording.open session=%s file=%s", session_id, filename)
        return writer, filename.name
    except Exception as exc:
        logger.warning("recording.open failed session=%s error=%s", session_id, exc)
        return None, None


def close_recording(writer: Optional[wave.Wave_write]) -> None:
    if writer is None:
        return
    try:
        writer.close()
        logger.info("recording.close done")
    except Exception as exc:
        logger.warning("recording.close failed error=%s", exc)


def cleanup_old_recordings(recordings_dir: str = RECORDINGS_DIR, max_count: int = RECORDINGS_MAX_COUNT) -> None:
    try:
        path = Path(recordings_dir)
        if not path.is_dir():
            return
        files = sorted(path.glob("recording_*.wav"), key=lambda f: f.stat().st_mtime)
        excess = len(files) - max_count
        for f in files[:excess]:
            try:
                f.unlink()
                logger.info("recording.cleanup deleted=%s", f.name)
            except Exception as exc:
                logger.warning("recording.cleanup delete_failed file=%s error=%s", f.name, exc)
    except Exception as exc:
        logger.warning("recording.cleanup failed error=%s", exc)
