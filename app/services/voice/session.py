import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import ValidationError

from ..command_forwarder import forward_command_match
from ..events import record_event_log
from ..transcripts import append_transcript_segment, finalize_transcript
from ...auth import validate_access_token
from ..audio_enhancement import AudioEnhancementPipeline, EnhancementConfig
from ..commands import get_command_service
from .recognizer import pcm_bytes_to_float32
from .speaker import SpeakerCandidate, SpeakerEmbedder, identify_user

logger = logging.getLogger("asr.session")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _ms(samples: int, sr: int) -> int:
    return int(samples * 1000 / sr)


class AsrSession:
    def __init__(self, websocket: WebSocket, app: FastAPI):
        self.ws = websocket
        self.app = app
        self.args = app.state.args
        self.recognizer = app.state.recognizer
        self.embedder: SpeakerEmbedder = app.state.embedder
        settings = getattr(app.state, "system_settings", None)
        self.speaker_recognition_enabled = bool(getattr(settings, "enable_speaker_recognition", True))

        self.sample_rate_client = getattr(self.embedder, "sample_rate", None) or self.args.sample_rate or 16000
        self.dtype = "float32"
        self.dtype_hint = "float32"

        self.stream = self.recognizer.create_stream()
        self.total_samples_in = 0
        self.cur_utt_start_sample = 0
        self.cur_utt_audio = []
        self.cur_utt_speaker_guess_sent = False
        self.segment_id = 0
        self.session_started_at = time.perf_counter()
        self.cur_utt_started_at = self.session_started_at
        self.handshake_received = False
        self.session_info: Dict[str, Any] = {}
        self.latest_topk: List[SpeakerCandidate] = []
        self.current_speaker_candidate: Optional[SpeakerCandidate] = None
        self._last_partial_logged_text: Optional[str] = None
        self._last_partial_logged_at = 0.0
        self._partial_log_interval = 0.5  # seconds
        self._last_partial_text_sent: Optional[str] = None
        self.session_id = websocket.scope.get("session_id")
        self._last_speaker_eval_latency_ms: Optional[int] = None
        self.command_service = get_command_service()
        self.command_user_id: Optional[int] = None
        self.command_matching_enabled = False
        self.command_match_threshold: Optional[float] = None
        self.enhancement_pipeline: Optional[AudioEnhancementPipeline] = getattr(app.state, "enhancement_pipeline", None)
        self.enhancement_config = EnhancementConfig()

    def _concat_cur_utt_audio(self) -> np.ndarray:
        if not self.cur_utt_audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.cur_utt_audio, axis=0)

    def _try_speaker(
        self, force: bool = False
    ) -> Tuple[str, float, Optional[SpeakerCandidate], List[SpeakerCandidate]]:
        if not self.speaker_recognition_enabled or self.embedder is None:
            return "unknown", 0.0, None, []
        self._last_speaker_eval_latency_ms = None
        buf = self._concat_cur_utt_audio()
        need_len = int(self.args.min_spk_seconds * self.sample_rate_client)
        if (not force) and (buf.size < need_len):
            return "unknown", 0.0, None, []

        eval_start = time.perf_counter()
        st = self.embedder.create_stream()
        st.accept_waveform(sample_rate=self.sample_rate_client, waveform=buf)
        if force:
            st.input_finished()
        if not self.embedder.is_ready(st):
            return "unknown", 0.0, None, []
        emb = self.embedder.compute(st)
        emb = np.asarray(emb, dtype=np.float32)

        matched, top_sim, topk = identify_user(emb, threshold=self.args.threshold)
        self._last_speaker_eval_latency_ms = int((time.perf_counter() - eval_start) * 1000)

        if force:
            metrics = getattr(self.app.state, "session_metrics", None)
            if metrics is not None:
                sims = metrics.setdefault("similarity_samples", [])
                sims.append(float(top_sim))
                if len(sims) > 200:
                    metrics["similarity_samples"] = sims[-200:]
                if matched is None:
                    metrics["undetermined_count"] = int(metrics.get("undetermined_count", 0)) + 1

        if matched is None:
            return "unknown", float(top_sim), None, topk
        return matched["username"], float(top_sim), matched, topk

    async def _send_speaker_state(self, candidate: SpeakerCandidate, similarity: float):
        latency_ms = self._last_speaker_eval_latency_ms
        logger.info(
            "speaker.update session=%s id=%s username=%s role=%s confidence=%.3f latency_ms=%s",
            self.ws.scope.get("session_id"),
            candidate.get("id"),
            candidate.get("username"),
            candidate.get("identity"),
            similarity,
            latency_ms if latency_ms is not None else "n/a",
        )
        payload = {
            "type": "speaker",
            "data": {
                "id": str(candidate.get("id", "")),
                "name": candidate.get("username") or "未知说话人",
                "role": candidate.get("identity") or "—",
                "confidence": float(similarity),
                "username": candidate.get("username"),
            },
        }
        await self.ws.send_json(payload)

    async def _send_meta(self, data: Dict[str, Any]):
        await self.ws.send_json({"type": "meta", "data": data})

    def _apply_enhancement_preferences(self, payload: Dict[str, Any]) -> None:
        enhancement_payload = payload.get("enhancement") or {}

        def _read(keys, default=None):
            for key in keys:
                if key in enhancement_payload and enhancement_payload[key] is not None:
                    return enhancement_payload[key]
                if key in payload and payload[key] is not None:
                    return payload[key]
            return default

        cfg_data = {
            "noise_mode": (_read(["noiseMode", "noise_mode"], "none") or "none").lower(),
            "noise_strength": _read(["noiseStrength", "noise_strength"], 1.0),
            "enable_dereverb": bool(_read(["enableDereverb", "enable_dereverb"], False)),
            "dereverb_delay": _read(["dereverbDelay", "dereverb_delay"], self.enhancement_config.dereverb_delay),
            "dereverb_taps": _read(["dereverbTaps", "dereverb_taps"], self.enhancement_config.dereverb_taps),
            "dereverb_iterations": _read(
                ["dereverbIterations", "dereverb_iterations"],
                self.enhancement_config.dereverb_iterations,
            ),
        }

        self.enhancement_config = EnhancementConfig(**cfg_data)

    def _get_operator_label(self) -> Optional[str]:
        operator = self.session_info.get("operator")
        if isinstance(operator, dict):
            return operator.get("name") or operator.get("username") or (
                str(operator.get("id")) if operator.get("id") is not None else None
            )
        if isinstance(operator, str):
            cleaned = operator.strip()
            return cleaned or None
        return None

    def _should_log_partial(self, text: str) -> bool:
        if text == self._last_partial_text_sent:
            return False
        now = time.perf_counter()
        if self._last_partial_logged_text != text or (now - self._last_partial_logged_at) >= self._partial_log_interval:
            self._last_partial_logged_text = text
            self._last_partial_logged_at = now
            return True
        return False

    async def _send_partial(self, text: str, speaker: str):
        if text == self._last_partial_text_sent:
            return
        self._last_partial_text_sent = text
        if self._should_log_partial(text):
            logger.info(
                "asr.partial session=%s segment=%s speaker=%s start_ms=%s text=%s",
                self.ws.scope.get("session_id"),
                self.segment_id,
                speaker,
                _ms(self.cur_utt_start_sample, self.sample_rate_client),
                text,
            )
        await self.ws.send_json(
            {
                "type": "partial",
                "segment_id": self.segment_id,
                "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
                "time_ms": _ms(self.total_samples_in, self.sample_rate_client),
                "text": text,
                "speaker": speaker,
            }
        )

    async def _send_final(
        self,
        text: str,
        speaker: str,
        similarity: float,
        topk: List[SpeakerCandidate],
        candidate: Optional[SpeakerCandidate],
    ):
        meta_topk = [
            {"username": item.get("username"), "similarity": item.get("similarity", 0.0)}
            for item in topk
            if item.get("username")
        ]
        latency_ms = int((time.perf_counter() - self.cur_utt_started_at) * 1000)
        command_match = self._evaluate_command_match(text)
        self._maybe_forward_command(command_match, speaker)
        await self.ws.send_json(
            {
                "type": "final",
                "segment_id": self.segment_id,
                "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
                "end_ms": _ms(self.total_samples_in, self.sample_rate_client),
                "text": text,
                "speaker": speaker,
                "similarity": similarity,
                "topk": meta_topk,
                "command_match": command_match,
            }
        )
        logger.info(
            "asr.final session=%s segment=%s speaker=%s similarity=%.3f latency_ms=%s text=%s topk=%s",
            self.ws.scope.get("session_id"),
            self.segment_id,
            speaker,
            similarity,
            latency_ms,
            text,
            meta_topk,
        )

        await self._send_meta({"latency": latency_ms})

        record_event_log(
            session_id=self.ws.scope.get("session_id"),
            user_id=(candidate or {}).get("id"),
            username=speaker if speaker != "unknown" else None,
            operator=speaker if speaker != "unknown" else None,
            event_type="transcript",
            category="final",
            authorized=speaker != "unknown",
            payload={
                "text": text,
                "segment_id": self.segment_id,
                "similarity": similarity,
                "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
                "end_ms": _ms(self.total_samples_in, self.sample_rate_client),
                "topk": meta_topk,
                "command_match": command_match,
            },
        )

        self._persist_transcript_segment(
            text=text,
            speaker=speaker,
            similarity=similarity,
            start_ms=_ms(self.cur_utt_start_sample, self.sample_rate_client),
            end_ms=_ms(self.total_samples_in, self.sample_rate_client),
            topk=topk,
            candidate=candidate,
        )

        metrics = getattr(self.app.state, "session_metrics", None)
        if metrics is not None:
            latency_samples = metrics.setdefault("latency_samples", [])
            latency_samples.append(latency_ms)
            if len(latency_samples) > 200:
                metrics["latency_samples"] = latency_samples[-200:]

        self.segment_id += 1
        self.cur_utt_start_sample = self.total_samples_in
        self.cur_utt_audio.clear()
        self.cur_utt_speaker_guess_sent = False
        self.cur_utt_started_at = time.perf_counter()
        self.latest_topk = []
        self.current_speaker_candidate = None

    def _persist_transcript_segment(
        self,
        *,
        text: str,
        speaker: str,
        similarity: float,
        start_ms: int,
        end_ms: int,
        topk: List[SpeakerCandidate],
        candidate: Optional[SpeakerCandidate],
    ) -> None:
        if not self.session_id or not text:
            return

        append_transcript_segment(
            session_id=self.session_id,
            segment_id=self.segment_id,
            text=text,
            speaker_name=speaker,
            speaker_user_id=(candidate or {}).get("id"),
            similarity=similarity,
            start_ms=start_ms,
            end_ms=end_ms,
            topk=topk,
            locale=self.session_info.get("locale"),
            channel=self.session_info.get("channel"),
            operator=self._get_operator_label(),
        )

    async def handle_binary_audio(self, data: bytes):
        chunk_start = time.perf_counter()
        samples = pcm_bytes_to_float32(data, self.dtype_hint)
        if self.enhancement_pipeline is not None and not self.enhancement_config.is_passthrough:
            samples = self.enhancement_pipeline.process(samples, self.sample_rate_client, self.enhancement_config)
        self.total_samples_in += samples.size
        self.cur_utt_audio.append(samples)
        metrics = getattr(self.app.state, "session_metrics", None)
        if metrics is not None:
            metrics["audio_queue_depth"] = max(
                int(metrics.get("audio_queue_depth", 0)),
                len(self.cur_utt_audio),
            )

        chunk_ms = _ms(samples.size, self.sample_rate_client)
        self.stream.accept_waveform(self.sample_rate_client, samples)
        decode_iters = 0
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
            decode_iters += 1
        decode_time_ms = int((time.perf_counter() - chunk_start) * 1000)

        text = self.recognizer.get_result(self.stream)
        speaker = (
            (self.current_speaker_candidate or {}).get("username")
            if self.cur_utt_speaker_guess_sent
            else "unknown"
        )
        endpoint_detected = self.recognizer.is_endpoint(self.stream)

        # logger.info(
        #     (
        #         "asr.decode session=%s chunk_ms=%s decode_ms=%s decodes=%s "
        #         "total_ms=%s endpoint=%s text_len=%s"
        #     ),
        #     self.ws.scope.get("session_id"),
        #     chunk_ms,
        #     decode_time_ms,
        #     decode_iters,
        #     _ms(self.total_samples_in, self.sample_rate_client),
        #     endpoint_detected,
        #     len(text),
        # )

        if self.speaker_recognition_enabled and not self.cur_utt_speaker_guess_sent:
            guess, sim, cand, topk = self._try_speaker(force=False)
            speaker = guess
            self.latest_topk = topk
            if cand is not None:
                self.cur_utt_speaker_guess_sent = True
                self.current_speaker_candidate = cand
                await self._send_speaker_state(cand, sim)
        elif not self.speaker_recognition_enabled:
            self.latest_topk = []
        await self._send_partial(text, speaker)

        if endpoint_detected:
            final_text = text.strip()
            if not final_text:
                self.recognizer.reset(self.stream)
                self.cur_utt_audio.clear()
                self._last_partial_text_sent = None
                self.cur_utt_started_at = time.perf_counter()
                self.cur_utt_start_sample = self.total_samples_in
            else:
                if self.speaker_recognition_enabled:
                    final_spk, final_sim, cand, topk = self._try_speaker(force=True)
                    self.latest_topk = topk
                    if cand is not None:
                        self.current_speaker_candidate = cand
                        await self._send_speaker_state(cand, final_sim)
                else:
                    final_spk, final_sim, cand, topk = "unknown", 0.0, None, []
                    self.latest_topk = []
                await self._send_final(
                    final_text,
                    final_spk,
                    final_sim,
                    topk or self.latest_topk,
                    cand or self.current_speaker_candidate,
                )
                self._last_partial_text_sent = None
                self.recognizer.reset(self.stream)

    async def handle_done(self):
        text = self.recognizer.get_result(self.stream).strip()
        if text:
            if self.speaker_recognition_enabled:
                final_spk, final_sim, cand, topk = self._try_speaker(force=True)
                self.latest_topk = topk
                if cand is not None:
                    self.current_speaker_candidate = cand
                    await self._send_speaker_state(cand, final_sim)
            else:
                final_spk, final_sim, cand, topk = "unknown", 0.0, None, []
                self.latest_topk = []
            await self._send_final(
                text,
                final_spk,
                final_sim,
                topk or self.latest_topk,
                cand or self.current_speaker_candidate,
            )
        finalize_transcript(session_id=self.session_id, status="completed")

    async def handle_text_message(self, raw: str) -> bool:
        text = raw.strip()
        if not text:
            return False

        if text.upper() == "DONE":
            await self.handle_done()
            await self.ws.send_json({"type": "done"})
            await self.ws.close()
            return True

        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            return False

        msg_type = message.get("type")
        data = message.get("data") or {}

        if msg_type == "audio.start":
            await self._handle_audio_start(data)
            return False

        if msg_type == "audio.stop":
            await self.handle_done()
            await self.ws.send_json({"type": "done"})
            await self.ws.close()
            return True

        if msg_type == "control.ping":
            await self.ws.send_json({"type": "control.pong", "time": int(time.time() * 1000)})
            return False

        return False

    async def _handle_audio_start(self, data: Dict[str, Any]):
        sample_rate = int(data.get("sampleRate") or self.sample_rate_client)
        channels = int(data.get("channels") or 1)
        fmt = str(data.get("format", "PCM16")).upper()

        if fmt not in {"PCM16", "PCM16LE"}:
            await self.ws.send_json(
                {
                    "type": "error",
                    "code": "UNSUPPORTED_AUDIO_FORMAT",
                    "message": f"Unsupported audio format: {fmt}",
                }
            )
            return

        if channels != 1:
            await self.ws.send_json(
                {
                    "type": "error",
                    "code": "UNSUPPORTED_CHANNELS",
                    "message": "Only mono audio is supported",
                }
            )
            return

        try:
            self._apply_enhancement_preferences(data)
        except ValidationError as exc:
            await self.ws.send_json(
                {
                    "type": "error",
                    "code": "INVALID_ENHANCEMENT_CONFIG",
                    "message": "Invalid enhancement settings",
                    "details": exc.errors(),
                }
            )
            return

        self.sample_rate_client = sample_rate
        self.dtype_hint = "int16" if fmt.startswith("PCM16") else "float32"
        self.handshake_received = True
        session_id = data.get("sessionId") or self.ws.scope.get("session_id")
        self.session_info = {
            "session_id": session_id,
            "operator": data.get("operator"),
            "locale": data.get("locale", "zh-CN"),
            "channel": data.get("channel"),
            "token": data.get("token"),
        }
        await self._initialize_command_matching(data.get("token"))

        logger.info(
            "audio.start session=%s sample_rate=%s format=%s channels=%s operator=%s",
            session_id,
            self.sample_rate_client,
            fmt,
            channels,
            data.get("operator"),
        )

        await self._send_meta(
            {
                "sessionId": session_id,
                "threshold": self.args.threshold,
                "sampleRate": self.sample_rate_client,
                "model": getattr(self.args, "tokens", None),
                "speakerModel": getattr(self.embedder, "model_path", None),
                "heartbeatInterval": 20000,
                "commandMatchingEnabled": self.command_matching_enabled,
                "commandMatchThreshold": self.command_match_threshold,
                "speakerRecognitionEnabled": self.speaker_recognition_enabled,
                "enhancement": self.enhancement_config.export_metadata(),
            }
        )

        record_event_log(
            session_id=session_id,
            user_id=None,
            username=None,
            operator=self._get_operator_label(),
            event_type="session",
            category="audio_start",
            authorized=True,
            payload={
                "sample_rate": self.sample_rate_client,
                "format": fmt,
                "channels": channels,
                "locale": self.session_info.get("locale"),
                "command_matching_enabled": self.command_matching_enabled,
                "speaker_recognition_enabled": self.speaker_recognition_enabled,
            },
        )

    async def _initialize_command_matching(self, token: Optional[str]):
        self.command_user_id = None
        self.command_matching_enabled = False
        self.command_match_threshold = None

        if not token:
            return
        try:
            user_payload = await validate_access_token(token)
        except HTTPException as exc:
            detail = getattr(exc, "detail", str(exc))
            logger.warning("command.match auth_failed error=%s", detail)
            return

        settings = self.command_service.get_settings(user_payload.id)
        self.command_user_id = user_payload.id
        self.command_matching_enabled = bool(settings.enable_matching)
        self.command_match_threshold = settings.match_threshold or self.command_service.default_threshold

    def _evaluate_command_match(self, text: str) -> Dict[str, Any]:
        if not self.command_user_id:
            return {"matched": False}
        try:
            settings = self.command_service.get_settings(self.command_user_id)
            self.command_matching_enabled = bool(settings.enable_matching)
            self.command_match_threshold = settings.match_threshold or self.command_service.default_threshold
            if not settings.enable_matching:
                return {"matched": False}
            result = self.command_service.match_command(
                self.command_user_id,
                text,
                threshold_override=self.command_match_threshold,
                settings=settings,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("command.match failed error=%s", exc, exc_info=True)
            return {"matched": False}

        payload: Dict[str, Any] = {"matched": bool(result.matched)}
        if result.matched:
            payload["command"] = result.command
            if result.command_code:
                payload["code"] = result.command_code
            if result.command_id is not None:
                payload["command_id"] = result.command_id
        payload["score"] = result.score
        return payload

    def _maybe_forward_command(self, command_match: Dict[str, Any], speaker: str) -> None:
        code = (command_match or {}).get("code")
        if not code:
            return

        async def _run():
            try:
                await forward_command_match(code=code, speaker=speaker)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("command.forward failed code=%s error=%s", code, exc)

        asyncio.create_task(_run())


__all__ = ["AsrSession"]
