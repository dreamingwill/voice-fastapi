import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import ValidationError

from ..command_forwarder import forward_command_match
from ..events import record_event_log
from ..transcripts import append_transcript_segment, finalize_transcript
from ...auth import validate_access_token
from ..audio_enhancement import AudioEnhancementPipeline, EnhancementConfig
from ..commands import get_command_service
from .recognizer import create_recognizer, pcm_bytes_to_float32
from .vad import EnergyVad, VadResult, VadTransition
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
        self.recognizer = create_recognizer(
            tokens=self.args.tokens,
            encoder=self.args.encoder,
            decoder=self.args.decoder,
            joiner=self.args.joiner,
            num_threads=self.args.num_threads,
            sample_rate=self.args.sample_rate,
            feature_dim=self.args.feature_dim,
            decoding_method=self.args.decoding_method,
            max_active_paths=self.args.max_active_paths,
            provider=self.args.provider,
            hotwords_file=self.args.hotwords_file,
            hotwords_score=self.args.hotwords_score,
            blank_penalty=self.args.blank_penalty,
            hr_rule_fsts=self.args.hr_rule_fsts,
            hr_lexicon=self.args.hr_lexicon,
            rule1_min_trailing_silence=self.args.rule1_min_trailing_silence,
            rule2_min_trailing_silence=self.args.rule2_min_trailing_silence,
            rule3_min_utterance_length=self.args.rule3_min_utterance_length,
        )
        self.embedder: SpeakerEmbedder = app.state.embedder
        settings = getattr(app.state, "system_settings", None)
        self.speaker_recognition_enabled = bool(getattr(settings, "enable_speaker_recognition", True))

        self.sample_rate_client = getattr(self.embedder, "sample_rate", None) or self.args.sample_rate or 16000
        self.dtype = "float32"
        self.dtype_hint = "float32"

        self.stream = self.recognizer.create_stream()
        self.total_samples_in = 0
        self.cur_utt_start_sample = 0
        self.cur_utt_audio: Deque[np.ndarray] = deque()
        self.cur_utt_audio_samples = 0
        self._speaker_buffer_max_s = 8.0
        self.cur_utt_speaker_guess_sent = False
        self.segment_id = 0
        self.session_started_at = time.perf_counter()
        self.cur_utt_started_at = self.session_started_at
        self.cur_utt_end_sample: Optional[int] = None
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
        self._last_speaker_eval_samples = 0
        self._last_speaker_eval_at = 0.0
        self._speaker_eval_min_interval_s = 0.6
        self._speaker_eval_min_delta_s = 1.2
        self.command_service = get_command_service()
        self.command_user_id: Optional[int] = None
        self.command_matching_enabled = False
        self.command_match_threshold: Optional[float] = None
        self.enhancement_pipeline: Optional[AudioEnhancementPipeline] = getattr(app.state, "enhancement_pipeline", None)
        self.enhancement_config = EnhancementConfig()
        self.vad = self._create_vad()
        self._last_vad_result: Optional[VadResult] = None

    def _concat_cur_utt_audio(self) -> np.ndarray:
        if not self.cur_utt_audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self.cur_utt_audio), axis=0)

    def _resample_for_speaker(self, samples: np.ndarray) -> np.ndarray:
        target_sr = getattr(self.embedder, "sample_rate", None) or self.sample_rate_client
        src_sr = self.sample_rate_client
        if samples.size == 0 or src_sr == target_sr:
            return samples
        new_len = int(round(samples.size * target_sr / src_sr))
        if new_len <= 1:
            return samples
        x_old = np.arange(samples.size, dtype=np.float32)
        x_new = np.linspace(0, samples.size - 1, new_len, dtype=np.float32)
        resampled = np.interp(x_new, x_old, samples).astype(np.float32)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def _try_speaker(
        self, force: bool = False
    ) -> Tuple[str, float, Optional[SpeakerCandidate], List[SpeakerCandidate]]:
        if not self.speaker_recognition_enabled or self.embedder is None:
            return "unknown", 0.0, None, []
        self._last_speaker_eval_latency_ms = None
        if not force:
            now = time.perf_counter()
            if (now - self._last_speaker_eval_at) < self._speaker_eval_min_interval_s:
                return "unknown", 0.0, None, []
        buf = self._concat_cur_utt_audio()
        need_len = int(self.args.min_spk_seconds * self.sample_rate_client)
        if (not force) and (buf.size < need_len):
            return "unknown", 0.0, None, []
        if not force:
            min_delta = int(self._speaker_eval_min_delta_s * self.sample_rate_client)
            if (buf.size - self._last_speaker_eval_samples) < min_delta:
                return "unknown", 0.0, None, []
        self._last_speaker_eval_samples = buf.size
        self._last_speaker_eval_at = time.perf_counter()
        buf = self._resample_for_speaker(buf)

        eval_start = time.perf_counter()
        st = self.embedder.create_stream()
        st.accept_waveform(sample_rate=self.embedder.sample_rate, waveform=buf)
        if force:
            st.input_finished()
        try:
            if not self.embedder.is_ready(st):
                return "unknown", 0.0, None, []
            emb = self.embedder.compute(st)
            emb = np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            logger.warning("speaker.embed failed error=%s", exc)
            return "unknown", 0.0, None, []

        try:
            matched, top_sim, topk = identify_user(emb, threshold=self.args.threshold)
        except Exception as exc:
            logger.warning("speaker.identify failed error=%s", exc)
            return "unknown", 0.0, None, []
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

    def _create_vad(self) -> EnergyVad:
        max_utt = getattr(self.args, "vad_max_utterance_ms", None)
        if isinstance(max_utt, (int, float)) and max_utt <= 0:
            max_utt = None
        return EnergyVad(
            sample_rate=self.sample_rate_client,
            pre_roll_ms=getattr(self.args, "vad_pre_roll_ms", 300),
            post_roll_ms=getattr(self.args, "vad_post_roll_ms", 700),
            snr_open_db=getattr(self.args, "vad_snr_open_db", 10.0),
            open_min_ms=getattr(self.args, "vad_open_min_ms", 120),
            end_silence_ms=getattr(self.args, "vad_end_silence_ms", 900),
            max_utterance_ms=max_utt,
        )

    def _rebuild_vad(self, sample_rate: int) -> None:
        self._rebuild_vad(sample_rate)
        self.vad = self._create_vad()

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

    def _append_cur_utt_audio(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self.cur_utt_audio.append(samples)
        self.cur_utt_audio_samples += samples.size
        max_samples = int(self._speaker_buffer_max_s * self.sample_rate_client)
        while self.cur_utt_audio_samples > max_samples and self.cur_utt_audio:
            dropped = self.cur_utt_audio.popleft()
            self.cur_utt_audio_samples -= dropped.size

    def _reset_utterance_state(self, start_sample: Optional[int] = None) -> None:
        self.cur_utt_audio.clear()
        self.cur_utt_audio_samples = 0
        self.cur_utt_speaker_guess_sent = False
        self.latest_topk = []
        self.current_speaker_candidate = None
        self._last_partial_text_sent = None
        self.cur_utt_started_at = time.perf_counter()
        if start_sample is not None:
            self.cur_utt_start_sample = start_sample
        else:
            self.cur_utt_start_sample = self.total_samples_in
        self.cur_utt_end_sample = None

    def _log_vad_transitions(self, transitions: List[VadTransition]) -> None:
        if not transitions:
            return
        for transition in transitions:
            logger.info(
                (
                    "vad.transition session=%s from=%s to=%s reason=%s snr_db=%.2f "
                    "total_ms=%s utt_ms=%s dropped_ms=%s"
                ),
                self.ws.scope.get("session_id"),
                transition.prev_state,
                transition.new_state,
                transition.reason,
                transition.snr_db,
                transition.total_ms,
                transition.utt_ms,
                transition.dropped_ms,
            )

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
        end_sample = self.cur_utt_end_sample or self.total_samples_in
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
                "end_ms": _ms(end_sample, self.sample_rate_client),
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
                "end_ms": _ms(end_sample, self.sample_rate_client),
                "topk": meta_topk,
                "command_match": command_match,
            },
        )

        self._persist_transcript_segment(
            text=text,
            speaker=speaker,
            similarity=similarity,
            start_ms=_ms(self.cur_utt_start_sample, self.sample_rate_client),
            end_ms=_ms(end_sample, self.sample_rate_client),
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
        self.cur_utt_audio_samples = 0
        self.cur_utt_speaker_guess_sent = False
        self.cur_utt_started_at = time.perf_counter()
        self.latest_topk = []
        self.current_speaker_candidate = None
        self.cur_utt_end_sample = None

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

        vad_result = self.vad.process(samples)
        self._last_vad_result = vad_result
        self._log_vad_transitions(vad_result.transitions)

        if vad_result.started:
            try:
                self.recognizer.reset(self.stream)
            except Exception:
                pass
            self._reset_utterance_state(start_sample=vad_result.start_sample)

        if not vad_result.feed_samples:
            return

        feed = (
            np.concatenate(vad_result.feed_samples, axis=0)
            if len(vad_result.feed_samples) > 1
            else vad_result.feed_samples[0]
        )
        if feed.size == 0:
            return

        self._append_cur_utt_audio(feed)
        metrics = getattr(self.app.state, "session_metrics", None)
        if metrics is not None:
            metrics["audio_queue_depth"] = max(
                int(metrics.get("audio_queue_depth", 0)),
                len(self.cur_utt_audio),
            )

        chunk_ms = _ms(feed.size, self.sample_rate_client)
        self.stream.accept_waveform(self.sample_rate_client, feed)
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

        if vad_result.ended:
            await self._finalize_vad_utterance(
                end_reason=vad_result.end_reason or "end_silence",
                end_sample=vad_result.end_sample,
                snr_db=vad_result.snr_db,
                dropped_samples=vad_result.dropped_samples,
            )

    async def _finalize_vad_utterance(
        self,
        *,
        end_reason: str,
        end_sample: Optional[int],
        snr_db: float,
        dropped_samples: int,
    ) -> None:
        try:
            self.stream.input_finished()
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
        except Exception as exc:
            logger.warning("vad.flush failed error=%s", exc)

        self.cur_utt_end_sample = end_sample
        final_text = self.recognizer.get_result(self.stream).strip()
        final_triggered = bool(final_text)

        utt_ms = _ms(
            (end_sample or self.total_samples_in) - self.cur_utt_start_sample,
            self.sample_rate_client,
        )

        if final_triggered:
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
        else:
            self._reset_utterance_state()

        logger.info(
            (
                "vad.final session=%s reason=%s triggered=%s utt_ms=%s snr_db=%.2f "
                "dropped_ms=%s text_len=%s"
            ),
            self.ws.scope.get("session_id"),
            end_reason,
            final_triggered,
            utt_ms,
            snr_db,
            _ms(dropped_samples, self.sample_rate_client),
            len(final_text),
        )

        self._last_partial_text_sent = None
        try:
            self.recognizer.reset(self.stream)
        except Exception as exc:
            logger.warning("vad.stream.reset failed error=%s", exc)

    async def handle_done(self):
        if self.vad.state in {"SPEECH", "TAIL"}:
            last = self._last_vad_result
            await self._finalize_vad_utterance(
                end_reason="client_done",
                end_sample=self.total_samples_in,
                snr_db=(last.snr_db if last else 0.0),
                dropped_samples=(last.dropped_samples if last else 0),
            )
        else:
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
        self._flush_and_reset_stream()

    def _flush_and_reset_stream(self) -> None:
        try:
            self.stream.input_finished()
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            self.recognizer.reset(self.stream)
        except Exception as exc:
            logger.warning("asr.stream.cleanup failed error=%s", exc)
        self.cur_utt_audio.clear()
        self.cur_utt_audio_samples = 0
        self.cur_utt_end_sample = None

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

        # Allow per-session override for speaker recognition (visitor mode)
        try:
            if data.get("speakerRecognitionEnabled") is not None:
                self.speaker_recognition_enabled = bool(data.get("speakerRecognitionEnabled"))
        except Exception:
            # ignore malformed override
            pass

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
        # Enable global command matching regardless of authentication.
        # Use a truthy placeholder user id; CommandService internally uses global scope.
        self.command_user_id = 1
        self.command_matching_enabled = False
        self.command_match_threshold = None

        # Try to validate token if provided (for logging/auth metrics only).
        if token:
            try:
                await validate_access_token(token)
            except HTTPException as exc:  # pragma: no cover - non-fatal for global matching
                detail = getattr(exc, "detail", str(exc))
                logger.warning("command.match auth_failed error=%s", detail)

        settings = self.command_service.get_settings(self.command_user_id)
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
                error = await forward_command_match(code=code, speaker=speaker)
                if error:
                    try:
                        await self.ws.send_json(
                            {
                                "type": "command.forward.error",
                                "code": code,
                                "speaker": speaker,
                                "error": error,
                            }
                        )
                    except Exception:
                        pass
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("command.forward failed code=%s error=%s", code, exc)

        # Block forwarding if speaker is unknown
        if not speaker or speaker == "unknown":
            try:
                # annotate the command_match payload for frontend awareness
                command_match["blocked"] = True
                command_match["block_reason"] = "unknown_speaker"
            except Exception:
                pass
            logger.info(
                "command.forward blocked due to unknown speaker code=%s speaker=%s",
                code,
                speaker,
            )
            return

        # otherwise forward asynchronously
        asyncio.create_task(_run())


__all__ = ["AsrSession"]
