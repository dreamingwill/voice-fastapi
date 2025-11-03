import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sherpa_onnx

from fastapi import FastAPI, WebSocket

from ..database import SessionLocal
from ..models import User
from ..services.events import record_event_log


logger = logging.getLogger("asr.session")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class SpeakerEmbedder:
    def __init__(
        self,
        model_path: str = "./models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
        sample_rate: int = 16000,
        threshold: float = 0.6,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.model_path,
            num_threads=4,
            provider="cpu",
        )
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(self.config)
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        self.threshold = threshold

    def create_stream(self):
        return self.extractor.create_stream()

    def is_ready(self, stream):
        return self.extractor.is_ready(stream)

    def compute(self, stream):
        return self.extractor.compute(stream)

    def embed(self, samples, sample_rate):
        stream = self.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        if not self.is_ready(stream):
            raise RuntimeError("Speaker embedding extractor is not ready")

        embedding = self.compute(stream)
        return np.asarray(embedding, dtype=np.float32)


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


SpeakerCandidate = Dict[str, Any]


def identify_user(
    query_embedding: np.ndarray, threshold: float
) -> Tuple[Optional[SpeakerCandidate], float, List[SpeakerCandidate]]:
    sims: List[Tuple[SpeakerCandidate, float]] = []
    if query_embedding is None or query_embedding.size == 0:
        return None, 0.0, []

    with SessionLocal() as db:
        users = db.query(User).filter(User.embedding.isnot(None)).all()

        for user in users:
            if not user.embedding:
                continue
            try:
                stored_embedding = np.array(json.loads(user.embedding), dtype=np.float32)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if stored_embedding.size == 0:
                continue
            sim = cosine_similarity(query_embedding, stored_embedding)
            candidate: SpeakerCandidate = {
                "id": user.id,
                "username": user.username,
                "identity": user.identity,
            }
            sims.append((candidate, sim))

    if not sims:
        return None, 0.0, []

    sims.sort(key=lambda x: x[1], reverse=True)
    topk: List[SpeakerCandidate] = []
    for candidate, score in sims[:5]:
        enriched = dict(candidate)
        enriched["similarity"] = float(score)
        topk.append(enriched)

    best_candidate = topk[0] if topk else None
    top_sim = best_candidate.get("similarity", 0.0) if best_candidate else 0.0
    matched = best_candidate if best_candidate and top_sim >= threshold else None

    return matched, float(top_sim), topk


def create_recognizer(
    tokens: str = "",
    encoder: str = "",
    decoder: str = "",
    joiner: str = "",
    num_threads: int = 1,
    sample_rate: int = 16000,
    feature_dim: int = 80,
    decoding_method: str = "greedy_search",
    max_active_paths: int = 4,
    provider: str = "cpu",
    hotwords_file: str = "",
    hotwords_score: float = 1.5,
    blank_penalty: float = 0.0,
    hr_rule_fsts: str = "",
    hr_lexicon: str = "",
):
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=num_threads,
        sample_rate=sample_rate,
        feature_dim=feature_dim,
        decoding_method=decoding_method,
        max_active_paths=max_active_paths,
        provider=provider,
        hotwords_file=hotwords_file,
        hotwords_score=hotwords_score,
        blank_penalty=blank_penalty,
        hr_rule_fsts=hr_rule_fsts,
        hr_lexicon=hr_lexicon,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
    )

    return recognizer


def _pcm_bytes_to_float32(data: bytes, dtype: Optional[str]) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)

    try:
        if dtype == "int16":
            return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if dtype == "float32":
            return np.frombuffer(data, dtype=np.float32)
        if dtype == "int24":
            x = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
            x = x.reshape(-1, 3)
            signed = (
                (x[:, 0].astype(np.int32) << 16)
                | (x[:, 1].astype(np.int32) << 8)
                | x[:, 2].astype(np.int32)
            )
            signed = np.where(signed >= 0x800000, signed - 0x1000000, signed)
            return signed.astype(np.float32) / 8388608.0
        # default fallback: try float32 then int16
        x = np.frombuffer(data, dtype=np.float32)
        if x.size == 0:
            return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return x
    except Exception:
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _ms(samples: int, sr: int) -> int:
    return int(samples * 1000 / sr)


class AsrSession:
    def __init__(self, websocket: WebSocket, app: FastAPI):
        self.ws = websocket
        self.app = app
        self.args = app.state.args
        self.recognizer = app.state.recognizer
        self.embedder: SpeakerEmbedder = app.state.embedder

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

    def _concat_cur_utt_audio(self) -> np.ndarray:
        if not self.cur_utt_audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.cur_utt_audio, axis=0)

    def _try_speaker(
        self, force: bool = False
    ) -> Tuple[str, float, Optional[SpeakerCandidate], List[SpeakerCandidate]]:
        buf = self._concat_cur_utt_audio()
        need_len = int(self.args.min_spk_seconds * self.sample_rate_client)
        if (not force) and (buf.size < need_len):
            return "unknown", 0.0, None, []

        st = self.embedder.create_stream()
        st.accept_waveform(sample_rate=self.sample_rate_client, waveform=buf)
        if force:
            st.input_finished()
        if not self.embedder.is_ready(st):
            return "unknown", 0.0, None, []
        emb = self.embedder.compute(st)
        emb = np.asarray(emb, dtype=np.float32)

        matched, top_sim, topk = identify_user(emb, threshold=self.args.threshold)

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
        logger.info(
            "speaker.update session=%s id=%s username=%s role=%s confidence=%.3f",
            self.ws.scope.get("session_id"),
            candidate.get("id"),
            candidate.get("username"),
            candidate.get("identity"),
            similarity,
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

    async def _send_partial(self, text: str, speaker: str):
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

        latency_ms = int((time.perf_counter() - self.cur_utt_started_at) * 1000)
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
            },
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

    async def handle_binary_audio(self, data: bytes):
        samples = _pcm_bytes_to_float32(data, self.dtype_hint)
        self.total_samples_in += samples.size
        self.cur_utt_audio.append(samples)
        metrics = getattr(self.app.state, "session_metrics", None)
        if metrics is not None:
            metrics["audio_queue_depth"] = max(
                int(metrics.get("audio_queue_depth", 0)),
                len(self.cur_utt_audio),
            )

        self.stream.accept_waveform(self.sample_rate_client, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        text = self.recognizer.get_result(self.stream)
        speaker = (
            (self.current_speaker_candidate or {}).get("username")
            if self.cur_utt_speaker_guess_sent
            else "unknown"
        )

        if not self.cur_utt_speaker_guess_sent:
            guess, sim, cand, topk = self._try_speaker(force=False)
            speaker = guess
            self.latest_topk = topk
            if cand is not None:
                self.cur_utt_speaker_guess_sent = True
                self.current_speaker_candidate = cand
                await self._send_speaker_state(cand, sim)
        await self._send_partial(text, speaker)

        if self.recognizer.is_endpoint(self.stream):
            final_spk, final_sim, cand, topk = self._try_speaker(force=True)
            self.latest_topk = topk
            if cand is not None:
                self.current_speaker_candidate = cand
                await self._send_speaker_state(cand, final_sim)
            await self._send_final(text, final_spk, final_sim, topk or self.latest_topk, cand or self.current_speaker_candidate)
            self.recognizer.reset(self.stream)

    async def handle_done(self):
        text = self.recognizer.get_result(self.stream)
        if text.strip():
            final_spk, final_sim, cand, topk = self._try_speaker(force=True)
            self.latest_topk = topk
            if cand is not None:
                self.current_speaker_candidate = cand
                await self._send_speaker_state(cand, final_sim)
            await self._send_final(
                text,
                final_spk,
                final_sim,
                topk or self.latest_topk,
                cand or self.current_speaker_candidate,
            )

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
            await self.ws.send_json({
                "type": "error",
                "code": "UNSUPPORTED_AUDIO_FORMAT",
                "message": f"Unsupported audio format: {fmt}",
            })
            return

        if channels != 1:
            await self.ws.send_json({
                "type": "error",
                "code": "UNSUPPORTED_CHANNELS",
                "message": "Only mono audio is supported",
            })
            return

        self.sample_rate_client = sample_rate
        self.dtype_hint = "int16" if fmt.startswith("PCM16") else "float32"
        self.handshake_received = True
        session_id = data.get("sessionId") or self.ws.scope.get("session_id")
        self.session_info = {
            "session_id": session_id,
            "operator": data.get("operator"),
            "locale": data.get("locale", "zh-CN"),
            "token": data.get("token"),
        }

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
            }
        )

        record_event_log(
            session_id=session_id,
            user_id=None,
            username=None,
            operator=(self.session_info.get("operator") or {}).get("name")
            if isinstance(self.session_info.get("operator"), dict)
            else None,
            event_type="session",
            category="audio_start",
            authorized=True,
            payload={
                "sample_rate": self.sample_rate_client,
                "format": fmt,
                "channels": channels,
                "locale": self.session_info.get("locale"),
            },
        )


__all__ = ["SpeakerEmbedder", "create_recognizer", "identify_user", "AsrSession"]
