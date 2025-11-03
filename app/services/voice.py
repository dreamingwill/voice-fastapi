import json
from typing import List, Optional, Tuple

import numpy as np
import sherpa_onnx

from fastapi import FastAPI, WebSocket

from ..database import SessionLocal
from ..models import User
from ..services.events import record_event_log


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


def identify_user(query_embedding: np.ndarray, threshold: float) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
    sims: List[Tuple[str, float]] = []
    if query_embedding is None or query_embedding.size == 0:
        return None, 0.0, sims

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
            sims.append((user.username, sim))

    if not sims:
        return None, 0.0, []

    sims.sort(key=lambda x: x[1], reverse=True)
    topk = sims[:5]
    top_user, top_sim = topk[0]

    matched = top_user if top_sim >= threshold else None
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
    try:
        x = np.frombuffer(data, dtype=np.float32)
        if x.size == 0:
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
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

    def _concat_cur_utt_audio(self) -> np.ndarray:
        if not self.cur_utt_audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.cur_utt_audio, axis=0)

    def _try_speaker(self, force: bool = False) -> Tuple[str, float]:
        buf = self._concat_cur_utt_audio()
        need_len = int(self.args.min_spk_seconds * self.sample_rate_client)
        if (not force) and (buf.size < need_len):
            return "unknown", 0.0

        st = self.embedder.create_stream()
        st.accept_waveform(sample_rate=self.sample_rate_client, waveform=buf)
        if force:
            st.input_finished()
        if not self.embedder.is_ready(st):
            return "unknown", 0.0
        emb = self.embedder.compute(st)
        emb = np.asarray(emb, dtype=np.float32)

        matched, top_sim, _topk = identify_user(emb, threshold=self.args.threshold)

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
            return "unknown", float(top_sim)
        return matched, float(top_sim)

    async def _send_partial(self, text: str, speaker: str):
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

    async def _send_final(self, text: str, speaker: str, similarity: float):
        await self.ws.send_json(
            {
                "type": "final",
                "segment_id": self.segment_id,
                "start_ms": _ms(self.cur_utt_start_sample, self.sample_rate_client),
                "end_ms": _ms(self.total_samples_in, self.sample_rate_client),
                "text": text,
                "speaker": speaker,
                "similarity": similarity,
            }
        )

        record_event_log(
            session_id=self.ws.scope.get("session_id"),
            user_id=None,
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
            },
        )

        self.segment_id += 1
        self.cur_utt_start_sample = self.total_samples_in
        self.cur_utt_audio.clear()
        self.cur_utt_speaker_guess_sent = False

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
        speaker = "unknown"
        if not self.cur_utt_speaker_guess_sent:
            guess, _sim = self._try_speaker(force=False)
            speaker = guess
            if guess != "unknown":
                self.cur_utt_speaker_guess_sent = True

        await self._send_partial(text, speaker)

        if self.recognizer.is_endpoint(self.stream):
            final_spk, final_sim = self._try_speaker(force=True)
            await self._send_final(text, final_spk, final_sim)
            self.recognizer.reset(self.stream)

    async def handle_done(self):
        text = self.recognizer.get_result(self.stream)
        if text.strip():
            final_spk, final_sim = self._try_speaker(force=True)
            await self._send_final(text, final_spk, final_sim)


__all__ = ["SpeakerEmbedder", "create_recognizer", "identify_user", "AsrSession"]
