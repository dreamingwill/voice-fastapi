import json
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sherpa_onnx

from ...database import SessionLocal
from ...models import User

SpeakerCandidate = Dict[str, Any]


class SpeakerEmbedder:
    def __init__(
        self,
        model_path: str = "./models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
        sample_rate: int = 16000,
        threshold: float = 0.6,
        provider: str = "cpu",
        num_threads: int = 4,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.model_path,
            num_threads=num_threads,
            provider=provider,
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

    def embed_from_waveform(self, samples: np.ndarray, sample_rate: int, force: bool = False) -> Optional[np.ndarray]:
        stream = self.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        if force:
            stream.input_finished()
        if not self.is_ready(stream):
            return None
        embedding = self.compute(stream)
        return np.asarray(embedding, dtype=np.float32)


class RknnSpeakerEmbedder:
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        threshold: float = 0.6,
        feature_dim: int = 80,
        num_frames: int = 300,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        core_mask: str = "auto",
        l2_normalize: bool = False,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.core_mask = core_mask
        self.l2_normalize = l2_normalize
        self._lock = threading.Lock()
        self._rknn = None
        self._runtime_ready = False

    def embed_from_waveform(self, samples: np.ndarray, sample_rate: int, force: bool = False) -> Optional[np.ndarray]:
        if samples.size == 0:
            return None
        feats = _compute_fbank(
            samples,
            sample_rate,
            feature_dim=self.feature_dim,
            frame_length_ms=self.frame_length_ms,
            frame_shift_ms=self.frame_shift_ms,
        )
        feats = _fix_num_frames(feats, self.num_frames)
        emb = self._rknn_embed(feats)
        if self.l2_normalize:
            denom = np.linalg.norm(emb) + 1e-10
            emb = emb / denom
        return emb

    def _ensure_runtime(self) -> None:
        if self._runtime_ready:
            return
        self._rknn = _create_rknn_runtime(self.model_path, self.core_mask)
        self._runtime_ready = True

    def _rknn_embed(self, feats: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self._runtime_ready:
                self._ensure_runtime()
            assert self._rknn is not None
            x = feats[np.newaxis, :, :].astype(np.float32)
            out = self._rknn.inference(inputs=[x])[0]
            emb = np.squeeze(out)
            return emb.astype(np.float32)


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def identify_user(
    query_embedding: np.ndarray, threshold: float
) -> Tuple[Optional[SpeakerCandidate], float, List[SpeakerCandidate]]:
    sims: List[Tuple[SpeakerCandidate, float]] = []
    if query_embedding is None or query_embedding.size == 0:
        return None, 0.0, []

    with SessionLocal() as db:
        users = (
            db.query(User)
            .filter(User.embedding.isnot(None))
            .filter(User.status != "disabled")
            .all()
        )

        for user in users:
            if not user.embedding:
                continue
            try:
                stored_embedding = np.array(json.loads(user.embedding), dtype=np.float32)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if stored_embedding.size == 0:
                continue
            if stored_embedding.shape != query_embedding.shape:
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


def create_speaker_embedder(
    model_path: str,
    sample_rate: int,
    threshold: float,
    provider: str,
    num_threads: int,
    rknn_feature_dim: int,
    rknn_num_frames: int,
    rknn_frame_length_ms: float,
    rknn_frame_shift_ms: float,
    rknn_core: str,
    rknn_l2_normalize: bool,
):
    if model_path.endswith(".rknn"):
        return RknnSpeakerEmbedder(
            model_path=model_path,
            sample_rate=sample_rate,
            threshold=threshold,
            feature_dim=rknn_feature_dim,
            num_frames=rknn_num_frames,
            frame_length_ms=rknn_frame_length_ms,
            frame_shift_ms=rknn_frame_shift_ms,
            core_mask=rknn_core,
            l2_normalize=rknn_l2_normalize,
        )
    return SpeakerEmbedder(
        model_path=model_path,
        sample_rate=sample_rate,
        threshold=threshold,
        provider=provider,
        num_threads=num_threads,
    )


def _compute_fbank(
    samples: np.ndarray,
    sample_rate: int,
    feature_dim: int,
    frame_length_ms: float,
    frame_shift_ms: float,
) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for RKNN speaker embeddings") from exc

    frame_length = int(sample_rate * frame_length_ms / 1000.0)
    frame_shift = int(sample_rate * frame_shift_ms / 1000.0)
    if frame_length <= 0 or frame_shift <= 0:
        raise RuntimeError("Frame length/shift must be positive")
    n_fft = 1
    while n_fft < frame_length:
        n_fft <<= 1
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=frame_shift,
        n_mels=feature_dim,
        fmin=0.0,
        fmax=sample_rate / 2.0,
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel, 1e-10))
    log_mel -= np.mean(log_mel, axis=1, keepdims=True)
    return log_mel.astype(np.float32)


def _fix_num_frames(feats: np.ndarray, num_frames: int) -> np.ndarray:
    if feats.ndim != 2:
        raise RuntimeError("Expected feature shape (feature_dim, frames)")
    feature_dim, frames = feats.shape
    if frames == num_frames:
        return feats
    if frames < num_frames:
        pad = np.zeros((feature_dim, num_frames - frames), dtype=feats.dtype)
        return np.concatenate([feats, pad], axis=1)
    start = max(0, (frames - num_frames) // 2)
    return feats[:, start : start + num_frames]


def _resolve_core_mask(value: str) -> int:
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError("rknnlite is required for RKNN speaker embeddings") from exc

    def _get(name: str, fallback: Optional[int] = None) -> Optional[int]:
        return getattr(RKNNLite, name, fallback)

    mapping = {
        "auto": _get("NPU_CORE_AUTO"),
        "0": _get("NPU_CORE_0"),
        "1": _get("NPU_CORE_1"),
        "2": _get("NPU_CORE_2"),
        "01": _get("NPU_CORE_0_1"),
        "12": _get("NPU_CORE_1_2"),
        "012": _get("NPU_CORE_0_1_2"),
    }
    chosen = mapping.get(value)
    if chosen is None:
        fallback = _get("NPU_CORE_AUTO")
        if fallback is None:
            raise RuntimeError(f"RKNNLite does not expose core mask constants for '{value}'")
        return fallback
    return chosen


def _create_rknn_runtime(model_path: str, core_mask: str):
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError("rknnlite is required for RKNN speaker embeddings") from exc

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime(core_mask=_resolve_core_mask(core_mask))
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    return rknn


__all__ = [
    "SpeakerEmbedder",
    "RknnSpeakerEmbedder",
    "create_speaker_embedder",
    "identify_user",
    "SpeakerCandidate",
]
