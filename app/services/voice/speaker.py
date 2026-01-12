import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sherpa_onnx

from ...database import SessionLocal
from ...models import User

SpeakerCandidate = Dict[str, Any]
logger = logging.getLogger("speaker.cache")


class _EmbeddingCache:
    def __init__(self, ttl_s: float = 30.0):
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._loaded_at = 0.0
        self._vectors: Optional[np.ndarray] = None
        self._candidates: List[SpeakerCandidate] = []

    def invalidate(self) -> None:
        with self._lock:
            self._loaded_at = 0.0
            self._vectors = None
            self._candidates = []

    def _load_from_db(self) -> None:
        vectors: List[np.ndarray] = []
        candidates: List[SpeakerCandidate] = []
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
                vec = np.array(json.loads(user.embedding), dtype=np.float32)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if vec.size == 0:
                continue
            norm = float(np.linalg.norm(vec)) + 1e-10
            vec = vec / norm
            vectors.append(vec)
            candidates.append(
                {
                    "id": user.id,
                    "username": user.username,
                    "identity": user.identity,
                }
            )
        self._vectors = np.stack(vectors, axis=0) if vectors else None
        self._candidates = candidates
        self._loaded_at = time.time()

    def get(self) -> Tuple[Optional[np.ndarray], List[SpeakerCandidate]]:
        now = time.time()
        if self._vectors is not None and (now - self._loaded_at) < self._ttl_s:
            return self._vectors, self._candidates
        with self._lock:
            now = time.time()
            if self._vectors is not None and (now - self._loaded_at) < self._ttl_s:
                return self._vectors, self._candidates
            try:
                self._load_from_db()
            except Exception as exc:
                logger.warning("speaker.cache.load failed error=%s", exc)
            return self._vectors, self._candidates


_EMBEDDING_CACHE = _EmbeddingCache()


def invalidate_embedding_cache() -> None:
    _EMBEDDING_CACHE.invalidate()


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


def identify_user(
    query_embedding: np.ndarray, threshold: float
) -> Tuple[Optional[SpeakerCandidate], float, List[SpeakerCandidate]]:
    sims: List[Tuple[SpeakerCandidate, float]] = []
    if query_embedding is None or query_embedding.size == 0:
        return None, 0.0, []

    vectors, candidates = _EMBEDDING_CACHE.get()
    if vectors is None or not candidates:
        return None, 0.0, []
    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query)) + 1e-10
    query = query / query_norm
    sims_array = vectors @ query
    if sims_array.size == 0:
        return None, 0.0, []
    top_n = min(5, sims_array.size)
    idx = np.argpartition(-sims_array, top_n - 1)[:top_n]
    sorted_idx = idx[np.argsort(-sims_array[idx])]
    for i in sorted_idx.tolist():
        candidate = candidates[i]
        sims.append((candidate, float(sims_array[i])))

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


__all__ = ["SpeakerEmbedder", "identify_user", "SpeakerCandidate", "invalidate_embedding_cache"]
