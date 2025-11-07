import json
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


__all__ = ["SpeakerEmbedder", "identify_user", "SpeakerCandidate"]
