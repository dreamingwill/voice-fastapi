from argparse import Namespace
from typing import Optional

import numpy as np

from .asr_engines import (
    AsrEngineFactory,
    BaseAsrEngine,
    OfflineSenseVoiceEngine,
    StreamingTransducerEngine,
    build_online_transducer_recognizer,
    create_asr_engine,
    create_offline_sense_voice_engine,
    create_streaming_transducer_engine,
    get_asr_mode,
    get_asr_model_path,
)


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
    rule1_min_trailing_silence: float = 1.2,
    rule2_min_trailing_silence: float = 0.8,
    rule3_min_utterance_length: int = 300,
):
    return build_online_transducer_recognizer(
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
        rule1_min_trailing_silence=rule1_min_trailing_silence,
        rule2_min_trailing_silence=rule2_min_trailing_silence,
        rule3_min_utterance_length=rule3_min_utterance_length,
    )


def create_asr_engine_from_kwargs(**kwargs) -> BaseAsrEngine:
    return create_asr_engine(Namespace(**kwargs))


def pcm_bytes_to_float32(data: bytes, dtype: Optional[str]) -> np.ndarray:
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
        x = np.frombuffer(data, dtype=np.float32)
        if x.size == 0:
            return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return x
    except Exception:
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


__all__ = [
    "AsrEngineFactory",
    "BaseAsrEngine",
    "OfflineSenseVoiceEngine",
    "StreamingTransducerEngine",
    "create_asr_engine",
    "create_asr_engine_from_kwargs",
    "create_offline_sense_voice_engine",
    "create_recognizer",
    "create_streaming_transducer_engine",
    "get_asr_mode",
    "get_asr_model_path",
    "pcm_bytes_to_float32",
]
