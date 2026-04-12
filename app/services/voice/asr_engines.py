import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sherpa_onnx


logger = logging.getLogger("asr.engines")


def get_asr_mode(args: Namespace) -> str:
    return getattr(args, "asr_mode", "streaming_transducer") or "streaming_transducer"


def get_asr_model_path(args: Namespace) -> str:
    if get_asr_mode(args) == "offline_sense_voice":
        return getattr(args, "sense_voice_model", "") or ""
    return getattr(args, "encoder", "") or ""


def _normalize_text(text: Optional[str]) -> str:
    return (text or "").strip()


def _sense_voice_language(args: Namespace) -> str:
    language = getattr(args, "sense_voice_language", "auto") or "auto"
    return "" if language == "auto" else language


def build_online_transducer_recognizer(
    *,
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
        rule1_min_trailing_silence=rule1_min_trailing_silence,
        rule2_min_trailing_silence=rule2_min_trailing_silence,
        rule3_min_utterance_length=rule3_min_utterance_length,
        enable_endpoint_detection=True,
    )

    logger.info(
        "streaming_transducer rules rule1=%s rule2=%s rule3=%s",
        rule1_min_trailing_silence,
        rule2_min_trailing_silence,
        rule3_min_utterance_length,
    )
    return recognizer


def build_offline_sense_voice_recognizer(args: Namespace):
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=getattr(args, "sense_voice_model", ""),
        tokens=getattr(args, "tokens", ""),
        num_threads=getattr(args, "num_threads", 1),
        sample_rate=getattr(args, "sample_rate", 16000),
        feature_dim=getattr(args, "feature_dim", 80),
        decoding_method=getattr(args, "decoding_method", "greedy_search"),
        provider=getattr(args, "provider", "cpu"),
        language=_sense_voice_language(args),
        use_itn=bool(getattr(args, "sense_voice_use_itn", True)),
        hr_rule_fsts=getattr(args, "hr_rule_fsts", ""),
        hr_lexicon=getattr(args, "hr_lexicon", ""),
    )


@dataclass(frozen=True)
class AsrEngineResult:
    text: str
    is_final: bool
    has_partial: bool = False


class BaseAsrEngine(ABC):
    def __init__(self, *, mode: str, provider: str, model_path: str):
        self.mode = mode
        self.provider = provider
        self.model_path = model_path

    @abstractmethod
    def on_utterance_start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def finalize_utterance(self, sample_rate: int, samples: np.ndarray) -> str:
        raise NotImplementedError

    @abstractmethod
    def flush_session(self) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class StreamingTransducerEngine(BaseAsrEngine):
    def __init__(self, args: Namespace):
        super().__init__(
            mode="streaming_transducer",
            provider=getattr(args, "provider", "cpu"),
            model_path=getattr(args, "encoder", "") or "",
        )
        self.recognizer = build_online_transducer_recognizer(
            tokens=getattr(args, "tokens", ""),
            encoder=getattr(args, "encoder", ""),
            decoder=getattr(args, "decoder", ""),
            joiner=getattr(args, "joiner", ""),
            num_threads=getattr(args, "num_threads", 1),
            sample_rate=getattr(args, "sample_rate", 16000),
            feature_dim=getattr(args, "feature_dim", 80),
            decoding_method=getattr(args, "decoding_method", "greedy_search"),
            max_active_paths=getattr(args, "max_active_paths", 4),
            provider=getattr(args, "provider", "cpu"),
            hotwords_file=getattr(args, "hotwords_file", ""),
            hotwords_score=getattr(args, "hotwords_score", 1.5),
            blank_penalty=getattr(args, "blank_penalty", 0.0),
            hr_rule_fsts=getattr(args, "hr_rule_fsts", ""),
            hr_lexicon=getattr(args, "hr_lexicon", ""),
            rule1_min_trailing_silence=getattr(args, "rule1_min_trailing_silence", 1.2),
            rule2_min_trailing_silence=getattr(args, "rule2_min_trailing_silence", 0.8),
            rule3_min_utterance_length=getattr(args, "rule3_min_utterance_length", 300),
        )
        self.stream = self.recognizer.create_stream()

    def on_utterance_start(self) -> None:
        self.reset()

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> Optional[str]:
        if samples.size == 0:
            return None
        self.stream.accept_waveform(sample_rate, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return _normalize_text(self.recognizer.get_result(self.stream))

    def finalize_utterance(self, sample_rate: int, samples: np.ndarray) -> str:
        # Streaming mode is fed incrementally through accept_waveform();
        # finalize only flushes the current stream state.
        text = self.flush_session() or ""
        self.reset()
        return text

    def flush_session(self) -> Optional[str]:
        try:
            self.stream.input_finished()
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
        except Exception:
            logger.exception("streaming_transducer flush failed")
        return _normalize_text(self.recognizer.get_result(self.stream))

    def reset(self) -> None:
        try:
            self.recognizer.reset(self.stream)
        except Exception:
            logger.exception("streaming_transducer reset failed; recreating stream")
            self.stream = self.recognizer.create_stream()


class OfflineSenseVoiceEngine(BaseAsrEngine):
    def __init__(self, args: Namespace):
        super().__init__(
            mode="offline_sense_voice",
            provider=getattr(args, "provider", "cpu"),
            model_path=getattr(args, "sense_voice_model", "") or "",
        )
        self.recognizer = build_offline_sense_voice_recognizer(args)
        self.use_itn = bool(getattr(args, "sense_voice_use_itn", True))
        self.language = getattr(args, "sense_voice_language", "auto") or "auto"

    def on_utterance_start(self) -> None:
        return None

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> Optional[str]:
        return None

    def finalize_utterance(self, sample_rate: int, samples: np.ndarray) -> str:
        if samples.size == 0:
            return ""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, np.ascontiguousarray(samples, dtype=np.float32))
        self.recognizer.decode_stream(stream)
        return _normalize_text(getattr(stream.result, "text", ""))

    def flush_session(self) -> Optional[str]:
        return None

    def reset(self) -> None:
        return None


class AsrEngineFactory:
    @staticmethod
    def create(args: Namespace) -> BaseAsrEngine:
        mode = get_asr_mode(args)
        if mode == "streaming_transducer":
            return StreamingTransducerEngine(args)
        if mode == "offline_sense_voice":
            return OfflineSenseVoiceEngine(args)
        raise ValueError(f"Unsupported ASR mode: {mode}")


def create_streaming_transducer_engine(args: Namespace) -> StreamingTransducerEngine:
    return StreamingTransducerEngine(args)


def create_offline_sense_voice_engine(args: Namespace) -> OfflineSenseVoiceEngine:
    return OfflineSenseVoiceEngine(args)


def create_asr_engine(args: Namespace) -> BaseAsrEngine:
    return AsrEngineFactory.create(args)


__all__ = [
    "AsrEngineFactory",
    "AsrEngineResult",
    "BaseAsrEngine",
    "OfflineSenseVoiceEngine",
    "StreamingTransducerEngine",
    "build_offline_sense_voice_recognizer",
    "build_online_transducer_recognizer",
    "create_asr_engine",
    "create_offline_sense_voice_engine",
    "create_streaming_transducer_engine",
    "get_asr_mode",
    "get_asr_model_path",
]
