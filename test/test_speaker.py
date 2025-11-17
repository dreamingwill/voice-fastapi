#!/usr/bin/env python3
"""
Utility script for validating the speaker embedding model without running FastAPI.

Example:
    python test/test_speaker.py \
        --audio /path/to/sample.wav \
        --config config/app_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import init_db  # noqa: E402
from app.services.voice.speaker import SpeakerEmbedder, identify_user  # noqa: E402

DEFAULT_CONFIG = Path("config/app_config.json")


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON config: {cfg_path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must be a JSON object: {cfg_path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Speaker model tester")
    parser.add_argument("--audio", "-a", required=True, help="Path to a mono/stereo audio file (wav/flac)")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Optional JSON config file that provides defaults (default: config/app_config.json)",
    )
    parser.add_argument("--model-path", help="Override the speaker embedding ONNX path")
    parser.add_argument("--sample-rate", type=int, help="Expected sample rate for the model")
    parser.add_argument("--threshold", type=float, help="Similarity threshold for accepting a speaker")
    parser.add_argument("--topk", type=int, default=5, help="How many candidates to display")
    parser.add_argument("--dump-embedding", action="store_true", help="Print the embedding vector")
    parser.add_argument("--skip-identify", action="store_true", help="Only compute embeddings without DB lookup")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.topk <= 0:
        parser.error("--topk must be > 0")
    return args


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    samples, sample_rate = sf.read(path, dtype="float32")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1:
        raise RuntimeError("Unsupported audio format; expecting mono or stereo")
    return samples.astype(np.float32), sample_rate


def resample_audio(samples: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr or samples.size == 0:
        return samples
    duration = samples.shape[0] / float(original_sr)
    target_length = int(round(duration * target_sr))
    if target_length <= 0:
        return np.asarray([], dtype=np.float32)
    if target_length == 1:
        return np.asarray([float(samples[0])], dtype=np.float32)
    indices = np.arange(samples.shape[0], dtype=np.float32)
    target_indices = np.linspace(0, samples.shape[0] - 1, num=target_length)
    resampled = np.interp(target_indices, indices, samples)
    return resampled.astype(np.float32)


def format_candidate(candidate: Dict[str, Any]) -> str:
    username = candidate.get("username") or "<unknown>"
    identity = candidate.get("identity") or "-"
    similarity = candidate.get("similarity")
    similarity_str = f"{similarity:.4f}" if isinstance(similarity, (float, int)) else "n/a"
    return f"{username} (id={candidate.get('id')}, identity={identity}) -> similarity={similarity_str}"


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_path = args.model_path or cfg.get("model_path")
    if not model_path:
        raise RuntimeError("Model path not provided via --model-path or config file")
    sample_rate = args.sample_rate or cfg.get("sample_rate", 16000)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.6)

    audio_path = Path(args.audio).expanduser()
    audio, original_sr = load_audio(audio_path)
    processed_audio = resample_audio(audio, original_sr, sample_rate)
    if processed_audio.size == 0:
        raise RuntimeError("Audio file does not contain any samples")

    print(f"[speaker] model={model_path}")
    print(f"[speaker] samples={processed_audio.shape[0]}, duration={processed_audio.shape[0] / sample_rate:.2f}s")

    embedder = SpeakerEmbedder(model_path=model_path, sample_rate=sample_rate, threshold=threshold)
    embedding = embedder.embed(processed_audio, sample_rate)
    print(f"[speaker] embedding_dim={embedding.shape[0]}")
    if args.dump_embedding:
        np.set_printoptions(precision=5, suppress=True)
        print("[speaker] embedding=", embedding)

    if args.skip_identify:
        return

    init_db()
    matched, similarity, candidates = identify_user(embedding, threshold=threshold)
    if matched:
        print(f"[match] PASS threshold={threshold:.2f}, similarity={similarity:.4f}")
        print(f"[match] candidate={format_candidate(matched)}")
    else:
        print(f"[match] FAIL threshold={threshold:.2f}, best_similarity={similarity:.4f}")

    if candidates:
        max_rows = min(args.topk, len(candidates))
        print(f"[match] top{max_rows} candidates:")
        for idx, candidate in enumerate(candidates[:max_rows], 1):
            print(f"  {idx}. {format_candidate(candidate)}")
    else:
        print("[match] No eligible candidates found in the database.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
