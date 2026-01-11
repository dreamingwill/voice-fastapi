#!/usr/bin/env python3
"""
Utility script for validating the RKNN speaker embedding model without running FastAPI.

Example:
    python test/test_speaker_tdnn.py \
        --audio tmp/zrh_voice.wav \
        --config config/app_config_tdnn.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError as exc:
    raise RuntimeError("librosa is required for fbank extraction") from exc

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise RuntimeError("rknnlite is required for RKNN inference") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = Path("config/app_config_tdnn.json")


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
    parser = argparse.ArgumentParser("Speaker RKNN model tester")
    parser.add_argument("--audio", "-a", help="Path to a mono/stereo audio file (wav/flac)")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Optional JSON config file that provides defaults (default: config/app_config_tdnn.json)",
    )
    parser.add_argument("--model-path", help="Override the speaker embedding RKNN path")
    parser.add_argument("--sample-rate", type=int, help="Expected sample rate for the model")
    parser.add_argument("--threshold", type=float, help="Similarity threshold for accepting a speaker")
    parser.add_argument("--speaker-provider", help="Provider label for speaker embedding model")
    parser.add_argument("--speaker-num-threads", type=int, help="Thread count for speaker embedding model")
    parser.add_argument("--database-url", help="Override database URL for speaker embeddings")
    parser.add_argument("--feature-dim", type=int, default=80, help="Fbank feature dimension")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of frames expected by the model")
    parser.add_argument("--frame-length-ms", type=float, default=25.0, help="Frame length in milliseconds")
    parser.add_argument("--frame-shift-ms", type=float, default=10.0, help="Frame shift in milliseconds")
    parser.add_argument("--l2-normalize", action="store_true", help="Apply L2 normalization to embeddings")
    parser.add_argument("--topk", type=int, default=5, help="How many candidates to display")
    parser.add_argument("--dump-embedding", action="store_true", help="Print the embedding vector")
    parser.add_argument("--skip-identify", action="store_true", help="Only compute embeddings without DB lookup")
    parser.add_argument("--enroll-user-id", type=int, help="Update a user's embedding in the DB")
    parser.add_argument("--record", action="store_true", help="Record from microphone before testing")
    parser.add_argument("--duration", type=float, default=5.0, help="Record duration in seconds when --record")
    parser.add_argument("--device", help="Optional sounddevice input device name/index")
    parser.add_argument(
        "--rknn-core",
        default="auto",
        choices=["auto", "0", "1", "2", "01", "12", "012"],
        help="RKNN NPU core mask (auto/0/1/2/01/12/012)",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.topk <= 0:
        parser.error("--topk must be > 0")
    if args.record and (args.duration is None or args.duration <= 0):
        parser.error("--duration must be > 0 when using --record")
    if not args.audio and not args.record:
        parser.error("--audio is required unless --record is set")
    if args.num_frames <= 0:
        parser.error("--num-frames must be > 0")
    if args.feature_dim <= 0:
        parser.error("--feature-dim must be > 0")
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


def record_audio(path: Path, sample_rate: int, duration: float, device: Optional[str]) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for recording") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration * sample_rate)
    if frames <= 0:
        raise RuntimeError("Record duration must be > 0 seconds")
    print(f"[rec] recording {duration:.2f}s ...")
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", device=device)
    sd.wait()
    sf.write(path, audio, sample_rate)
    print(f"[rec] saved {path}")


def resample_audio(samples: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr or samples.size == 0:
        return samples
    resampled = librosa.resample(samples, orig_sr=original_sr, target_sr=target_sr)
    return resampled.astype(np.float32)


def compute_fbank(
    samples: np.ndarray,
    sample_rate: int,
    feature_dim: int,
    frame_length_ms: float,
    frame_shift_ms: float,
) -> np.ndarray:
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


def fix_num_frames(feats: np.ndarray, num_frames: int) -> np.ndarray:
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


def resolve_core_mask(value: str) -> int:
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


def rknn_embed(model_path: str, feats: np.ndarray, core_mask: int) -> np.ndarray:
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    x = feats[np.newaxis, :, :].astype(np.float32)
    out = rknn.inference(inputs=[x])[0]
    emb = np.squeeze(out)
    return emb.astype(np.float32)


def format_candidate(candidate: Dict[str, Any]) -> str:
    username = candidate.get("username") or "<unknown>"
    identity = candidate.get("identity") or "-"
    similarity = candidate.get("similarity")
    similarity_str = f"{similarity:.4f}" if isinstance(similarity, (float, int)) else "n/a"
    return f"{username} (id={candidate.get('id')}, identity={identity}) -> similarity={similarity_str}"


def enroll_embedding(user_id: int, embedding: np.ndarray) -> None:
    from app.database import SessionLocal
    from app.models import User

    payload = json.dumps(embedding.tolist())
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).one_or_none()
        if user is None:
            raise RuntimeError(f"User id={user_id} not found")
        user.embedding = payload
        db.add(user)
        db.commit()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_path = args.model_path or cfg.get("model_path")
    if not model_path:
        raise RuntimeError("Model path not provided via --model-path or config file")
    sample_rate = args.sample_rate or cfg.get("sample_rate", 16000)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.6)
    speaker_provider = args.speaker_provider or cfg.get("speaker_provider", "rknn")
    speaker_num_threads = args.speaker_num_threads or cfg.get("speaker_num_threads", 1)
    database_url = args.database_url or cfg.get("database_url")
    if database_url:
        import os
        from app import config as app_config

        os.environ["DATABASE_URL"] = database_url
        app_config.DATABASE_URL = database_url

    if args.record:
        audio_path = Path(args.audio or "tmp/tdnn_record.wav").expanduser()
        record_audio(audio_path, sample_rate, args.duration, args.device)
    else:
        audio_path = Path(args.audio).expanduser()

    audio, original_sr = load_audio(audio_path)
    processed_audio = resample_audio(audio, original_sr, sample_rate)
    if processed_audio.size == 0:
        raise RuntimeError("Audio file does not contain any samples")

    print(f"[speaker] model={model_path}")
    print(f"[speaker] provider={speaker_provider} threads={speaker_num_threads}")
    print(f"[speaker] samples={processed_audio.shape[0]}, duration={processed_audio.shape[0] / sample_rate:.2f}s")

    feats = compute_fbank(
        processed_audio,
        sample_rate,
        feature_dim=args.feature_dim,
        frame_length_ms=args.frame_length_ms,
        frame_shift_ms=args.frame_shift_ms,
    )
    feats = fix_num_frames(feats, args.num_frames)

    emb = rknn_embed(model_path, feats, resolve_core_mask(args.rknn_core))
    if args.l2_normalize:
        denom = np.linalg.norm(emb) + 1e-10
        emb = emb / denom

    print(f"[speaker] embedding_dim={emb.shape[0]}")
    if args.dump_embedding:
        np.set_printoptions(precision=5, suppress=True)
        print("[speaker] embedding=", emb)

    if args.enroll_user_id is not None:
        from app.database import init_db

        init_db()
        enroll_embedding(args.enroll_user_id, emb)
        print(f"[enroll] updated user_id={args.enroll_user_id}")

    if args.skip_identify:
        return

    from app.database import init_db
    from app.services.voice.speaker import identify_user

    init_db()
    matched, similarity, candidates = identify_user(emb, threshold=threshold)
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
