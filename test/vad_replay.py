#!/usr/bin/env python3
"""
Offline VAD + streaming ASR replay.

Example:
  python test/vad_replay.py --wav ./tmp/noise.wav
  python test/vad_replay.py --wav ./tmp/commands.wav --config config/app_config.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.services.voice.recognizer import create_recognizer
from app.services.voice.vad import EnergyVad


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")
    return data


def _resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or samples.size == 0:
        return samples
    new_len = int(round(samples.size * dst_sr / src_sr))
    if new_len <= 1:
        return samples
    x_old = np.arange(samples.size, dtype=np.float32)
    x_new = np.linspace(0, samples.size - 1, new_len, dtype=np.float32)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Offline VAD + streaming ASR replay")
    parser.add_argument("--wav", required=True, help="Input wav path")
    parser.add_argument("--config", default="config/app_config.json", help="JSON config path")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for ASR/VAD")
    parser.add_argument("--chunk-ms", type=int, default=40, help="Chunk size in ms for replay")
    parser.add_argument("--vad-pre-roll-ms", type=int, default=None)
    parser.add_argument("--vad-post-roll-ms", type=int, default=None)
    parser.add_argument("--vad-snr-open-db", type=float, default=None)
    parser.add_argument("--vad-open-min-ms", type=int, default=None)
    parser.add_argument("--vad-end-silence-ms", type=int, default=None)
    parser.add_argument("--vad-max-utterance-ms", type=int, default=None)
    parser.add_argument("--vad-reopen-min-ms", type=int, default=None)
    parser.add_argument("--vad-noise-margin-db", type=float, default=None)
    parser.add_argument("--vad-noise-bootstrap-ms", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = _load_config(args.config)

    wav_path = Path(args.wav).expanduser()
    samples, sr = sf.read(wav_path, dtype="float32")
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1).astype(np.float32)

    target_sr = int(cfg.get("sample_rate") or args.sample_rate)
    if sr != target_sr:
        samples = _resample_linear(samples, sr, target_sr)
        sr = target_sr

    recognizer = create_recognizer(
        tokens=cfg.get("tokens", ""),
        encoder=cfg.get("encoder", ""),
        decoder=cfg.get("decoder", ""),
        joiner=cfg.get("joiner", ""),
        num_threads=int(cfg.get("num_threads", 4)),
        sample_rate=target_sr,
        feature_dim=int(cfg.get("feature_dim", 80)),
        decoding_method=cfg.get("decoding_method", "greedy_search"),
        max_active_paths=int(cfg.get("max_active_paths", 4)),
        provider=cfg.get("provider", "cpu"),
        hotwords_file=cfg.get("hotwords_file", ""),
        hotwords_score=float(cfg.get("hotwords_score", 1.5)),
        blank_penalty=float(cfg.get("blank_penalty", 0.0)),
        hr_rule_fsts=cfg.get("hr_rule_fsts", ""),
        hr_lexicon=cfg.get("hr_lexicon", ""),
        rule1_min_trailing_silence=float(cfg.get("rule1_min_trailing_silence", 0.8)),
        rule2_min_trailing_silence=float(cfg.get("rule2_min_trailing_silence", 0.4)),
        rule3_min_utterance_length=int(cfg.get("rule3_min_utterance_length", 15)),
    )
    stream = recognizer.create_stream()

    vad = EnergyVad(
        sample_rate=target_sr,
        pre_roll_ms=int(args.vad_pre_roll_ms or cfg.get("vad_pre_roll_ms", 300)),
        post_roll_ms=int(args.vad_post_roll_ms or cfg.get("vad_post_roll_ms", 700)),
        snr_open_db=float(args.vad_snr_open_db or cfg.get("vad_snr_open_db", 10.0)),
        open_min_ms=int(args.vad_open_min_ms or cfg.get("vad_open_min_ms", 120)),
        end_silence_ms=int(args.vad_end_silence_ms or cfg.get("vad_end_silence_ms", 900)),
        max_utterance_ms=int(args.vad_max_utterance_ms or cfg.get("vad_max_utterance_ms", 0)) or None,
        reopen_min_ms=int(args.vad_reopen_min_ms or cfg.get("vad_reopen_min_ms", 120)),
        noise_update_margin_db=float(args.vad_noise_margin_db or cfg.get("vad_noise_margin_db", 3.0)),
        noise_bootstrap_ms=int(args.vad_noise_bootstrap_ms or cfg.get("vad_noise_bootstrap_ms", 1000)),
    )

    chunk_samples = max(1, int(sr * args.chunk_ms / 1000))
    total_samples = 0
    last_partial = ""
    segments: List[Tuple[int, int, str]] = []
    seg_start_sample: Optional[int] = None

    for i in range(0, samples.size, chunk_samples):
        chunk = samples[i : i + chunk_samples]
        total_samples += chunk.size
        vad_result = vad.process(chunk)

        if vad_result.started:
            recognizer.reset(stream)
            last_partial = ""
            seg_start_sample = vad_result.start_sample
            if seg_start_sample is not None:
                print(f"[vad] start @ {int(seg_start_sample * 1000 / sr)} ms")

        if vad_result.feed_samples:
            feed = (
                np.concatenate(vad_result.feed_samples, axis=0)
                if len(vad_result.feed_samples) > 1
                else vad_result.feed_samples[0]
            )
            if feed.size > 0:
                stream.accept_waveform(sr, feed)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                text = recognizer.get_result(stream)
                if text and text != last_partial:
                    print(f"[partial] {int(total_samples * 1000 / sr)} ms: {text}")
                    last_partial = text

        if vad_result.ended:
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            final_text = recognizer.get_result(stream).strip()
            end_sample = vad_result.end_sample or total_samples
            start_sample = seg_start_sample or 0
            print(
                f"[final] {int(start_sample * 1000 / sr)}-{int(end_sample * 1000 / sr)} ms: {final_text}"
            )
            segments.append((int(start_sample * 1000 / sr), int(end_sample * 1000 / sr), final_text))
            recognizer.reset(stream)
            last_partial = ""
            seg_start_sample = None

    print("\nSegments:")
    for idx, (start_ms, end_ms, text) in enumerate(segments, 1):
        print(f"  {idx:02d}. {start_ms}-{end_ms} ms: {text}")


if __name__ == "__main__":
    main()
