"""
Local microphone streaming test without FastAPI.

Usage:
    python test/ws_test_no_port.py \
        --tokens ./models/.../tokens.txt \
        --encoder ./models/.../encoder.onnx \
        --decoder ./models/.../decoder.onnx \
        --joiner ./models/.../joiner.onnx
"""

import argparse
import json
import queue
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import sherpa_onnx
import sounddevice as sd

DEFAULT_CONFIG = Path("config/app_config_test_rknn.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Local sherpa-onnx mic tester (no FastAPI)")
    parser.add_argument("--tokens", help="Path to tokens.txt")
    parser.add_argument("--encoder", help="Encoder ONNX")
    parser.add_argument("--decoder", help="Decoder ONNX")
    parser.add_argument("--joiner", help="Joiner ONNX")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate")
    parser.add_argument("--feature-dim", type=int, default=80)
    parser.add_argument("--decoding-method", default="modified_beam_search")
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--chunk-ms", type=int, default=100, help="Chunk size from mic (ms)")
    parser.add_argument("--device", default=None, help="sounddevice input device")
    parser.add_argument("--hotwords-file", help="Optional hotwords list")
    parser.add_argument("--hotwords-score", type=float, default=1.5, help="Hotword boosting score (>1.0)")
    parser.add_argument("--rule1", type=float, default=1.2, help="rule1_min_trailing_silence (s)")
    parser.add_argument("--rule2", type=float, default=0.8, help="rule2_min_trailing_silence (s)")
    parser.add_argument("--rule3", type=int, default=300, help="rule3_min_utterance_length (ms)")
    parser.add_argument("--noise-gate", type=float, default=0.01, help="RMS threshold for zeroing quiet chunks (0 to disable)")
    return parser


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config (default: config/app_config.json)")
    return parser


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
        raise RuntimeError(f"Config file must contain a JSON object: {cfg_path}")
    return data


def parse_args() -> argparse.Namespace:
    cfg_parser = config_parser()
    cfg_args, remaining = cfg_parser.parse_known_args()
    cfg = load_config(cfg_args.config)

    parser = build_parser()
    defaults: Dict[str, Any] = {}
    mapping = {
        "tokens": "tokens",
        "encoder": "encoder",
        "decoder": "decoder",
        "joiner": "joiner",
        "sample_rate": "sample_rate",
        "feature_dim": "feature_dim",
        "decoding_method": "decoding_method",
        "max_active_paths": "max_active_paths",
        "provider": "provider",
        "num_threads": "num_threads",
        "hotwords_file":"hotwords_file",
        "hotwords_score":"hotwords_score",
        "rule1": "rule1_min_trailing_silence",
        "rule2": "rule2_min_trailing_silence",
        "rule3": "rule3_min_utterance_length",
    }
    for dest, key in mapping.items():
        if key in cfg:
            defaults[dest] = cfg[key]

    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    required = ["tokens", "encoder", "decoder", "joiner"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)} (provide via CLI or config)")

    return args


class MicSource:
    def __init__(self, sample_rate: int, chunk_ms: int, device: Optional[str], noise_gate: float):
        self.sample_rate = sample_rate
        self.blocksize = int(sample_rate * chunk_ms / 1000)
        self.device = device
        self.noise_gate = max(0.0, min(noise_gate, 1.0))
        self.queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=100)
        self._stream: Optional[sd.RawInputStream] = None

    def __enter__(self):
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=1,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self.queue.put_nowait(None)

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] status: {status}", file=sys.stderr)
        data = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
        if self.noise_gate > 0:
            rms = np.sqrt(np.mean(np.square(data / 32768.0)))
            if rms < self.noise_gate:
                data.fill(0)
        self.queue.put(data.astype(np.int16).tobytes())


def print_final(text: str, start_time: float):
    if not text:
        return
    latency = int((time.perf_counter() - start_time) * 1000)
    print(f"\n[final] {text} (latency={latency}ms)")


def pcm16_to_float(samples: bytes) -> np.ndarray:
    arr = np.frombuffer(samples, dtype=np.int16).astype(np.float32)
    arr /= 32768.0
    return arr


def main():
    args = parse_args()

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=args.feature_dim,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        provider=args.provider,
        rule1_min_trailing_silence=args.rule1,
        rule2_min_trailing_silence=args.rule2,
        rule3_min_utterance_length=args.rule3,
        enable_endpoint_detection=True,
        hotwords_file="./models/hotwords_test.txt",
        hotwords_score=2.0,
        modeling_unit="cjkchar+bpe",
        bpe_vocab="./models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/bpe.vocab",
    )

    stream = recognizer.create_stream()
    last_partial = ""
    utterance_start = time.perf_counter()
    interrupted = False

    def handle_stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[client] stopping…")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_stop)

    with MicSource(args.sample_rate, args.chunk_ms, args.device, args.noise_gate) as mic:
        while True:
            chunk = mic.queue.get()
            if chunk is None:
                break
            samples = pcm16_to_float(chunk)
            stream.accept_waveform(args.sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

                

            text = recognizer.get_result(stream)
            if text and text != last_partial:
                print(f"\r[partial] {text} ", end="", flush=True)
                last_partial = text

            if recognizer.is_endpoint(stream):
                final_text = recognizer.get_result(stream).strip()
                if final_text:
                    print_final(final_text, utterance_start)
                recognizer.reset(stream)
                last_partial = ""
                utterance_start = time.perf_counter()

            if interrupted:
                break

    # flush remaining
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    final_text = recognizer.get_result(stream).strip()
    if final_text:
        print_final(final_text, utterance_start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[client] interrupted", file=sys.stderr)
