#!/usr/bin/env python3
"""
Record microphone audio and store it as a WAV file.

Examples:
    python test/record_wav.py --output tmp/my_voice.wav --duration 5
    python test/record_wav.py --output tmp/my_voice.wav   # press Ctrl+C to stop
"""

import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Microphone recorder")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Destination WAV path (directories are created automatically)",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (Hz)")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 2], help="Number of channels")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 means until Ctrl+C)")
    parser.add_argument("--device", help="Optional sounddevice input device name/index")
    return parser


def record_fixed_duration(output: Path, sample_rate: int, channels: int, duration: float, device: Optional[str]):
    frames = int(duration * sample_rate)
    if frames <= 0:
        raise ValueError("Duration must be > 0 seconds for fixed recording")
    print(f"[rec] recording {duration:.2f}s ...")
    data = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    sf.write(output, data, sample_rate)
    print(f"[rec] saved {output}")


def record_until_interrupt(output: Path, sample_rate: int, channels: int, device: Optional[str]):
    stop_flag = False

    def handle_stop(signum, frame):
        nonlocal stop_flag
        stop_flag = True
        print("\n[rec] stopping...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_stop)

    buffer: list[np.ndarray] = []
    blocksize = int(sample_rate * 0.5)  # 500ms chunks
    print("[rec] recording... press Ctrl+C to stop.")

    with sd.InputStream(
        samplerate=sample_rate,
        blocksize=blocksize,
        channels=channels,
        dtype="float32",
        device=device,
    ) as stream:
        while not stop_flag:
            chunk, _ = stream.read(blocksize)
            buffer.append(chunk.copy())

    if not buffer:
        print("[rec] no audio captured.")
        return

    data = np.concatenate(buffer, axis=0)
    sf.write(output, data, sample_rate)
    print(f"[rec] saved {output} ({data.shape[0] / sample_rate:.2f}s)")


def main():
    args = build_parser().parse_args()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.duration > 0:
        record_fixed_duration(output, args.sample_rate, args.channels, args.duration, args.device)
    else:
        record_until_interrupt(output, args.sample_rate, args.channels, args.device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[rec] interrupted.")
        sys.exit(1)
