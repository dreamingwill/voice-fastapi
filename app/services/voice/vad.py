import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class VadTransition:
    prev_state: str
    new_state: str
    reason: str
    snr_db: float
    total_ms: int
    utt_ms: int
    dropped_ms: int


@dataclass
class VadResult:
    feed_samples: List[np.ndarray]
    transitions: List[VadTransition]
    started: bool
    ended: bool
    end_reason: Optional[str]
    start_sample: Optional[int]
    end_sample: Optional[int]
    snr_db: float
    utt_ms: int
    dropped_samples: int
    fed_samples: int
    total_samples: int


class EnergyVad:
    def __init__(
        self,
        *,
        sample_rate: int,
        pre_roll_ms: int = 300,
        post_roll_ms: int = 700,
        snr_open_db: float = 10.0,
        open_min_ms: int = 120,
        end_silence_ms: int = 900,
        max_utterance_ms: Optional[int] = None,
        frame_ms: int = 20,
        noise_ema_alpha: float = 0.95,
        min_noise_floor: float = 1e-4,
    ):
        self.sample_rate = int(sample_rate)
        self.pre_roll_ms = int(pre_roll_ms)
        self.post_roll_ms = int(post_roll_ms)
        self.snr_open_db = float(snr_open_db)
        self.open_min_ms = int(open_min_ms)
        self.end_silence_ms = int(end_silence_ms)
        self.max_utterance_ms = int(max_utterance_ms) if max_utterance_ms else None
        self.frame_ms = int(frame_ms)
        self.noise_ema_alpha = float(noise_ema_alpha)
        self.min_noise_floor = float(min_noise_floor)

        self.frame_samples = max(1, int(self.sample_rate * self.frame_ms / 1000))
        self._pre_roll_max_samples = max(1, int(self.sample_rate * self.pre_roll_ms / 1000))
        self._post_roll_samples = max(1, int(self.sample_rate * self.post_roll_ms / 1000))
        self._open_min_samples = max(1, int(self.sample_rate * self.open_min_ms / 1000))
        self._end_silence_samples = max(1, int(self.sample_rate * self.end_silence_ms / 1000))
        self._max_utt_samples = (
            max(1, int(self.sample_rate * self.max_utterance_ms / 1000))
            if self.max_utterance_ms
            else None
        )

        self._state = "IDLE"
        self._noise_floor = self.min_noise_floor
        self._open_samples = 0
        self._silence_samples = 0
        self._tail_samples = 0
        self._utt_samples = 0
        self._total_samples = 0
        self._fed_samples_total = 0
        self._frame_buffer = np.zeros(0, dtype=np.float32)
        self._pre_roll: Deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._last_snr_db = 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def _push_pre_roll(self, frame: np.ndarray) -> None:
        if frame.size == 0:
            return
        self._pre_roll.append(frame)
        self._pre_roll_samples += frame.size
        while self._pre_roll_samples > self._pre_roll_max_samples and self._pre_roll:
            dropped = self._pre_roll.popleft()
            self._pre_roll_samples -= dropped.size

    def _drain_pre_roll(self) -> np.ndarray:
        if not self._pre_roll:
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate(list(self._pre_roll), axis=0)
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        return buf

    def _update_noise_floor(self, rms: float) -> None:
        alpha = self.noise_ema_alpha
        updated = alpha * self._noise_floor + (1.0 - alpha) * rms
        self._noise_floor = max(self.min_noise_floor, float(updated))

    def _snr_db(self, rms: float) -> float:
        denom = max(self._noise_floor, self.min_noise_floor)
        snr = 20.0 * math.log10((rms + 1e-12) / (denom + 1e-12))
        return float(snr)

    def process(self, samples: np.ndarray) -> VadResult:
        if samples.size == 0:
            return VadResult(
                feed_samples=[],
                transitions=[],
                started=False,
                ended=False,
                end_reason=None,
                start_sample=None,
                end_sample=None,
                snr_db=self._last_snr_db,
                utt_ms=int(self._utt_samples * 1000 / self.sample_rate),
                dropped_samples=0,
                fed_samples=0,
                total_samples=0,
            )

        if self._frame_buffer.size > 0:
            samples = np.concatenate([self._frame_buffer, samples], axis=0)
            self._frame_buffer = np.zeros(0, dtype=np.float32)

        feed_samples: List[np.ndarray] = []
        transitions: List[VadTransition] = []
        started = False
        ended = False
        end_reason: Optional[str] = None
        start_sample: Optional[int] = None
        end_sample: Optional[int] = None
        fed_samples = 0

        idx = 0
        total_in = samples.size
        while idx + self.frame_samples <= total_in:
            frame = samples[idx : idx + self.frame_samples]
            idx += self.frame_samples
            self._total_samples += frame.size
            rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
            snr_db = self._snr_db(rms)
            self._last_snr_db = snr_db

            if self._state == "IDLE":
                self._update_noise_floor(rms)
                self._push_pre_roll(frame)
                if snr_db >= self.snr_open_db:
                    self._open_samples += frame.size
                else:
                    self._open_samples = 0
                if self._open_samples >= self._open_min_samples:
                    pre_roll = self._drain_pre_roll()
                    if pre_roll.size > 0:
                        feed_samples.append(pre_roll)
                        fed_samples += pre_roll.size
                        self._utt_samples += pre_roll.size
                    prev_state = self._state
                    self._state = "SPEECH"
                    started = True
                    start_sample = self._total_samples - (pre_roll.size if pre_roll.size > 0 else frame.size)
                    transitions.append(
                        VadTransition(
                            prev_state=prev_state,
                            new_state="SPEECH",
                            reason="snr_open",
                            snr_db=snr_db,
                            total_ms=int(self._total_samples * 1000 / self.sample_rate),
                            utt_ms=int(self._utt_samples * 1000 / self.sample_rate),
                            dropped_ms=int(
                                max(0, self._total_samples - self._fed_samples_total - fed_samples)
                                * 1000
                                / self.sample_rate
                            ),
                        )
                    )
                    self._open_samples = 0
                    self._silence_samples = 0
                    self._tail_samples = 0
                continue

            if self._state in {"SPEECH", "TAIL"}:
                feed_samples.append(frame)
                fed_samples += frame.size
                self._utt_samples += frame.size

                if self._state == "SPEECH":
                    if snr_db < self.snr_open_db:
                        self._silence_samples += frame.size
                    else:
                        self._silence_samples = 0
                    if self._silence_samples >= self._end_silence_samples:
                        prev_state = self._state
                        self._state = "TAIL"
                        self._tail_samples = 0
                        transitions.append(
                            VadTransition(
                                prev_state=prev_state,
                                new_state="TAIL",
                                reason="end_silence",
                                snr_db=snr_db,
                                total_ms=int(self._total_samples * 1000 / self.sample_rate),
                                utt_ms=int(self._utt_samples * 1000 / self.sample_rate),
                                dropped_ms=int(
                                    max(0, self._total_samples - self._fed_samples_total - fed_samples)
                                    * 1000
                                    / self.sample_rate
                                ),
                            )
                        )

                if self._state == "TAIL":
                    self._tail_samples += frame.size
                    if self._tail_samples >= self._post_roll_samples:
                        prev_state = self._state
                        self._state = "IDLE"
                        ended = True
                        end_reason = "end_silence"
                        end_sample = self._total_samples
                        transitions.append(
                            VadTransition(
                                prev_state=prev_state,
                                new_state="END",
                                reason="post_roll",
                                snr_db=snr_db,
                                total_ms=int(self._total_samples * 1000 / self.sample_rate),
                                utt_ms=int(self._utt_samples * 1000 / self.sample_rate),
                                dropped_ms=int(
                                    max(0, self._total_samples - self._fed_samples_total - fed_samples)
                                    * 1000
                                    / self.sample_rate
                                ),
                            )
                        )
                        self._open_samples = 0
                        self._silence_samples = 0
                        self._tail_samples = 0
                        self._utt_samples = 0

                if self._max_utt_samples and self._utt_samples >= self._max_utt_samples:
                    prev_state = self._state
                    self._state = "IDLE"
                    ended = True
                    end_reason = "max_utterance"
                    end_sample = self._total_samples
                    transitions.append(
                        VadTransition(
                            prev_state=prev_state,
                            new_state="END",
                            reason="max_utterance",
                            snr_db=snr_db,
                            total_ms=int(self._total_samples * 1000 / self.sample_rate),
                            utt_ms=int(self._utt_samples * 1000 / self.sample_rate),
                            dropped_ms=int(
                                max(0, self._total_samples - self._fed_samples_total - fed_samples)
                                * 1000
                                / self.sample_rate
                            ),
                        )
                    )
                    self._open_samples = 0
                    self._silence_samples = 0
                    self._tail_samples = 0
                    self._utt_samples = 0
                continue

        if idx < total_in:
            self._frame_buffer = samples[idx:]

        self._fed_samples_total += fed_samples

        dropped_samples = max(0, self._total_samples - self._fed_samples_total)
        return VadResult(
            feed_samples=feed_samples,
            transitions=transitions,
            started=started,
            ended=ended,
            end_reason=end_reason,
            start_sample=start_sample,
            end_sample=end_sample,
            snr_db=self._last_snr_db,
            utt_ms=int(self._utt_samples * 1000 / self.sample_rate),
            dropped_samples=dropped_samples,
            fed_samples=fed_samples,
            total_samples=samples.size,
        )


__all__ = ["EnergyVad", "VadResult", "VadTransition"]
