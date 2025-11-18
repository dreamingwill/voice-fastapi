from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from .base import BaseNoiseReducer

logger = logging.getLogger("audio.enhancement.spectral")


@dataclass
class _SpectralParams:
    n_fft: int = 512
    hop_length: int = 128
    noise_frames: int = 6
    beta: float = 0.01  # spectral floor


class ClassicSpectralSubtraction(BaseNoiseReducer):
    """Simple over-subtraction spectral subtraction."""

    def __init__(self, params: _SpectralParams | None = None):
        self.params = params or _SpectralParams()

    def process(self, audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
        if audio.size < self.params.n_fft:
            return audio

        strength = max(0.5, min(strength, 5.0))
        stft = librosa.stft(
            audio,
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
            win_length=self.params.n_fft,
            center=True,
        )
        magnitude = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        nf = min(self.params.noise_frames, magnitude.shape[1]) or 1
        noise_mag = np.mean(magnitude[:, :nf], axis=1, keepdims=True)

        power = np.maximum(magnitude**2 - strength * (noise_mag**2), self.params.beta * (noise_mag**2))
        denoised = np.sqrt(power) * phase
        enhanced = librosa.istft(denoised, hop_length=self.params.hop_length, length=audio.size)
        return enhanced.astype(np.float32)


class ImprovedSpectralSubtraction(BaseNoiseReducer):
    """Adds SNR-adaptive gain to reduce musical noise artifacts."""

    def __init__(
        self,
        params: _SpectralParams | None = None,
        alpha_min: float = 0.7,
        alpha_max: float = 4.5,
        snr_smoothing: float = 0.98,
    ):
        self.params = params or _SpectralParams()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.snr_smoothing = snr_smoothing

    def process(self, audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
        if audio.size < self.params.n_fft:
            return audio

        stft = librosa.stft(
            audio,
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
            win_length=self.params.n_fft,
            center=True,
        )
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))

        nf = min(self.params.noise_frames, mag.shape[1]) or 1
        noise_psd = np.mean(mag[:, :nf] ** 2, axis=1, keepdims=True)
        wideband_energy = np.mean(mag**2, axis=1, keepdims=True)
        noise_psd = self.snr_smoothing * noise_psd + (1 - self.snr_smoothing) * wideband_energy

        snr = mag**2 / (noise_psd + 1e-8)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-snr)
        alpha *= strength

        power = np.maximum(mag**2 - alpha * noise_psd, self.params.beta * noise_psd)
        denoised = np.sqrt(power) * phase
        enhanced = librosa.istft(denoised, hop_length=self.params.hop_length, length=audio.size)
        return enhanced.astype(np.float32)
