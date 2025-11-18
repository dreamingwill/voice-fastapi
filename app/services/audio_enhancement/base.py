from __future__ import annotations

import abc
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

NoiseMode = Literal["none", "classic", "improved"]


class EnhancementConfig(BaseModel):
    """Runtime knobs that control the enhancement pipeline."""

    noise_mode: NoiseMode = "none"
    noise_strength: float = Field(1.0, ge=0.5, le=5.0)
    enable_dereverb: bool = False
    dereverb_delay: int = Field(3, ge=1, le=10)
    dereverb_taps: int = Field(10, ge=2, le=30)
    dereverb_iterations: int = Field(3, ge=1, le=10)

    @property
    def is_passthrough(self) -> bool:
        return (self.noise_mode == "none") and (not self.enable_dereverb)

    def export_metadata(self) -> dict:
        return {
            "noiseMode": self.noise_mode,
            "noiseStrength": self.noise_strength,
            "enableDereverb": self.enable_dereverb,
            "dereverb": {
                "delay": self.dereverb_delay,
                "taps": self.dereverb_taps,
                "iterations": self.dereverb_iterations,
            },
        }


class BaseNoiseReducer(abc.ABC):
    @abc.abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
        """Return the denoised samples; implementers must preserve length."""


class BaseEnhancementModule(abc.ABC):
    @abc.abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        """Return processed samples; implementers must preserve length."""
