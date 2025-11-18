from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from .base import BaseNoiseReducer, EnhancementConfig
from .dereverb import DereverbWPE
from .spectral import ClassicSpectralSubtraction, ImprovedSpectralSubtraction

logger = logging.getLogger("audio.enhancement.pipeline")


class AudioEnhancementPipeline:
    def __init__(self):

        self._noise_modules: Dict[str, BaseNoiseReducer] = {
            "classic": ClassicSpectralSubtraction(),
            "improved": ImprovedSpectralSubtraction(),
        }
        self._dereverb = DereverbWPE()

    def process(self, audio: np.ndarray, sample_rate: int, config: EnhancementConfig | None) -> np.ndarray:
        if config is None or config.is_passthrough or audio.size == 0:
            return audio

        enhanced = audio

        reducer = self._noise_modules.get(config.noise_mode)
        if reducer is not None:
            try:
                enhanced = reducer.process(enhanced, sample_rate, config.noise_strength)
            except Exception as exc:
                logger.warning("noise reducer %s failed: %s", config.noise_mode, exc)

        if config.enable_dereverb:
            try:
                enhanced = self._dereverb.process(enhanced, sample_rate, config=config)
            except Exception as exc:
                logger.warning("dereverb failed: %s", exc)

        return enhanced.astype(np.float32)
