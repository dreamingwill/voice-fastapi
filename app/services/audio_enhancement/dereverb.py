from __future__ import annotations

import logging

import numpy as np
import pyroomacoustics as pra

from .base import BaseEnhancementModule, EnhancementConfig

logger = logging.getLogger("audio.enhancement.dereverb")


class DereverbWPE(BaseEnhancementModule):
    """Wrapper around pyroomacoustics WPE implementation."""

    def __init__(self, block_size: int = 1024):
        self.block_size = block_size

    def process(self, audio: np.ndarray, sample_rate: int, config: EnhancementConfig | None = None, **_: object) -> np.ndarray:
        if audio.size < self.block_size:
            return audio

        cfg = config or EnhancementConfig()
        x = np.expand_dims(audio.astype(np.float32), axis=0)
        try:
            enhanced = pra.dereverberation.wpe.wpe(
                x,
                taps=cfg.dereverb_taps,
                delay=cfg.dereverb_delay,
                iterations=cfg.dereverb_iterations,
            )
        except Exception as exc:
            logger.warning("dereverb failed: %s", exc)
            return audio
        return enhanced.squeeze(0).astype(np.float32)
