from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_ops import as_float_gray


@dataclass(frozen=True)
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 4
    winsize: int = 21
    iterations: int = 5
    poly_n: int = 7
    poly_sigma: float = 1.5
    flags: int = 0

    def validate(self) -> None:
        if not 0.0 < self.pyr_scale < 1.0:
            raise ValueError("pyr_scale must be between 0 and 1")
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
        if self.winsize < 3 or self.winsize % 2 == 0:
            raise ValueError("winsize must be an odd integer >= 3")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.poly_n < 5 or self.poly_n % 2 == 0:
            raise ValueError("poly_n must be an odd integer >= 5")
        if self.poly_sigma <= 0:
            raise ValueError("poly_sigma must be positive")


def farneback_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    config: FarnebackConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or FarnebackConfig()
    cfg.validate()

    previous = (as_float_gray(frame1) * 255.0).round().astype(np.uint8)
    current = (as_float_gray(frame2) * 255.0).round().astype(np.uint8)
    if previous.shape != current.shape:
        raise ValueError("frame1 and frame2 must have the same shape")

    flow = cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        cfg.pyr_scale,
        cfg.levels,
        cfg.winsize,
        cfg.iterations,
        cfg.poly_n,
        cfg.poly_sigma,
        cfg.flags,
    )
    return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)
