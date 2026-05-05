from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_ops import build_gaussian_pyramid, resize_flow, spatial_temporal_gradients, warp_bilinear


@dataclass(frozen=True)
class HornSchunckConfig:
    alpha: float = 15.0
    iterations: int = 300
    tolerance: float = 1e-5
    pyramid_levels: int = 4
    warps_per_level: int = 3
    relaxation: float = 1.0
    epsilon: float = 1e-6

    def validate(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.pyramid_levels < 1:
            raise ValueError("pyramid_levels must be >= 1")
        if self.warps_per_level < 1:
            raise ValueError("warps_per_level must be >= 1")
        if not 0.0 < self.relaxation < 2.0:
            raise ValueError("relaxation must be between 0 and 2")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def _local_average(field: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(field, (5, 5), 0, borderType=cv2.BORDER_REFLECT101)


def horn_schunck(
    frame1: np.ndarray,
    frame2: np.ndarray,
    config: HornSchunckConfig | None = None,
    initial_u: np.ndarray | None = None,
    initial_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or HornSchunckConfig()
    cfg.validate()

    ix, iy, it = spatial_temporal_gradients(frame1, frame2)
    if initial_u is None:
        u = np.zeros_like(ix, dtype=np.float32)
    else:
        u = np.asarray(initial_u, dtype=np.float32).copy()
    if initial_v is None:
        v = np.zeros_like(iy, dtype=np.float32)
    else:
        v = np.asarray(initial_v, dtype=np.float32).copy()
    if u.shape != ix.shape or v.shape != ix.shape:
        raise ValueError("initial flow fields must match frame shape")

    denominator = cfg.alpha * cfg.alpha + ix * ix + iy * iy + cfg.epsilon
    for _ in range(cfg.iterations):
        previous_u = u.copy()
        previous_v = v.copy()
        avg_u = _local_average(u)
        avg_v = _local_average(v)
        residual = ix * avg_u + iy * avg_v + it
        candidate_u = avg_u - ix * residual / denominator
        candidate_v = avg_v - iy * residual / denominator
        u = u + cfg.relaxation * (candidate_u - u)
        v = v + cfg.relaxation * (candidate_v - v)

        delta = np.linalg.norm(u - previous_u) + np.linalg.norm(v - previous_v)
        if delta / u.size < cfg.tolerance:
            break

    return u.astype(np.float32), v.astype(np.float32)


def multiresolution_horn_schunck(
    frame1: np.ndarray,
    frame2: np.ndarray,
    config: HornSchunckConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or HornSchunckConfig()
    cfg.validate()

    pyramid1 = build_gaussian_pyramid(frame1, cfg.pyramid_levels)
    pyramid2 = build_gaussian_pyramid(frame2, cfg.pyramid_levels)
    levels = min(len(pyramid1), len(pyramid2))

    coarsest_shape = pyramid1[levels - 1].shape
    u = np.zeros(coarsest_shape, dtype=np.float32)
    v = np.zeros(coarsest_shape, dtype=np.float32)

    for level in range(levels - 1, -1, -1):
        current1 = pyramid1[level]
        current2 = pyramid2[level]
        if u.shape != current1.shape:
            u, v = resize_flow(u, v, current1.shape)

        for _ in range(cfg.warps_per_level):
            warped2 = warp_bilinear(current2, u, v)
            du, dv = horn_schunck(current1, warped2, cfg)
            u = u + du
            v = v + dv

    return u.astype(np.float32), v.astype(np.float32)
