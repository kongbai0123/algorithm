from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_ops import as_float_gray


@dataclass(frozen=True)
class LucasKanadeConfig:
    max_corners: int = 300
    quality_level: float = 0.01
    min_distance: int = 7
    block_size: int = 7
    win_size: int = 21
    max_level: int = 3
    criteria_count: int = 30
    criteria_epsilon: float = 0.01

    def validate(self) -> None:
        if self.max_corners < 1:
            raise ValueError("max_corners must be >= 1")
        if not 0.0 < self.quality_level < 1.0:
            raise ValueError("quality_level must be between 0 and 1")
        if self.min_distance < 1:
            raise ValueError("min_distance must be >= 1")
        if self.block_size < 3:
            raise ValueError("block_size must be >= 3")
        if self.win_size < 3 or self.win_size % 2 == 0:
            raise ValueError("win_size must be an odd integer >= 3")
        if self.max_level < 0:
            raise ValueError("max_level must be >= 0")
        if self.criteria_count < 1:
            raise ValueError("criteria_count must be >= 1")
        if self.criteria_epsilon <= 0:
            raise ValueError("criteria_epsilon must be positive")


def lucas_kanade_sparse_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    mask: np.ndarray | None = None,
    config: LucasKanadeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = config or LucasKanadeConfig()
    cfg.validate()

    previous = (as_float_gray(frame1) * 255.0).astype(np.uint8)
    current = (as_float_gray(frame2) * 255.0).astype(np.uint8)
    if previous.shape != current.shape:
        raise ValueError("frame1 and frame2 must have the same shape")

    feature_mask = None
    if mask is not None:
        if mask.shape != previous.shape:
            raise ValueError("mask must match frame shape")
        feature_mask = (mask > 0).astype(np.uint8) * 255

    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
        mask=feature_mask,
        blockSize=cfg.block_size,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(cfg.win_size, cfg.win_size),
        maxLevel=cfg.max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.criteria_count, cfg.criteria_epsilon),
    )
    if next_points is None or status is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    valid = status.reshape(-1) == 1
    start = points.reshape(-1, 2)[valid].astype(np.float32)
    end = next_points.reshape(-1, 2)[valid].astype(np.float32)
    errors = error.reshape(-1)[valid].astype(np.float32) if error is not None else np.zeros((start.shape[0],), dtype=np.float32)
    return start, end, errors


def sparse_points_to_flow(
    start_points: np.ndarray,
    end_points: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    u = np.zeros((height, width), dtype=np.float32)
    v = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.uint8)

    for start, end in zip(start_points, end_points):
        x = int(round(float(start[0])))
        y = int(round(float(start[1])))
        if 0 <= x < width and 0 <= y < height:
            u[y, x] = float(end[0] - start[0])
            v[y, x] = float(end[1] - start[1])
            valid[y, x] = 255

    return u, v, valid

