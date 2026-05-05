from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowStats:
    valid_pixel_count: int
    mean_magnitude: float
    std_magnitude: float
    median_magnitude: float
    mean_u: float
    mean_v: float
    direction_angle_deg: float | None
    consistency: float


def flow_magnitude_angle(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    magnitude = cv2.magnitude(u.astype(np.float32), v.astype(np.float32))
    angle = np.arctan2(v.astype(np.float32), u.astype(np.float32))
    return magnitude, angle


def road_masked_flow_stats(
    u: np.ndarray,
    v: np.ndarray,
    road_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    min_magnitude: float = 1e-4,
) -> FlowStats:
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    mask = np.ones(u.shape, dtype=bool)
    if road_mask is not None:
        if road_mask.shape != u.shape:
            raise ValueError("road_mask must match flow shape")
        mask &= road_mask > 0
    if valid_mask is not None:
        if valid_mask.shape != u.shape:
            raise ValueError("valid_mask must match flow shape")
        mask &= valid_mask > 0

    magnitude, angle = flow_magnitude_angle(u, v)
    mask &= magnitude >= min_magnitude

    count = int(mask.sum())
    if count == 0:
        return FlowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0)

    selected_mag = magnitude[mask]
    selected_u = u[mask].astype(np.float32)
    selected_v = v[mask].astype(np.float32)
    selected_angle = angle[mask]

    mean_u = float(selected_u.mean())
    mean_v = float(selected_v.mean())
    direction = float(np.degrees(np.arctan2(mean_v, mean_u)))
    mean_sin = float(np.mean(np.sin(selected_angle)))
    mean_cos = float(np.mean(np.cos(selected_angle)))
    consistency = float(np.sqrt(mean_sin * mean_sin + mean_cos * mean_cos))

    return FlowStats(
        valid_pixel_count=count,
        mean_magnitude=float(selected_mag.mean()),
        std_magnitude=float(selected_mag.std()),
        median_magnitude=float(np.median(selected_mag)),
        mean_u=mean_u,
        mean_v=mean_v,
        direction_angle_deg=direction,
        consistency=consistency,
    )

