from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .horn_schunck import HornSchunckConfig, multiresolution_horn_schunck
from .image_ops import warp_bilinear
from .road_detection import RoadDetectionConfig, detect_road, make_road_overlay


@dataclass(frozen=True)
class FlowRoadFusionConfig:
    static_weight: float = 0.62
    motion_weight: float = 0.23
    temporal_weight: float = 0.15
    fused_threshold: float = 0.42
    motion_sigma_floor: float = 0.15
    angle_concentration: float = 2.0
    min_component_ratio: float = 0.02
    close_kernel_size: int = 17
    open_kernel_size: int = 7

    def validate(self) -> None:
        weights = np.array([self.static_weight, self.motion_weight, self.temporal_weight], dtype=np.float32)
        if np.any(weights < 0):
            raise ValueError("fusion weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("at least one fusion weight must be positive")
        if not 0.0 < self.fused_threshold < 1.0:
            raise ValueError("fused_threshold must be between 0 and 1")
        if self.motion_sigma_floor <= 0:
            raise ValueError("motion_sigma_floor must be positive")
        if self.angle_concentration <= 0:
            raise ValueError("angle_concentration must be positive")
        if not 0.0 < self.min_component_ratio < 1.0:
            raise ValueError("min_component_ratio must be between 0 and 1")
        if self.close_kernel_size < 3 or self.close_kernel_size % 2 == 0:
            raise ValueError("close_kernel_size must be an odd integer >= 3")
        if self.open_kernel_size < 3 or self.open_kernel_size % 2 == 0:
            raise ValueError("open_kernel_size must be an odd integer >= 3")


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _normalize01(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo < epsilon:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + epsilon)).astype(np.float32)


def _seed_mask_from_road_config(height: int, width: int, config: RoadDetectionConfig) -> np.ndarray:
    center_x = width / 2.0
    top_y = int(round(height * config.seed_top_ratio))
    bottom_y = height - 1
    half_bottom = width * config.seed_bottom_width_ratio / 2.0
    half_top = width * config.seed_top_width_ratio / 2.0
    polygon = np.array(
        [
            [center_x - half_bottom, bottom_y],
            [center_x - half_top, top_y],
            [center_x + half_top, top_y],
            [center_x + half_bottom, bottom_y],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.reshape((-1, 1, 2)), 255)
    return mask


def _largest_bottom_component(mask: np.ndarray, min_component_ratio: float) -> np.ndarray:
    height, width = mask.shape
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if component_count <= 1:
        return mask.astype(np.uint8)

    center_band = np.zeros_like(mask, dtype=np.uint8)
    band_width = max(8, width // 8)
    center_x = width // 2
    center_band[int(height * 0.82) :, max(0, center_x - band_width) : min(width, center_x + band_width)] = 255
    min_area = int(height * width * min_component_ratio)

    best_label = 0
    best_area = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        touches_bottom_center = np.any((labels == label) & (center_band > 0))
        if touches_bottom_center and area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        return mask.astype(np.uint8)
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _motion_consistency_score(
    u: np.ndarray,
    v: np.ndarray,
    seed_mask: np.ndarray,
    config: FlowRoadFusionConfig,
) -> np.ndarray:
    magnitude = cv2.magnitude(u.astype(np.float32), v.astype(np.float32))
    angle = np.arctan2(v.astype(np.float32), u.astype(np.float32))

    seed = seed_mask > 0
    if int(seed.sum()) < 10:
        return np.ones_like(magnitude, dtype=np.float32)

    seed_mag = magnitude[seed]
    mag_mu = float(np.median(seed_mag))
    mag_sigma = max(float(np.std(seed_mag)), config.motion_sigma_floor)
    magnitude_score = np.exp(-0.5 * ((magnitude - mag_mu) / mag_sigma) ** 2)

    seed_angle = angle[seed]
    mean_sin = float(np.mean(np.sin(seed_angle)))
    mean_cos = float(np.mean(np.cos(seed_angle)))
    mean_angle = float(np.arctan2(mean_sin, mean_cos))
    angular_similarity = np.cos(angle - mean_angle)
    angle_score = np.exp(config.angle_concentration * (angular_similarity - 1.0))
    return np.clip(magnitude_score * angle_score, 0.0, 1.0).astype(np.float32)


def _postprocess_mask(score: np.ndarray, threshold: float, config: FlowRoadFusionConfig) -> np.ndarray:
    raw = np.where(score >= threshold, 255, 0).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.close_kernel_size, config.close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.open_kernel_size, config.open_kernel_size))
    mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return _largest_bottom_component(mask, config.min_component_ratio)


def detect_road_with_optical_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    road_config: RoadDetectionConfig | None = None,
    flow_config: HornSchunckConfig | None = None,
    fusion_config: FlowRoadFusionConfig | None = None,
    prev_score: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    road_cfg = road_config or RoadDetectionConfig()
    flow_cfg = flow_config or HornSchunckConfig(alpha=8.0, iterations=180, pyramid_levels=4, warps_per_level=3)
    fusion_cfg = fusion_config or FlowRoadFusionConfig()
    road_cfg.validate()
    flow_cfg.validate()
    fusion_cfg.validate()

    prev = _ensure_bgr(prev_frame)
    curr = _ensure_bgr(curr_frame)
    if prev.shape != curr.shape:
        raise ValueError("prev_frame and curr_frame must have identical shape")

    static_mask, static_score = detect_road(curr, road_cfg)
    static_score = np.clip(static_score.astype(np.float32), 0.0, 1.0)
    u, v = multiresolution_horn_schunck(prev, curr, flow_cfg)

    height, width = static_score.shape
    seed_mask = _seed_mask_from_road_config(height, width, road_cfg)
    motion_score = _motion_consistency_score(u, v, seed_mask, fusion_cfg)

    if prev_score is not None:
        previous = np.asarray(prev_score, dtype=np.float32)
        if previous.shape != static_score.shape:
            raise ValueError("prev_score must have the same HxW shape as the frames")
        back_u, back_v = multiresolution_horn_schunck(curr, prev, flow_cfg)
        temporal_score = np.clip(warp_bilinear(previous, back_u, back_v), 0.0, 1.0)
    else:
        temporal_score = static_score.copy()

    weights = np.array(
        [fusion_cfg.static_weight, fusion_cfg.motion_weight, fusion_cfg.temporal_weight],
        dtype=np.float32,
    )
    weights /= float(weights.sum())
    fused_score = weights[0] * static_score + weights[1] * motion_score + weights[2] * temporal_score
    fused_score = cv2.GaussianBlur(np.clip(fused_score.astype(np.float32), 0.0, 1.0), (7, 7), 0)
    fused_mask = _postprocess_mask(fused_score, fusion_cfg.fused_threshold, fusion_cfg)

    debug = {
        "static_mask": static_mask,
        "static_score": static_score,
        "motion_score": motion_score,
        "temporal_score": temporal_score,
        "flow_u": u,
        "flow_v": v,
        "flow_magnitude": cv2.magnitude(u, v),
        "seed_mask": seed_mask,
        "fusion_weights": weights,
    }
    return fused_mask, fused_score, debug


def make_flow_road_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return make_road_overlay(image, mask)


def save_fusion_debug_outputs(
    output_dir: str | Path,
    curr_frame: np.ndarray,
    fused_mask: np.ndarray,
    fused_score: np.ndarray,
    debug: dict[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    curr = _ensure_bgr(curr_frame)
    cv2.imwrite(str(out / "01_fused_overlay.png"), make_flow_road_overlay(curr, fused_mask))
    cv2.imwrite(str(out / "02_fused_mask.png"), fused_mask)
    cv2.imwrite(str(out / "03_fused_score.png"), (_normalize01(fused_score) * 255).astype(np.uint8))
    cv2.imwrite(str(out / "04_static_score.png"), (_normalize01(debug["static_score"]) * 255).astype(np.uint8))
    cv2.imwrite(str(out / "05_motion_score.png"), (_normalize01(debug["motion_score"]) * 255).astype(np.uint8))
    cv2.imwrite(str(out / "06_temporal_score.png"), (_normalize01(debug["temporal_score"]) * 255).astype(np.uint8))
    cv2.imwrite(str(out / "07_flow_magnitude.png"), (_normalize01(debug["flow_magnitude"]) * 255).astype(np.uint8))


def detect_fused_road_from_paths(
    prev_image_path: str | Path,
    curr_image_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prev = cv2.imread(str(prev_image_path))
    curr = cv2.imread(str(curr_image_path))
    if prev is None:
        raise FileNotFoundError(f"Unable to read previous image: {prev_image_path}")
    if curr is None:
        raise FileNotFoundError(f"Unable to read current image: {curr_image_path}")
    mask, score, debug = detect_road_with_optical_flow(prev, curr)
    if output_dir is not None:
        save_fusion_debug_outputs(output_dir, curr, mask, score, debug)
    return mask, score, debug

