from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RoadDetectionConfig:
    seed_top_ratio: float = 0.40
    seed_bottom_width_ratio: float = 0.78
    seed_top_width_ratio: float = 0.14
    expansion_bottom_width_ratio: float = 1.0
    expansion_top_width_ratio: float = 0.62
    color_sigma_floor: float = 10.0
    score_threshold: float = 0.34
    expansion_threshold: float = 0.20
    min_component_ratio: float = 0.02
    row_max_step_ratio: float = 0.16

    def validate(self) -> None:
        if not 0.0 < self.seed_top_ratio < 1.0:
            raise ValueError("seed_top_ratio must be between 0 and 1")
        if not 0.0 < self.seed_top_width_ratio <= self.seed_bottom_width_ratio <= 1.0:
            raise ValueError("seed width ratios must satisfy 0 < top <= bottom <= 1")
        if not 0.0 < self.expansion_top_width_ratio <= self.expansion_bottom_width_ratio <= 1.0:
            raise ValueError("expansion width ratios must satisfy 0 < top <= bottom <= 1")
        if self.color_sigma_floor <= 0:
            raise ValueError("color_sigma_floor must be positive")
        if not 0.0 < self.score_threshold < 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if not 0.0 < self.expansion_threshold < self.score_threshold:
            raise ValueError("expansion_threshold must be between 0 and score_threshold")
        if not 0.0 < self.min_component_ratio < 1.0:
            raise ValueError("min_component_ratio must be between 0 and 1")
        if not 0.0 < self.row_max_step_ratio < 0.5:
            raise ValueError("row_max_step_ratio must be between 0 and 0.5")


def _ensure_color(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected a BGR color image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _seed_polygon(height: int, width: int, config: RoadDetectionConfig) -> np.ndarray:
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
    return polygon.reshape((-1, 1, 2))


def _seed_mask(height: int, width: int, config: RoadDetectionConfig) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, _seed_polygon(height, width, config), 255)
    return mask


def _expansion_corridor_mask(height: int, width: int, config: RoadDetectionConfig) -> np.ndarray:
    center_x = width / 2.0
    top_y = int(round(height * config.seed_top_ratio))
    bottom_y = height - 1
    half_bottom = width * config.expansion_bottom_width_ratio / 2.0
    half_top = width * config.expansion_top_width_ratio / 2.0
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


def _position_prior(height: int, width: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    width_allowance = 0.62 + 1.3 * y
    center_penalty = np.clip(np.abs(x) / width_allowance, 0.0, 2.0)
    center_prior = np.exp(-0.45 * center_penalty * center_penalty)
    vertical_prior = np.power(y, 1.5)
    return (0.3 + 0.7 * vertical_prior) * center_prior


def _edge_suppression(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = magnitude / (float(magnitude.max()) + 1e-6)
    return np.clip(1.0 - 0.85 * normalized, 0.0, 1.0)


def _largest_bottom_component(mask: np.ndarray, config: RoadDetectionConfig) -> np.ndarray:
    height, width = mask.shape
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask

    center_band = np.zeros_like(mask, dtype=np.uint8)
    band_width = max(8, width // 8)
    center_x = width // 2
    center_band[int(height * 0.82) :, max(0, center_x - band_width) : min(width, center_x + band_width)] = 255
    min_area = int(height * width * config.min_component_ratio)

    best_label = 0
    best_area = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        touches_band = np.any((labels == label) & (center_band > 0))
        if touches_band and area > best_area:
            best_area = area
            best_label = label

    if best_label == 0:
        return mask
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _segment_bounds(row_mask: np.ndarray) -> list[tuple[int, int]]:
    active = np.flatnonzero(row_mask > 0)
    if active.size == 0:
        return []

    bounds: list[tuple[int, int]] = []
    start = int(active[0])
    end = start
    for value in active[1:]:
        index = int(value)
        if index == end + 1:
            end = index
            continue
        bounds.append((start, end))
        start = index
        end = index
    bounds.append((start, end))
    return bounds


def _rowwise_road_envelope(candidate_mask: np.ndarray, guide_score: np.ndarray, config: RoadDetectionConfig) -> np.ndarray:
    height, width = candidate_mask.shape
    result = np.zeros_like(candidate_mask, dtype=np.uint8)
    max_step = max(6, int(round(width * config.row_max_step_ratio)))
    center_x = width // 2
    previous_bounds: tuple[int, int] | None = None

    for y in range(height - 1, -1, -1):
        segments = _segment_bounds(candidate_mask[y])
        if not segments:
            continue

        if previous_bounds is None:
            chosen = None
            for left, right in segments:
                if left <= center_x <= right:
                    chosen = (left, right)
                    break
            if chosen is None:
                chosen = max(segments, key=lambda bounds: bounds[1] - bounds[0])
        else:
            previous_left, previous_right = previous_bounds
            best_score = None
            chosen = None
            for left, right in segments:
                overlap_left = max(left, previous_left - max_step)
                overlap_right = min(right, previous_right + max_step)
                overlap = max(0, overlap_right - overlap_left + 1)
                width_score = right - left + 1
                score = overlap * 3 + width_score - abs(((left + right) // 2) - ((previous_left + previous_right) // 2))
                if best_score is None or score > best_score:
                    best_score = score
                    chosen = (left, right)

            if chosen is not None:
                left, right = chosen
                if right < previous_left - max_step or left > previous_right + max_step:
                    chosen = previous_bounds

        if chosen is None:
            continue

        left, right = chosen
        if previous_bounds is not None:
            previous_left, previous_right = previous_bounds
            left = max(0, max(left, previous_left - max_step))
            right = min(width - 1, min(right, previous_right + max_step))
            if right <= left:
                continue

        support = guide_score[y, left : right + 1]
        if support.size > 0 and float(support.mean()) >= config.expansion_threshold * 0.75:
            result[y, left : right + 1] = 255
            previous_bounds = (left, right)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)


def _enforce_perspective_corridor(mask: np.ndarray, config: RoadDetectionConfig) -> np.ndarray:
    height, width = mask.shape
    corridor = _expansion_corridor_mask(height, width, config)
    constrained = cv2.bitwise_and(mask, corridor)
    top_limit = max(0, int(round(height * config.seed_top_ratio)) - int(round(height * 0.03)))
    constrained[:top_limit, :] = 0
    return constrained


def _grabcut_refine(
    image: np.ndarray,
    core_mask: np.ndarray,
    candidate_mask: np.ndarray,
    guide_score: np.ndarray,
    config: RoadDetectionConfig,
) -> np.ndarray:
    height, width = core_mask.shape
    corridor = _expansion_corridor_mask(height, width, config)
    gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[corridor > 0] = cv2.GC_PR_BGD
    gc_mask[candidate_mask > 0] = cv2.GC_PR_FGD
    gc_mask[core_mask > 0] = cv2.GC_FGD

    top_limit = max(0, int(round(height * config.seed_top_ratio)) - int(round(height * 0.03)))
    gc_mask[:top_limit, :] = cv2.GC_BGD

    low_score_bg = (guide_score < config.expansion_threshold * 0.8) & (corridor > 0)
    gc_mask[low_score_bg] = cv2.GC_PR_BGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    refined = _enforce_perspective_corridor(refined, config)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)
    return refined


def _recover_lateral_road_regions(
    mask: np.ndarray,
    candidate_mask: np.ndarray,
    guide_score: np.ndarray,
    config: RoadDetectionConfig,
) -> np.ndarray:
    height, width = mask.shape
    recovered = mask.copy()
    corridor = _expansion_corridor_mask(height, width, config)
    min_score = config.expansion_threshold * 0.85
    max_gap = max(12, int(round(width * 0.06)))

    for y in range(int(height * config.seed_top_ratio), height):
        active = np.flatnonzero(recovered[y] > 0)
        candidate = np.flatnonzero((candidate_mask[y] > 0) & (corridor[y] > 0) & (guide_score[y] >= min_score))
        if active.size == 0 or candidate.size == 0:
            continue

        left = int(active[0])
        right = int(active[-1])
        candidate_left = int(candidate[0])
        candidate_right = int(candidate[-1])

        if left - candidate_left <= max_gap or y > int(height * 0.58):
            recovered[y, candidate_left:left] = 255
        if candidate_right - right <= max_gap or y > int(height * 0.58):
            recovered[y, right : candidate_right + 1] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    recovered = cv2.morphologyEx(recovered, cv2.MORPH_CLOSE, kernel)
    return _enforce_perspective_corridor(recovered, config)


def detect_road(
    image: np.ndarray,
    config: RoadDetectionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or RoadDetectionConfig()
    cfg.validate()

    bgr = _ensure_color(image)
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    seed = _seed_mask(height, width, cfg)
    seed_pixels = lab[seed > 0]
    mean = seed_pixels.mean(axis=0)
    sigma = np.maximum(seed_pixels.std(axis=0), cfg.color_sigma_floor)

    color_distance = ((lab - mean) / sigma) ** 2
    color_score = np.exp(-0.5 * color_distance.sum(axis=2))

    position_score = _position_prior(height, width)
    edge_score = _edge_suppression(gray)
    vertical_prior = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    score = color_score * position_score * edge_score
    score = cv2.GaussianBlur(score.astype(np.float32), (9, 9), 0)
    expansion_score = color_score * np.sqrt(edge_score) * np.sqrt(np.clip(position_score, 0.0, 1.0)) * (0.42 + 0.58 * vertical_prior)
    expansion_score = cv2.GaussianBlur(expansion_score.astype(np.float32), (11, 11), 0)
    raw_mask = np.where(score >= cfg.score_threshold, 255, 0).astype(np.uint8)
    candidate_mask = np.where(expansion_score >= cfg.expansion_threshold, 255, 0).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_close)

    seed_boost = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(raw_mask, seed_boost))
    candidate_mask = cv2.bitwise_and(candidate_mask, _expansion_corridor_mask(height, width, cfg))
    candidate_mask = cv2.bitwise_or(candidate_mask, mask)
    mask = _rowwise_road_envelope(candidate_mask, expansion_score, cfg)
    mask = _enforce_perspective_corridor(mask, cfg)
    mask = _grabcut_refine(bgr, mask, candidate_mask, expansion_score, cfg)
    mask = _recover_lateral_road_regions(mask, candidate_mask, expansion_score, cfg)
    mask = _largest_bottom_component(mask, cfg)
    return mask, score


def make_road_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = _ensure_color(image).copy()
    overlay = bgr.copy()
    overlay[mask > 0] = (40, 190, 40)
    return cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0.0)


def detect_road_from_path(
    image_path: str | Path,
    config: RoadDetectionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    mask, score = detect_road(image, config)
    overlay = make_road_overlay(image, mask)
    return image, mask, overlay
