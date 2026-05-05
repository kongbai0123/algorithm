from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RoadMetrics:
    road_area_ratio: float
    road_center_x: float | None
    road_center_offset_px: float | None
    road_center_offset_ratio: float | None
    boundary_smoothness: float
    valid_road: bool
    stability_label: str


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return (binary * 255).astype(np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def _boundary_smoothness(mask: np.ndarray) -> float:
    largest = _largest_component(mask)
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return 0.0

    circularity = min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter + 1e-6))
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    simplification = len(approx) / max(len(contour), 1)
    return float(np.clip(0.75 * circularity + 0.25 * (1.0 - simplification), 0.0, 1.0))


def analyze_road_mask(mask: np.ndarray, image_shape: tuple[int, int] | tuple[int, int, int]) -> RoadMetrics:
    height, width = int(image_shape[0]), int(image_shape[1])
    binary = (np.asarray(mask) > 0).astype(np.uint8)

    road_area_ratio = float(binary.mean())
    xs = np.flatnonzero(binary.sum(axis=0) > 0)
    if xs.size == 0:
        return RoadMetrics(
            road_area_ratio=road_area_ratio,
            road_center_x=None,
            road_center_offset_px=None,
            road_center_offset_ratio=None,
            boundary_smoothness=0.0,
            valid_road=False,
            stability_label="unstable",
        )

    moments = cv2.moments(binary)
    road_center_x = float(moments["m10"] / moments["m00"]) if moments["m00"] > 0 else float(xs.mean())
    image_center_x = (width - 1) / 2.0
    offset_px = road_center_x - image_center_x
    offset_ratio = offset_px / max(image_center_x, 1.0)
    smoothness = _boundary_smoothness(binary)

    valid_road = road_area_ratio >= 0.05
    if road_area_ratio < 0.05 or smoothness < 0.18:
        stability_label = "unstable"
    elif road_area_ratio < 0.20 or smoothness < 0.32:
        stability_label = "low_confidence"
    else:
        stability_label = "stable"

    return RoadMetrics(
        road_area_ratio=road_area_ratio,
        road_center_x=road_center_x,
        road_center_offset_px=offset_px,
        road_center_offset_ratio=offset_ratio,
        boundary_smoothness=smoothness,
        valid_road=valid_road,
        stability_label=stability_label,
    )


def overlay_road_metrics(image: np.ndarray, metrics: RoadMetrics) -> np.ndarray:
    canvas = np.asarray(image).copy()
    lines = [
        f"Road: {'YES' if metrics.valid_road else 'NO'}",
        f"Area: {metrics.road_area_ratio * 100.0:.1f}%",
        f"Offset: {0.0 if metrics.road_center_offset_px is None else metrics.road_center_offset_px:.1f} px",
        f"State: {metrics.stability_label}",
        f"Smooth: {metrics.boundary_smoothness:.3f}",
    ]

    y = 30
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30

    if metrics.road_center_x is not None:
        center_x = int(round(metrics.road_center_x))
        image_center_x = int(round((canvas.shape[1] - 1) / 2.0))
        cv2.line(canvas, (center_x, 0), (center_x, canvas.shape[0] - 1), (50, 230, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (image_center_x, 0), (image_center_x, canvas.shape[0] - 1), (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

