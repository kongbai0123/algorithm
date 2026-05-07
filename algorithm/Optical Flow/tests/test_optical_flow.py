from pathlib import Path

import cv2
import numpy as np

from optical_flow import (
    FlowRoadFusionConfig,
    HornSchunckConfig,
    detect_road_from_path,
    detect_road_with_optical_flow,
    endpoint_error,
    multiresolution_horn_schunck,
)
from optical_flow.image_ops import resize_flow, warp_bilinear


def test_resize_flow_scales_vectors() -> None:
    u = np.ones((8, 8), dtype=np.float32)
    v = np.ones((8, 8), dtype=np.float32) * 2

    resized_u, resized_v = resize_flow(u, v, (16, 24))

    assert resized_u.shape == (16, 24)
    assert resized_v.shape == (16, 24)
    np.testing.assert_allclose(resized_u.mean(), 3.0, atol=1e-6)
    np.testing.assert_allclose(resized_v.mean(), 4.0, atol=1e-6)


def test_warp_bilinear_moves_image_by_flow() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    image[:, 8:] = 1.0
    u = np.ones_like(image) * 2.0
    v = np.zeros_like(image)

    warped = warp_bilinear(image, u, v)

    assert warped[10, 5] < 0.1
    assert warped[10, 7] > 0.9


def test_multiresolution_horn_schunck_detects_synthetic_translation() -> None:
    size = 72
    shift_x, shift_y = 3, 2
    frame1 = np.zeros((size, size), dtype=np.float32)
    cv2.rectangle(frame1, (18, 20), (54, 52), 1.0, thickness=-1)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    frame2 = cv2.warpAffine(frame1, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    config = HornSchunckConfig(alpha=1.0, iterations=220, pyramid_levels=4, warps_per_level=4)
    u, v = multiresolution_horn_schunck(frame1, frame2, config)

    mask = frame1 > 0.5
    target_u = np.full_like(u, shift_x, dtype=np.float32)
    target_v = np.full_like(v, shift_y, dtype=np.float32)
    assert endpoint_error(u[mask], v[mask], target_u[mask], target_v[mask]) < 1.6


def test_horn_schunck_supports_relaxed_optimization() -> None:
    size = 56
    frame1 = np.zeros((size, size), dtype=np.float32)
    cv2.rectangle(frame1, (16, 16), (40, 40), 1.0, thickness=-1)
    matrix = np.float32([[1, 0, 2], [0, 1, 1]])
    frame2 = cv2.warpAffine(frame1, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    config = HornSchunckConfig(alpha=1.0, iterations=120, pyramid_levels=3, warps_per_level=3, relaxation=1.15)
    u, v = multiresolution_horn_schunck(frame1, frame2, config)

    mask = frame1 > 0.5
    assert float(u[mask].mean()) > 0.8
    assert float(v[mask].mean()) > 0.3


def test_detect_road_from_input_image_returns_plausible_mask() -> None:
    image_path = Path(__file__).resolve().parents[1] / "input" / "pexels_11794520.jpg"

    image, mask, overlay = detect_road_from_path(image_path)

    assert mask.shape == image.shape[:2]
    assert overlay.shape == image.shape
    road_ratio = float((mask > 0).mean())
    assert 0.16 < road_ratio < 0.75
    assert mask[int(mask.shape[0] * 0.92), mask.shape[1] // 2] > 0
    assert mask[int(mask.shape[0] * 0.92), int(mask.shape[1] * 0.28)] > 0
    assert mask[int(mask.shape[0] * 0.92), int(mask.shape[1] * 0.68)] > 0


def test_native_flow_road_fusion_returns_debug_maps() -> None:
    image_path = Path(__file__).resolve().parents[1] / "input" / "pexels_11794520.jpg"
    curr = cv2.imread(str(image_path))
    assert curr is not None
    matrix = np.float32([[1, 0, -2], [0, 1, 0]])
    prev = cv2.warpAffine(curr, matrix, (curr.shape[1], curr.shape[0]), flags=cv2.INTER_LINEAR)

    flow_config = HornSchunckConfig(alpha=8.0, iterations=30, pyramid_levels=3, warps_per_level=1, relaxation=1.05)
    fusion_config = FlowRoadFusionConfig(fused_threshold=0.35)
    mask, score, debug = detect_road_with_optical_flow(prev, curr, flow_config=flow_config, fusion_config=fusion_config)

    assert mask.shape == curr.shape[:2]
    assert score.shape == curr.shape[:2]
    assert {"static_score", "motion_score", "temporal_score", "flow_magnitude"}.issubset(debug)
    road_ratio = float((mask > 0).mean())
    assert 0.05 < road_ratio < 0.75
