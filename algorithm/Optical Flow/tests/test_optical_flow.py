from pathlib import Path

import cv2
import numpy as np

from optical_flow import (
    FlowRoadFusionConfig,
    HornSchunckConfig,
    ExponentialSmoother,
    MajorityVoteSmoother,
    VideoPipelineConfig,
    analyze_road_mask,
    compare_optical_flow_methods,
    create_flow_estimator,
    detect_road,
    detect_road_with_optical_flow,
    endpoint_error,
    farneback_flow,
    lucas_kanade_sparse_flow,
    multiresolution_horn_schunck,
    postprocess_road_mask,
    process_video,
    fuse_masks,
    mask_iou,
    road_masked_flow_stats,
    sparse_points_to_flow,
    warp_mask_with_flow,
)
from optical_flow.image_ops import resize_flow, warp_bilinear


def _sample_input_image() -> np.ndarray:
    input_dir = Path(__file__).resolve().parents[1] / "input"
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if image_paths:
        image = cv2.imread(str(image_paths[0]))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_paths[0]}")
        return image

    video_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
    if not video_paths:
        raise FileNotFoundError(f"No test input images or videos found under: {input_dir}")

    capture = cv2.VideoCapture(str(video_paths[0]))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise FileNotFoundError(f"Unable to read first frame from video: {video_paths[0]}")
    return frame


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
    image = _sample_input_image()
    mask, _ = detect_road(image)
    from optical_flow import make_road_overlay
    overlay = make_road_overlay(image, mask)
    metrics = analyze_road_mask(mask, image.shape)

    assert mask.shape == image.shape[:2]
    assert overlay.shape == image.shape
    road_ratio = float((mask > 0).mean())
    assert 0.20 < road_ratio < 0.65
    assert mask[int(mask.shape[0] * 0.92), mask.shape[1] // 2] > 0
    assert mask[int(mask.shape[0] * 0.92), int(mask.shape[1] * 0.18)] > 0
    bottom_band = mask[int(mask.shape[0] * 0.88) :, :]
    assert float((bottom_band > 0).mean()) > 0.25
    assert metrics.valid_road is True
    assert metrics.stability_label in {"stable", "low_confidence"}


def test_analyze_road_mask_reports_center_and_area() -> None:
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[20:, 10:60] = 255

    metrics = analyze_road_mask(mask, mask.shape)

    assert 0.30 < metrics.road_area_ratio < 0.33
    assert metrics.road_center_x is not None
    assert metrics.road_center_offset_px is not None
    assert metrics.valid_road is True


def test_postprocess_road_mask_keeps_bottom_component() -> None:
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[60:78, 10:90] = 255
    mask[5:15, 5:25] = 255

    cleaned = postprocess_road_mask(mask, mask.shape, min_area_ratio=0.005, bottom_roi_ratio=0.35)

    assert cleaned[70, 50] > 0
    assert cleaned[10, 10] == 0


def test_farneback_and_flow_stats_report_translation() -> None:
    frame1 = np.zeros((64, 64), dtype=np.float32)
    cv2.rectangle(frame1, (18, 18), (42, 42), 1.0, thickness=-1)
    matrix = np.float32([[1, 0, 2], [0, 1, 0]])
    frame2 = cv2.warpAffine(frame1, matrix, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = np.zeros_like(frame1, dtype=np.uint8)
    mask[14:46, 14:46] = 255

    u, v = farneback_flow(frame1, frame2)
    stats = road_masked_flow_stats(u, v, mask, min_magnitude=0.01)

    assert stats.valid_pixel_count > 0
    assert stats.mean_magnitude > 0.2
    assert stats.consistency > 0.5


def test_lucas_kanade_sparse_flow_returns_points() -> None:
    frame1 = np.zeros((64, 64), dtype=np.float32)
    cv2.rectangle(frame1, (16, 16), (44, 44), 1.0, thickness=2)
    matrix = np.float32([[1, 0, 2], [0, 1, 1]])
    frame2 = cv2.warpAffine(frame1, matrix, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = np.ones_like(frame1, dtype=np.uint8) * 255

    start, end, _ = lucas_kanade_sparse_flow(frame1, frame2, mask)
    u, v, valid = sparse_points_to_flow(start, end, frame1.shape)
    stats = road_masked_flow_stats(u, v, mask, valid)

    assert start.shape[0] > 0
    assert stats.valid_pixel_count > 0
    assert stats.mean_u > 0.5


def test_temporal_smoothers() -> None:
    ema = ExponentialSmoother(alpha=0.5)
    assert ema.update(1.0) == 1.0
    assert ema.update(3.0) == 2.0

    vote = MajorityVoteSmoother(window_size=3, required_true=2)
    assert vote.update(True) is False
    assert vote.update(False) is False
    assert vote.update(True) is True


def test_native_flow_road_fusion_returns_debug_maps() -> None:
    curr = _sample_input_image()
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


def test_video_pipeline_respects_max_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic_road.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    assert writer.isOpened()
    try:
        for index in range(4):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :] = (40, 40, 40)
            cv2.rectangle(frame, (8 + index, 36), (88, 63), (120, 120, 120), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()

    result = process_video(
        video_path,
        tmp_path / "out",
        VideoPipelineConfig(max_frames=2, progress_every=1, save_sample_every=1),
    )

    assert result.frame_count == 2
    assert result.overlay_video_path.exists()
    assert result.metrics_csv_path.exists()
    assert result.summary_json_path.exists()


def test_video_pipeline_accepts_yolo_segmenter(tmp_path: Path) -> None:
    class FakeSegmenter:
        def segment(self, image: np.ndarray) -> np.ndarray:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[int(image.shape[0] * 0.55) :, :] = 255
            return mask

    video_path = tmp_path / "synthetic_yolo.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    assert writer.isOpened()
    try:
        for _ in range(3):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :] = (80, 80, 80)
            writer.write(frame)
    finally:
        writer.release()

    result = process_video(
        video_path,
        tmp_path / "out_yolo",
        VideoPipelineConfig(method="yolo-seg", max_frames=2, progress_every=1),
        yolo_segmenter=FakeSegmenter(),
    )

    assert result.frame_count == 2
    assert result.metrics_csv_path.exists()


def test_flow_comparison_writes_metrics(tmp_path: Path) -> None:
    frame1 = np.zeros((80, 120, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (20, 42), (105, 79), (120, 120, 120), thickness=-1)
    matrix = np.float32([[1, 0, 2], [0, 1, 0]])
    frame2 = cv2.warpAffine(frame1, matrix, (120, 80), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    prev_path = tmp_path / "prev.png"
    curr_path = tmp_path / "curr.png"
    out_path = tmp_path / "flow" / "flow_metrics.csv"
    cv2.imwrite(str(prev_path), frame1)
    cv2.imwrite(str(curr_path), frame2)

    rows = compare_optical_flow_methods(prev_path, curr_path, out_path, hs_iterations=5)

    assert out_path.exists()
    assert [row["method"] for row in rows] == ["horn_schunck", "farneback", "lucas_kanade"]
    assert all("flow_backend" in row for row in rows)


def test_flow_backend_factory_and_pwc_placeholder() -> None:
    assert create_flow_estimator("farneback").backend == "farneback"
    estimator = create_flow_estimator("pwcnet")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        estimator.estimate(frame, frame)
    except NotImplementedError as exc:
        assert "PWC-Net backend" in str(exc)
    else:
        raise AssertionError("PWC-Net placeholder must require a concrete adapter")


def test_temporal_mask_fusion_helpers() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:14, 6:12] = 255
    u = np.ones((20, 20), dtype=np.float32) * 2.0
    v = np.zeros((20, 20), dtype=np.float32)

    warped = warp_mask_with_flow(mask, u, v)
    assert warped[10, 13] > 0
    assert mask_iou(mask, mask) == 1.0
    fused = fuse_masks(np.zeros_like(mask), mask, current_weight=0.4)
    assert int((fused > 0).sum()) == int((mask > 0).sum())
