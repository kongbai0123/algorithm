from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import cv2
import numpy as np

from .flow_road_fusion import detect_road_with_optical_flow
from .reporting import metrics_row, write_metrics_csv, write_summary_json
from .road_analysis import RoadMetrics, analyze_road_mask, overlay_road_metrics
from .road_detection import RoadDetectionConfig, detect_road, make_road_overlay
from .temporal_mask_fusion import flicker_flag, mask_iou
from .temporal_smoothing import ExponentialSmoother, MajorityVoteSmoother
from .segmenters import ClassicalRoadSegmenter
from .temporal import EMATemporalFusion


@dataclass(frozen=True)
class VideoPipelineConfig:
    method: str = "classical"
    start_frame: int = 1
    save_sample_every: int = 30
    max_frames: int | None = None
    write_mask_video: bool = False
    progress_every: int = 10
    smooth_alpha: float = 0.7
    stable_window_size: int = 5
    stable_required_true: int = 4
    fusion_flow_method: str = "farneback"
    fusion_alpha: float = 0.7

    def validate(self) -> None:
        if self.method not in {"classical", "fused", "yolo-seg", "yolo-seg-fused"}:
            raise ValueError("method must be 'classical', 'fused', 'yolo-seg', or 'yolo-seg-fused'")
        if self.start_frame < 1:
            raise ValueError("start_frame must be >= 1")
        if self.save_sample_every < 1:
            raise ValueError("save_sample_every must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 when provided")
        if self.progress_every < 1:
            raise ValueError("progress_every must be >= 1")
        if not 0.0 <= self.smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be between 0 and 1")
        if not 0.0 <= self.fusion_alpha <= 1.0:
            raise ValueError("fusion_alpha must be between 0 and 1")
        if self.stable_window_size < 1:
            raise ValueError("stable_window_size must be >= 1")
        if not 1 <= self.stable_required_true <= self.stable_window_size:
            raise ValueError("stable_required_true must be between 1 and stable_window_size")


@dataclass(frozen=True)
class VideoRunResult:
    overlay_video_path: Path
    mask_video_path: Path | None
    metrics_csv_path: Path
    summary_json_path: Path
    sample_dir: Path
    frame_count: int


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _smoothed_metrics(
    metrics: RoadMetrics,
    area_smoother: ExponentialSmoother,
    offset_smoother: ExponentialSmoother,
    smoothness_smoother: ExponentialSmoother,
    valid_vote: MajorityVoteSmoother,
) -> RoadMetrics:
    smoothed_area = area_smoother.update(metrics.road_area_ratio)
    offset_value = 0.0 if metrics.road_center_offset_px is None else metrics.road_center_offset_px
    smoothed_offset = offset_smoother.update(offset_value)
    smoothed_boundary = smoothness_smoother.update(metrics.boundary_smoothness)
    smoothed_valid = valid_vote.update(metrics.valid_road)
    offset_ratio = None
    if metrics.road_center_offset_px is not None and metrics.road_center_offset_ratio is not None:
        if abs(metrics.road_center_offset_px) > 1e-6:
            offset_ratio = metrics.road_center_offset_ratio * (smoothed_offset / metrics.road_center_offset_px)
        else:
            offset_ratio = 0.0

    if smoothed_area < 0.05 or smoothed_boundary < 0.18:
        stability_label = "unstable"
    elif smoothed_area < 0.20 or smoothed_boundary < 0.32:
        stability_label = "low_confidence"
    else:
        stability_label = "stable" if smoothed_valid else "low_confidence"

    return replace(
        metrics,
        road_area_ratio=smoothed_area,
        road_center_offset_px=smoothed_offset if metrics.road_center_offset_px is not None else None,
        road_center_offset_ratio=offset_ratio,
        boundary_smoothness=smoothed_boundary,
        valid_road=smoothed_valid,
        stability_label=stability_label,
    )


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    pipeline_config: VideoPipelineConfig | None = None,
    road_config: RoadDetectionConfig | None = None,
    yolo_segmenter: object | None = None,
) -> VideoRunResult:
    cfg = pipeline_config or VideoPipelineConfig()
    cfg.validate()
    road_cfg = road_config or RoadDetectionConfig()
    road_cfg.validate()

    source_path = Path(video_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions for: {source_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_video_path = out_dir / f"{source_path.stem}_road_overlay.mp4"
    overlay_writer = cv2.VideoWriter(str(overlay_video_path), fourcc, fps, (width, height))
    if not overlay_writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create overlay video: {overlay_video_path}")

    mask_video_path = out_dir / f"{source_path.stem}_road_mask.mp4" if cfg.write_mask_video else None
    mask_writer = None
    if mask_video_path is not None:
        mask_writer = cv2.VideoWriter(str(mask_video_path), fourcc, fps, (width, height))
        if not mask_writer.isOpened():
            overlay_writer.release()
            capture.release()
            raise RuntimeError(f"Unable to create mask video: {mask_video_path}")

    rows: list[dict[str, object]] = []
    prev_frame: np.ndarray | None = None
    prev_score: np.ndarray | None = None
    frame_index = 0
    absolute_frame_index = cfg.start_frame - 1
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_source_frames > 0 and cfg.start_frame > total_source_frames:
        capture.release()
        overlay_writer.release()
        if mask_writer is not None:
            mask_writer.release()
        raise ValueError(f"start_frame {cfg.start_frame} exceeds total frame count {total_source_frames}")
    if cfg.start_frame > 1:
        capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame - 1)
    total_frames = total_source_frames - (cfg.start_frame - 1) if total_source_frames > 0 else 0
    if cfg.max_frames is not None and total_frames > 0:
        total_frames = min(total_frames, cfg.max_frames)
    area_smoother = ExponentialSmoother(alpha=cfg.smooth_alpha)
    offset_smoother = ExponentialSmoother(alpha=cfg.smooth_alpha)
    smoothness_smoother = ExponentialSmoother(alpha=cfg.smooth_alpha)
    valid_vote = MajorityVoteSmoother(cfg.stable_window_size, cfg.stable_required_true)
    start_time = time.perf_counter()
    previous_yolo_mask: np.ndarray | None = None
    previous_output_mask: np.ndarray | None = None

    # Initialize abstractions
    segmenter = None
    fusion = None
    fusion_flow_estimator = None
    if cfg.method in {"yolo-seg", "yolo-seg-fused"}:
        segmenter = yolo_segmenter
    else:
        segmenter = ClassicalRoadSegmenter(road_cfg)
        
    if cfg.method == "yolo-seg-fused":
        from .temporal import WarpTemporalFusion
        fusion = WarpTemporalFusion(alpha=cfg.fusion_alpha)
        from .flow import create_flow_estimator
        fusion_flow_estimator = create_flow_estimator(cfg.fusion_flow_method)

    try:
        while True:
            if cfg.max_frames is not None and frame_index >= cfg.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            absolute_frame_index += 1

            if cfg.method == "fused" and prev_frame is not None:
                mask, score, _ = detect_road_with_optical_flow(prev_frame, frame, road_cfg, prev_score=prev_score)
                prev_score = score
            else:
                if segmenter is None:
                    raise ValueError(f"Segmenter not initialized for method: {cfg.method}")
                mask = segmenter.segment(frame)
                score = (mask > 0).astype(np.float32)
                
                if cfg.method == "yolo-seg-fused" and previous_yolo_mask is not None:
                    if fusion is None:
                        raise ValueError("Fusion not initialized for yolo-seg-fused")
                    
                    flow_result = None
                    if prev_frame is not None and fusion_flow_estimator is not None:
                        flow_result = fusion_flow_estimator.estimate(prev_frame, frame)
                        
                    mask = fusion.fuse(mask, previous_yolo_mask, flow_result)
                    score = (mask > 0).astype(np.float32)
                
                previous_yolo_mask = mask.copy()
                prev_score = score

            overlay = make_road_overlay(frame, mask)
            metrics = analyze_road_mask(mask, frame.shape)
            metrics = _smoothed_metrics(metrics, area_smoother, offset_smoother, smoothness_smoother, valid_vote)
            metrics_overlay = overlay_road_metrics(overlay, metrics)
            mask_iou_prev = mask_iou(previous_output_mask, mask) if previous_output_mask is not None else None
            flicker = flicker_flag(previous_output_mask, mask)

            overlay_writer.write(metrics_overlay)
            if mask_writer is not None:
                mask_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            sample_written = sample_dir / f"frame_{absolute_frame_index:06d}_metrics_overlay.png"
            mask_written = sample_dir / f"frame_{absolute_frame_index:06d}_mask.png"
            if frame_index == 1 or absolute_frame_index % cfg.save_sample_every == 0:
                cv2.imwrite(str(sample_written), metrics_overlay)
                cv2.imwrite(str(mask_written), mask)
            else:
                sample_written = overlay_video_path
                mask_written = mask_video_path or overlay_video_path

            row = metrics_row(
                f"frame_{absolute_frame_index:06d}",
                metrics,
                mask_written,
                overlay_video_path,
                sample_written,
            )
            row.update(
                {
                    "mask_iou_prev": mask_iou_prev,
                    "flicker": flicker,
                    "flow_backend": "none" if cfg.method in {"classical", "yolo-seg", "yolo-seg-fused"} else "horn_schunck",
                }
            )
            rows.append(row)
            prev_frame = frame.copy()
            previous_output_mask = mask.copy()

            if frame_index == 1 or frame_index % cfg.progress_every == 0 or (total_frames > 0 and frame_index == total_frames):
                elapsed = time.perf_counter() - start_time
                avg_per_frame = elapsed / max(frame_index, 1)
                remaining_frames = max(total_frames - frame_index, 0) if total_frames > 0 else None
                eta_seconds = avg_per_frame * remaining_frames if remaining_frames is not None else None
                percent = (100.0 * frame_index / total_frames) if total_frames > 0 else 0.0
                print(
                    f"progress={frame_index}/{total_frames if total_frames > 0 else '?'} "
                    f"({percent:.1f}%) elapsed={_format_seconds(elapsed)} "
                    f"eta={_format_seconds(eta_seconds)}",
                    flush=True,
                )

    finally:
        capture.release()
        overlay_writer.release()
        if mask_writer is not None:
            mask_writer.release()

    metrics_csv_path = out_dir / "video_metrics.csv"
    summary_json_path = out_dir / "video_summary.json"
    write_metrics_csv(metrics_csv_path, rows)
    write_summary_json(
        summary_json_path,
        rows,
        {
            "mode": "video",
            "method": cfg.method,
            "video_path": str(source_path),
            "output_dir": str(out_dir),
            "start_frame": cfg.start_frame,
            "save_sample_every": cfg.save_sample_every,
            "max_frames": cfg.max_frames,
            "progress_every": cfg.progress_every,
            "smooth_alpha": cfg.smooth_alpha,
            "stable_window_size": cfg.stable_window_size,
            "stable_required_true": cfg.stable_required_true,
        },
    )
    return VideoRunResult(
        overlay_video_path=overlay_video_path,
        mask_video_path=mask_video_path,
        metrics_csv_path=metrics_csv_path,
        summary_json_path=summary_json_path,
        sample_dir=sample_dir,
        frame_count=frame_index,
    )
