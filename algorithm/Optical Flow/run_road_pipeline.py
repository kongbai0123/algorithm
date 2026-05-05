from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from optical_flow import (
    RoadDetectionConfig,
    YoloRoadSegmenter,
    YoloSegmentationConfig,
    analyze_road_mask,
    compare_optical_flow_methods,
    detect_fused_road_from_paths,
    detect_road_from_path,
    metrics_row,
    overlay_road_metrics,
    write_metrics_csv,
    write_summary_json,
)
from optical_flow.video_pipeline import VideoPipelineConfig, process_video


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base_dir / path).resolve()


def _write_image_outputs(
    source_path: Path,
    output_dir: Path,
    config: RoadDetectionConfig,
    method: str,
    yolo_segmenter: object | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = sorted(path for path in source_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No input images found under: {source_path}")

    rows = []
    for path in image_paths:
        if method == "yolo-seg":
            if yolo_segmenter is None or not hasattr(yolo_segmenter, "segment"):
                raise ValueError("yolo-seg image mode requires --weights")
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {path}")
            mask = yolo_segmenter.segment(image)
            from optical_flow import make_road_overlay
            overlay = make_road_overlay(image, mask)
        else:
            image, mask, overlay = detect_road_from_path(path, config)
        metrics = analyze_road_mask(mask, image.shape)
        metrics_overlay = overlay_road_metrics(overlay, metrics)
        stem = path.stem
        mask_path = output_dir / f"{stem}_road_mask.png"
        overlay_path = output_dir / f"{stem}_road_overlay.png"
        metrics_path = output_dir / f"{stem}_road_metrics_overlay.png"
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(metrics_path), metrics_overlay)
        rows.append(metrics_row(path.name, metrics, mask_path, overlay_path, metrics_path))
        print(
            f"{path.name}: area={metrics.road_area_ratio:.3f} "
            f"offset={0.0 if metrics.road_center_offset_px is None else metrics.road_center_offset_px:.2f} "
            f"state={metrics.stability_label}"
        )

    csv_path = output_dir / "road_metrics.csv"
    summary_path = output_dir / "road_summary.json"
    write_metrics_csv(csv_path, rows)
    write_summary_json(
        summary_path,
        rows,
        {
            "mode": "images",
            "method": method,
            "source": str(source_path),
            "output_dir": str(output_dir),
        },
    )
    print(f"metrics_csv={csv_path}")
    print(f"summary_json={summary_path}")


def _run_pair(prev_path: Path, curr_path: Path, output_dir: Path) -> None:
    mask, score, _ = detect_fused_road_from_paths(prev_path, curr_path, output_dir)
    road_ratio = float((mask > 0).mean())
    print(f"road_ratio={road_ratio:.3f}")
    print(f"score_mean={float(score.mean()):.3f}")
    print(f"output={output_dir}")


def _run_flow_compare(prev_path: Path, curr_path: Path, output_dir: Path, hs_iterations: int) -> None:
    output_path = output_dir / "flow_metrics.csv"
    rows = compare_optical_flow_methods(prev_path, curr_path, output_path, hs_iterations)
    for row in rows:
        print(
            f"{row['method']}: count={row['valid_pixel_count']} "
            f"mean_mag={float(row['mean_magnitude']):.4f} "
            f"angle={row['direction_angle_deg']} "
            f"consistency={float(row['consistency']):.3f}"
        )
    print(f"metrics_csv={output_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Unified road pipeline entrypoint for images, video, pair fusion, and flow comparison.")
    parser.add_argument("--source", default=None, help="image path, image directory, or video path")
    parser.add_argument("--mode", choices=["auto", "images", "video", "pair", "flow-compare"], default="auto", help="input mode")
    parser.add_argument("--method", choices=["classical", "fused", "yolo-seg", "yolo-seg-fused"], default="classical", help="road processing method")
    parser.add_argument("--weights", default=None, help="YOLO segmentation weights for yolo-seg methods")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--min-area-ratio", type=float, default=0.01, help="minimum YOLO road component area ratio")
    parser.add_argument("--bottom-roi-ratio", type=float, default=0.35, help="bottom ROI ratio for YOLO road components")
    parser.add_argument("--prev", default=None, help="previous image path for pair mode")
    parser.add_argument("--curr", default=None, help="current image path for pair mode")
    parser.add_argument("--out", default="outputs/main_pipeline", help="output directory")
    parser.add_argument("--start-frame", type=int, default=1, help="1-based frame index for video resume")
    parser.add_argument("--save-sample-every", type=int, default=30, help="save metrics-overlay samples every N frames")
    parser.add_argument("--max-frames", type=int, default=None, help="optional frame cap for video processing")
    parser.add_argument("--progress-every", type=int, default=10, help="print video progress every N frames")
    parser.add_argument("--hs-iterations", type=int, default=120, help="Horn-Schunck iterations for flow-compare mode")
    parser.add_argument("--write-mask-video", action="store_true", help="also write a grayscale road mask video")
    args = parser.parse_args()

    output_dir = _resolve_path(args.out, base_dir)
    road_config = RoadDetectionConfig()
    yolo_segmenter = None
    if args.method in {"yolo-seg", "yolo-seg-fused"}:
        if args.weights is None:
            raise ValueError(f"{args.method} requires --weights")
        weights_path = _resolve_path(args.weights, base_dir)
        yolo_segmenter = YoloRoadSegmenter(
            YoloSegmentationConfig(
                weights=str(weights_path),
                imgsz=args.imgsz,
                conf=args.conf,
                min_area_ratio=args.min_area_ratio,
                bottom_roi_ratio=args.bottom_roi_ratio,
            )
        )

    mode = args.mode
    if mode == "auto":
        if args.source is None:
            raise ValueError("auto mode requires --source")
        source_path = _resolve_path(args.source, base_dir)
        if source_path.suffix.lower() in VIDEO_EXTENSIONS:
            mode = "video"
        elif source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
            mode = "images"
        elif source_path.is_dir():
            mode = "images"
        else:
            raise ValueError(f"Unable to infer mode from source: {source_path}")
    else:
        source_path = _resolve_path(args.source, base_dir) if args.source is not None else None

    if mode == "images":
        if source_path is None:
            raise ValueError("images mode requires --source")
        if args.method not in {"classical", "yolo-seg"}:
            raise ValueError("images mode supports --method classical or yolo-seg")
        _write_image_outputs(source_path, output_dir, road_config, args.method, yolo_segmenter)
        return

    if mode == "pair":
        prev_path = _resolve_path(args.prev, base_dir) if args.prev is not None else None
        curr_path = _resolve_path(args.curr, base_dir) if args.curr is not None else None
        if prev_path is None or curr_path is None:
            raise ValueError("pair mode requires --prev and --curr")
        _run_pair(prev_path, curr_path, output_dir)
        return

    if mode == "flow-compare":
        prev_path = _resolve_path(args.prev, base_dir) if args.prev is not None else None
        curr_path = _resolve_path(args.curr, base_dir) if args.curr is not None else None
        if prev_path is None or curr_path is None:
            raise ValueError("flow-compare mode requires --prev and --curr")
        _run_flow_compare(prev_path, curr_path, output_dir, args.hs_iterations)
        return

    if source_path is None:
        raise ValueError("video mode requires --source")
    result = process_video(
        source_path,
        output_dir,
        VideoPipelineConfig(
            method=args.method,
            start_frame=args.start_frame,
            save_sample_every=args.save_sample_every,
            max_frames=args.max_frames,
            write_mask_video=args.write_mask_video,
            progress_every=args.progress_every,
        ),
        road_config,
        yolo_segmenter,
    )
    print(f"frames={result.frame_count}")
    print(f"overlay_video={result.overlay_video_path}")
    if result.mask_video_path is not None:
        print(f"mask_video={result.mask_video_path}")
    print(f"metrics_csv={result.metrics_csv_path}")
    print(f"summary_json={result.summary_json_path}")
    print(f"samples={result.sample_dir}")


if __name__ == "__main__":
    main()
