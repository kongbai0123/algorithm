from __future__ import annotations

from pathlib import Path

import cv2

from optical_flow import RoadDetectionConfig, analyze_road_mask, detect_road_from_path, overlay_road_metrics


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    config = RoadDetectionConfig()
    image_paths = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError("No input images found under ./input")

    for path in image_paths:
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

        print(f"{path.name}: road_ratio={metrics.road_area_ratio:.3f}")
        print(f"  state={metrics.stability_label}")
        print(f"  offset_px={0.0 if metrics.road_center_offset_px is None else metrics.road_center_offset_px:.2f}")
        print(f"  smoothness={metrics.boundary_smoothness:.3f}")
        print(f"  mask={mask_path}")
        print(f"  overlay={overlay_path}")
        print(f"  metrics_overlay={metrics_path}")


if __name__ == "__main__":
    main()
