from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from optical_flow import (
    analyze_road_mask,
    make_road_overlay,
    metrics_row,
    overlay_road_metrics,
    select_road_mask_from_yolo_result,
    write_metrics_csv,
    write_summary_json,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run YOLO road_surface segmentation on local input images.")
    parser.add_argument("--weights", required=True, help="trained segmentation weights path")
    parser.add_argument("--source", default="input", help="image directory or image file")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--out", default="outputs/segmentation_infer", help="output directory")
    parser.add_argument("--min-area-ratio", type=float, default=0.01, help="minimum connected road component area ratio")
    parser.add_argument("--bottom-roi-ratio", type=float, default=0.35, help="bottom ROI ratio used to keep road components")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = (base_dir / weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Segmentation weights not found: {weights_path}")

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = sorted(path for path in source_path.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise FileNotFoundError(f"No input images found under: {source_path}")

    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        mask = select_road_mask_from_yolo_result(
            result,
            image.shape,
            min_area_ratio=args.min_area_ratio,
            bottom_roi_ratio=args.bottom_roi_ratio,
        )

        overlay = make_road_overlay(image, mask)
        metrics = analyze_road_mask(mask, image.shape)
        metrics_overlay = overlay_road_metrics(overlay, metrics)

        stem = image_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_seg_mask.png"), mask)
        overlay_path = out_dir / f"{stem}_seg_overlay.png"
        metrics_path = out_dir / f"{stem}_seg_metrics_overlay.png"
        mask_path = out_dir / f"{stem}_seg_mask.png"
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(metrics_path), metrics_overlay)
        rows.append(metrics_row(image_path.name, metrics, mask_path, overlay_path, metrics_path))
        print(f"{image_path.name}: area={metrics.road_area_ratio:.3f} offset={0.0 if metrics.road_center_offset_px is None else metrics.road_center_offset_px:.2f} state={metrics.stability_label}")

    csv_path = out_dir / "segmentation_metrics.csv"
    summary_path = out_dir / "segmentation_summary.json"
    write_metrics_csv(csv_path, rows)
    write_summary_json(
        summary_path,
        rows,
        {
            "method": "yolo_segmentation",
            "weights": str(weights_path),
            "source": str(source_path),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "min_area_ratio": args.min_area_ratio,
            "bottom_roi_ratio": args.bottom_roi_ratio,
        },
    )
    print(f"metrics_csv={csv_path}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
