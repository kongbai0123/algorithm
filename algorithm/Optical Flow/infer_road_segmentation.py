from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from optical_flow import analyze_road_mask, make_road_overlay, overlay_road_metrics


def _largest_mask(result) -> np.ndarray | None:
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        return None

    data = masks.data.detach().cpu().numpy()
    if data.ndim != 3:
        return None
    areas = data.reshape(data.shape[0], -1).sum(axis=1)
    index = int(np.argmax(areas))
    return (data[index] > 0.5).astype(np.uint8) * 255


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run YOLO road_surface segmentation on local input images.")
    parser.add_argument("--weights", required=True, help="trained segmentation weights path")
    parser.add_argument("--source", default="input", help="image directory or image file")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--out", default="outputs/segmentation_infer", help="output directory")
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

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        mask = _largest_mask(result)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        overlay = make_road_overlay(image, mask)
        metrics = analyze_road_mask(mask, image.shape)
        metrics_overlay = overlay_road_metrics(overlay, metrics)

        stem = image_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_seg_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_seg_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{stem}_seg_metrics_overlay.png"), metrics_overlay)
        print(f"{image_path.name}: area={metrics.road_area_ratio:.3f} offset={0.0 if metrics.road_center_offset_px is None else metrics.road_center_offset_px:.2f} state={metrics.stability_label}")


if __name__ == "__main__":
    main()

