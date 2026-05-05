from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a road_surface segmentation model on PC.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLO segmentation checkpoint or model name")
    parser.add_argument("--data", default="road_dataset/road.yaml", help="dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch", type=int, default=8, help="training batch size")
    parser.add_argument("--project", default="runs/road_segmentation", help="training project output folder")
    parser.add_argument("--name", default="yolov8n_seg_mvp", help="run name")
    parser.add_argument("--device", default=None, help="training device, e.g. cpu, 0")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (base_dir / data_path).resolve()
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (base_dir / project_path).resolve()

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": str(project_path),
        "name": args.name,
        "task": "segment",
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.35,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.25,
        "shear": 0.0,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

