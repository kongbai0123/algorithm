from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--workers", type=int, default=4, help="dataloader worker count")
    parser.add_argument("--seed", type=int, default=42, help="training random seed")
    parser.add_argument("--optimizer", default="auto", help="optimizer, e.g. auto, SGD, AdamW")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted training run")
    parser.add_argument("--exist-ok", action="store_true", help="allow overwriting an existing run folder")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (base_dir / data_path).resolve()
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (base_dir / project_path).resolve()
    run_dir = project_path / args.name

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": str(project_path),
        "name": args.name,
        "task": "segment",
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
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

    _write_json(
        run_dir / "run_metadata.json",
        {
            "script": "train_road_segmentation.py",
            "model": args.model,
            "data": str(data_path),
            "project": str(project_path),
            "name": args.name,
            "train_kwargs": train_kwargs,
        },
    )
    result = model.train(**train_kwargs)
    _write_json(
        run_dir / "run_summary.json",
        {
            "script": "train_road_segmentation.py",
            "model": args.model,
            "data": str(data_path),
            "project": str(project_path),
            "name": args.name,
            "result_type": type(result).__name__,
        },
    )


if __name__ == "__main__":
    main()
