from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO

def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8-seg model to ONNX.")
    parser.add_argument("--weights", default="yolov8n-seg.pt", help="path to PyTorch weights")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--dynamic", action="store_true", help="enable dynamic dimensions")
    parser.add_argument("--simplify", action="store_true", help="simplify ONNX graph")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Exporting to ONNX with imgsz={args.imgsz}, dynamic={args.dynamic}, simplify={args.simplify}, opset={args.opset}...")
    
    export_args = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "opset": args.opset,
    }
    
    path = model.export(**export_args)
    print(f"Export successful! File saved at: {path}")

if __name__ == "__main__":
    main()
