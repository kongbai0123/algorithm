# Deployment Guide

This guide describes how to deploy the road perception pipeline from a PC to an edge device like Jetson Orin Nano.

## 1. PC: PyTorch to ONNX

To export the YOLOv8-seg model to ONNX format on your PC, use the provided script:

```bash
python tools/export_onnx.py --weights yolov8n-seg.pt --imgsz 640 --dynamic --simplify --opset 12
```

This will produce a file named `yolov8n-seg.onnx` in the same directory as the weights (or specified path).

## 2. Jetson: ONNX to TensorRT

> [!IMPORTANT]
> TensorRT engines (`.engine`) are hardware-specific. You **must** generate the engine on the target Jetson device. Do not copy a `.engine` file from your PC to the Jetson.

On the Jetson device, use `trtexec` to convert the ONNX model to a TensorRT engine:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n-seg.onnx \
  --saveEngine=yolov8n-seg_fp16.engine \
  --fp16 \
  --workspace=2048
```

If you need INT8 quantization (requires a calibration cache):

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n-seg.onnx \
  --saveEngine=yolov8n-seg_int8.engine \
  --int8 \
  --calib=calibration.cache
```

## 3. Runtime Profiling

To analyze the execution time of different components on the PC or Jetson, use the `--profile` flag:

```bash
python run_road_pipeline.py --source videos/3_Video Project.mp4 --method yolo-seg-fused --weights yolov8n-seg.pt --profile
```

This will output:
- `outputs/profile_summary.json` (Aggregated statistics)
- `outputs/profile_metrics.csv` (Frame-by-frame timings)
