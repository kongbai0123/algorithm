# Optical Flow

This workspace contains a compact Python implementation of dense optical flow
based on the multiresolution Horn-Schunck method with bilinear interpolation.
It also includes a classical single-image road detection pipeline for images
placed under `input/`.

## Scope

- Single-scale Horn-Schunck refinement.
- Gaussian pyramid coarse-to-fine estimation.
- Bilinear image warping and flow prolongation.
- EPE and AAE metrics for evaluation.
- Synthetic translation demo.
- Road detection from input images.
- Static road score + optical-flow fusion for consecutive frames.
- Relaxed Horn-Schunck updates through configurable SOR-style optimization.
- PC-side road-surface segmentation project skeleton with YOLOv8-seg.
- Road mask engineering metrics: area, center offset, smoothness, stability.

## Run

```powershell
python -m pytest
python demo_synthetic.py
python detect_road_from_input.py
python train_road_segmentation.py --model yolov8n-seg.pt --data road_dataset/road.yaml
python infer_road_segmentation.py --weights runs/road_segmentation/yolov8n_seg_mvp/weights/best.pt
python detect_fused_road_pair.py --prev input/frame_0001.jpg --curr input/frame_0002.jpg
```

The demo writes `outputs/synthetic_flow.png`.
Road detection writes `outputs/*_road_mask.png` and `outputs/*_road_overlay.png`.
Road detection also writes `outputs/*_road_metrics_overlay.png`.
Flow-road fusion writes inspectable maps under `outputs/fusion_debug` by default.

## Road Surface Segmentation MVP

Dataset layout:

```text
road_dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  road.yaml
```

Target class:

```text
road_surface
```
