# Road Surface Segmentation and Optical Flow Fusion

This repository provides a PC-side validation pipeline for road surface
perception. It combines YOLOv8 segmentation, classical road scoring, and
optical-flow-based temporal fusion to evaluate road surface masks and engineering
metrics such as road area ratio, center offset, boundary smoothness, and
stability.

## Scope

- Single-scale Horn-Schunck refinement.
- Gaussian pyramid coarse-to-fine estimation.
- Bilinear image warping and flow prolongation.
- EPE and AAE metrics for evaluation.
- Synthetic translation demo.
- Road detection from input images.
- Static road score + optical-flow fusion for consecutive frames.
- Farneback and Lucas-Kanade optical-flow comparison inside road mask ROI.
- Relaxed Horn-Schunck updates through configurable SOR-style optimization.
- PC-side road-surface segmentation project skeleton with YOLOv8-seg.
- Road mask engineering metrics: area, center offset, smoothness, stability.

## Run

```powershell
python -m pytest
python demo_synthetic.py
python run_road_pipeline.py --source input
python run_road_pipeline.py --source "input/3_Video Project.mp4" --mode video --method classical --progress-every 10
python run_road_pipeline.py --source "input/3_Video Project.mp4" --mode video --start-frame 841 --progress-every 5
python detect_road_from_input.py
python train_road_segmentation.py --model yolov8n-seg.pt --data road_dataset/road.yaml
python infer_road_segmentation.py --weights runs/road_segmentation/yolov8n_seg_mvp/weights/best.pt
python detect_fused_road_pair.py --prev input/frame_0001.jpg --curr input/frame_0002.jpg
python compare_optical_flow.py --prev input/frame_0001.jpg --curr input/frame_0002.jpg
```

The demo writes `outputs/synthetic_flow.png`.
Road detection writes `outputs/*_road_mask.png` and `outputs/*_road_overlay.png`.
Road detection also writes `outputs/*_road_metrics_overlay.png`.
Road detection writes `outputs/road_metrics.csv` and `outputs/road_summary.json`.
The main pipeline writes `outputs/main_pipeline/*_road_overlay.mp4`, `video_metrics.csv`, and sampled overlays.
Video progress is printed as `progress=current/total`, elapsed time, and ETA.
Flow-road fusion writes inspectable maps under `outputs/fusion_debug` by default.
YOLO inference writes `outputs/segmentation_infer/segmentation_metrics.csv` and
`outputs/segmentation_infer/segmentation_summary.json`.
Optical-flow comparison writes `outputs/flow_compare/flow_metrics.csv`.

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

Labeling rules and split policy are documented in `road_dataset/README.md`.
