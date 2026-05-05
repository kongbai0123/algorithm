# 道路面分割與光流融合

此專案提供 PC 端道路面感知驗證流程，結合 YOLOv8 分割、傳統道路評分，以及基於光流的時間融合，用來評估道路面遮罩與工程指標，例如道路面積比例、中心偏移、邊界平滑度與穩定性。

## 範圍

- 單尺度 Horn-Schunck 細化。
- 高斯金字塔 coarse-to-fine 估計。
- 雙線性影像扭曲與光流延拓。
- EPE 與 AAE 評估指標。
- 合成平移示範。
- 從輸入影像進行道路偵測。
- 連續影格的靜態道路分數與光流融合。
- 在道路遮罩 ROI 內比較 Farneback 與 Lucas-Kanade 光流。
- 以可設定的類 SOR 最佳化做放寬式 Horn-Schunck 更新。
- 以 YOLOv8-seg 為核心的 PC 端道路面分割專案骨架。
- 道路遮罩工程指標：面積、中心偏移、平滑度、穩定性。

## 流程

```text
影像 / 影片 / 影格對
  -> run_road_pipeline.py
  -> 產生道路遮罩
  -> 遮罩後處理
  -> 道路指標分析
  -> 選擇性進行光流比較或融合
  -> 輸出疊圖影片或影像
  -> 產生 CSV 與 JSON 報告
```

## 執行方式

```powershell
python -m pytest
python demo_synthetic.py
python run_road_pipeline.py --source input
python run_road_pipeline.py --source "input/3_Video Project.mp4" --mode video --method classical --progress-every 10
python run_road_pipeline.py --source "input/3_Video Project.mp4" --mode video --start-frame 841 --progress-every 5
python run_road_pipeline.py --source input --mode images --method yolo-seg --weights runs/road_segmentation/yolov8n_seg_mvp/weights/best.pt
python run_road_pipeline.py --source "input/3_Video Project.mp4" --mode video --method yolo-seg --weights runs/road_segmentation/yolov8n_seg_mvp/weights/best.pt
python run_road_pipeline.py --source "input/3_Video Project.mp4" --mode video --method yolo-seg-fused --weights runs/road_segmentation/yolov8n_seg_mvp/weights/best.pt
python run_road_pipeline.py --mode pair --prev input/frame_0001.jpg --curr input/frame_0002.jpg
python run_road_pipeline.py --mode flow-compare --prev input/frame_0001.jpg --curr input/frame_0002.jpg
python run_road_pipeline.py --mode flow-compare --prev input/frame_0001.jpg --curr input/frame_0002.jpg --flow-method farneback
python detect_road_from_input.py
python train_road_segmentation.py --model yolov8n-seg.pt --data road_dataset/road.yaml --seed 42 --patience 20
python infer_road_segmentation.py --weights runs/road_segmentation/yolov8n_seg_mvp/weights/best.pt
python detect_fused_road_pair.py --prev input/frame_0001.jpg --curr input/frame_0002.jpg
python compare_optical_flow.py --prev input/frame_0001.jpg --curr input/frame_0002.jpg
```

`demo_synthetic.py` 會輸出 `outputs/synthetic_flow.png`。  
道路偵測會輸出 `outputs/*_road_mask.png`、`outputs/*_road_overlay.png` 與 `outputs/*_road_metrics_overlay.png`。  
道路偵測也會輸出 `outputs/road_metrics.csv` 與 `outputs/road_summary.json`。  
主流程會輸出 `outputs/main_pipeline/*_road_overlay.mp4`、`video_metrics.csv` 與取樣疊圖。  
影片進度會顯示為 `progress=current/total`、elapsed 與 ETA。  
光流與道路融合預設會把可檢查的中間圖輸出到 `outputs/fusion_debug`。  
YOLO 推論會輸出 `outputs/segmentation_infer/segmentation_metrics.csv` 與 `outputs/segmentation_infer/segmentation_summary.json`。  
光流比較會輸出 `outputs/flow_compare/flow_metrics.csv`。  
光流比較支援以 `--flow-method` 指定後端；`pwcnet` 目前保留為占位模式，待實際的 PWC-Net 轉接器與權重提供後再啟用。  
`detect_road_from_input.py`、`detect_fused_road_pair.py` 與 `compare_optical_flow.py` 為相容性入口；主要執行入口是 `run_road_pipeline.py`。

## 輸出指標

```text
road_area_ratio         道路遮罩面積 / 影像面積
road_center_offset_px   道路中心 X 與影像中心 X 的差值
boundary_smoothness     輪廓平滑度分數，越高表示邊界越乾淨
stability_label         unstable / low_confidence / stable
flow_consistency        道路 ROI 內光流方向集中程度
mask_iou_prev           與前一個處理影格的遮罩重疊程度
flicker                 相鄰遮罩 IoU 低於門檻時為 true
```

典型判讀方式：

```text
road_area_ratio < 0.05       沒有有效道路遮罩
0.05-0.20                    低信心道路遮罩
> 0.20                       可用的道路遮罩候選
abs(offset) 變大             道路區域偏左或偏右
smoothness 偏低              邊界破碎或雜訊多
flow_consistency 偏低        道路 ROI 內運動不穩定或方向混雜
```

## 道路面分割 MVP

資料集結構：

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

目標類別：

```text
road_surface
```

標註規則與切分政策：

```text
road_dataset/README.md
road_dataset/LABELING_GUIDE.md
road_dataset/SPLIT_POLICY.md
```

## 自動化

GitHub Actions 會執行：

```text
cd "algorithm/Optical Flow" && python -m pytest
```

## 基準測試

```powershell
python benchmark/run_benchmark.py --videos benchmark/videos --methods classical fused --max-frames 100
python benchmark/run_benchmark.py --videos benchmark/videos --methods yolo-seg yolo-seg-fused --weights runs/road_segmentation/yolov8n_seg_mvp/weights/best.pt
```

基準測試輸出：

```text
benchmark/reports/benchmark_metrics.csv
```

基準測試指標包含 `stable_rate`、`flicker_rate`、`mean_mask_iou_prev` 與 `runtime_fps`。
