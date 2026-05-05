# 基準測試規範

此資料夾定義可重複執行的道路面感知基準測試流程。

## 目標

讓同一批輸入影片經過多種方法後，可以直接比較：

- `mean_area`
- `mean_smoothness`
- `stable_rate`
- `flicker_rate`
- `mean_mask_iou_prev`
- `runtime_fps`

## 預期結構

```text
benchmark/
  videos/
  reports/
  report_template.md
  run_benchmark.py
```

輸入影片屬於本地 benchmark 資產，發布前應先人工整理與確認。

## 方法

初始比較目標：

- `classical`
- `fused`
- `yolo-seg`
- `yolo-seg-fused`
