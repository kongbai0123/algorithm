from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_training_report(run_dir: Path) -> None:
    # 確保輸出目錄存在
    run_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = run_dir / "results.csv"
    # 將日誌統一存放在 runs 底下 (run_dir.parent.parent 就是 runs/)
    unified_log_file = run_dir.parent.parent / "training_history_log.md"
    unified_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not results_file.exists():
        with open(unified_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n\n## 實驗: {run_dir.name}\n🔴 **無法生成報告**：找不到 {run_dir.name}/results.csv\n")
        return

    # 讀取 CSV
    epochs = []
    val_seg_losses = []
    train_seg_losses = []
    map50_m = []
    
    with open(results_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 移除標題欄位可能的前後空白
        reader.fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
        for row in reader:
            try:
                epochs.append(int(row['epoch'].strip()))
                val_seg_losses.append(float(row.get('val/seg_loss', row.get('val/seg_loss', 0)).strip()))
                train_seg_losses.append(float(row.get('train/seg_loss', row.get('train/seg_loss', 0)).strip()))
                map50_m.append(float(row.get('metrics/mAP50(M)', row.get('metrics/mAP50(M)', 0)).strip()))
            except (ValueError, KeyError):
                continue
                
    if not epochs:
        with open(unified_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n\n## 實驗: {run_dir.name}\n🔴 **無法生成報告**：results.csv 沒有有效數據\n")
        return

    best_map = max(map50_m) if map50_m else 0
    final_val_loss = val_seg_losses[-1] if val_seg_losses else 0
    min_val_loss = min(val_seg_losses) if val_seg_losses else 0
    
    report = [
        f"\n\n---",
        f"\n## 實驗紀錄: {run_dir.name}",
        f"**總訓練輪數 (Epochs)**: {len(epochs)} | **最佳 mAP@0.5**: {best_map:.4f}",
        "",
    ]
    
    is_bad = False
    reasons = []
    
    if best_map < 0.5:
        is_bad = True
        reasons.append("- **精確度過低**：模型的最佳 mAP50 低於 0.5，代表模型幾乎無法正確標出路面。可能原因：資料量太少、標籤品質不佳、或是訓練輪數不足。")
        
    if len(val_seg_losses) > 10 and final_val_loss > min_val_loss * 1.2:
        is_bad = True
        reasons.append("- **嚴重過擬合 (Overfitting)**：驗證集的損失值在訓練後期不降反升。可能原因：資料集太小導致模型死背訓練集、缺少足夠的背景圖片作為負樣本。")
        
    if best_map > 0.95 and len(epochs) < 50:
        reasons.append("- **潛在的資料洩漏或任務過於簡單**：模型在極短時間內就達到 0.95 以上的準確度。請檢查驗證集是否混入了訓練集的圖片。")

    if is_bad:
        report.append("### 🔴 總評：訓練品質不良 (BAD)")
        report.append("\n**為何會壞 (診斷原因)：**")
        report.extend(reasons)
        report.append("\n**改進建議：** 增加資料量、補充純背景圖片、確認多邊形標籤精確度。")
    else:
        report.append("### 🟢 總評：訓練品質良好或及格 (GOOD / ACCEPTABLE)")
        if reasons:
            report.append("\n**觀察與警告：**")
            report.extend(reasons)
        report.append("\n模型具備基礎的路面辨識能力，可進行後續的 Benchmark 或推論測試。")
        
    with open(unified_log_file, "a", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"\n[完成] 訓練日誌已附加至統一檔案: {unified_log_file}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a road_surface segmentation model on PC.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLO segmentation checkpoint or model name")
    parser.add_argument("--data", default="train/prepared/road.yaml", help="dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch", type=int, default=8, help="training batch size")
    parser.add_argument("--project", default="runs/road_segmentation", help="training project output folder")
    parser.add_argument("--name", default="road_surface_custom_v2", help="run name")
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
    
    # 訓練結束後，自動生成分析日誌
    run_dir = Path(project_path) / args.name
    generate_training_report(run_dir)
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
