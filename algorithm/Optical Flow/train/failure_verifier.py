import argparse
import csv
import json
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def create_binary_mask(txt_path: Path, img_shape: tuple, valid_classes: list = None) -> np.ndarray:
    """Reads YOLO polygon format and returns a binary mask of shape (H, W)."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not txt_path.exists():
        return mask
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # parts[0] is class_id
            if valid_classes is not None and parts[0] not in valid_classes:
                continue
            coords = [float(p) for p in parts[1:]]
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
    return mask

def synthesize_comparison(img: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Creates a side-by-side comparison image."""
    # Left: GT (Green)
    gt_overlay = img.copy()
    gt_overlay[gt_mask == 1] = gt_overlay[gt_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.putText(gt_overlay, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Right: Pred (Red)
    pred_overlay = img.copy()
    pred_overlay[pred_mask == 1] = pred_overlay[pred_mask == 1] * 0.5 + np.array([0, 0, 255]) * 0.5
    cv2.putText(pred_overlay, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return np.hstack([gt_overlay, pred_overlay])

def determine_failure_type(iou, recall, precision, gt_area_ratio, pred_area_ratio, top_region_recall, gt_area, pred_area):
    if gt_area_ratio < 0.001 and pred_area_ratio < 0.001:
        return "true_negative"
        
    if gt_area_ratio < 0.001 and pred_area_ratio >= 0.01:
        return "false_positive"
        
    if gt_area_ratio > 0.02 and pred_area_ratio < 0.01:
        return "false_negative"
        
    if top_region_recall < 0.3 and recall >= 0.5:
        return "perspective_failure"
        
    if gt_area > 0 and pred_area < 0.7 * gt_area and iou < 0.5:
        return "under_segmentation"
        
    if gt_area > 0 and pred_area > 1.3 * gt_area and iou < 0.5:
        return "over_segmentation"
        
    if iou < 0.5:
        return "other_failure"
        
    return "pass"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO best.pt")
    parser.add_argument("--val-dir", type=str, default="prepared/images/val", help="Path to val images dir")
    parser.add_argument("--labels-dir", type=str, default="prepared/labels/val", help="Path to val labels dir")
    parser.add_argument("--output-dir", type=str, default="reports/failure_taxonomy", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO inference confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default=None, help="YOLO inference device (e.g. 0 or cpu)")
    parser.add_argument("--classes", type=str, default=None, help="Comma separated class IDs to keep (e.g. '0,1,2'). Default keeps all.")
    args = parser.parse_args()
    
    valid_classes = args.classes.split(',') if args.classes else None
    
    base_dir = Path(__file__).resolve().parent
    val_dir = base_dir / args.val_dir if not Path(args.val_dir).is_absolute() else Path(args.val_dir)
    labels_dir = base_dir / args.labels_dir if not Path(args.labels_dir).is_absolute() else Path(args.labels_dir)
    out_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    
    if not val_dir.exists():
        print(f"Validation directory not found: {val_dir}")
        return
        
    # Create taxonomy dirs
    taxonomy_classes = ["false_negative", "under_segmentation", "over_segmentation", "perspective_failure", "boundary_collapse", "shadow_confusion", "label_noise", "other_failure", "false_positive", "true_negative", "pass"]
    for t in taxonomy_classes:
        (out_dir / t).mkdir(parents=True, exist_ok=True)
        
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)
    
    report_data = []
    
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        img_paths.extend(list(val_dir.glob(ext)))
        
    for img_path in tqdm(img_paths, desc="Verifying validation set"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 1. Get GT Mask
        txt_path = labels_dir / (img_path.stem + ".txt")
        gt_mask = create_binary_mask(txt_path, img.shape, valid_classes)
        
        # 2. Get Pred Mask
        if args.device:
            results = model(img, verbose=False, imgsz=args.imgsz, conf=args.conf, device=args.device)
        else:
            results = model(img, verbose=False, imgsz=args.imgsz, conf=args.conf)
            
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if len(results) > 0 and results[0].masks is not None:
            cls_idx = results[0].boxes.cls.cpu().numpy()
            for i, poly in enumerate(results[0].masks.xy):
                if valid_classes is not None and str(int(cls_idx[i])) not in valid_classes:
                    continue
                if len(poly) >= 3:
                    pts = np.array(poly, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(pred_mask, [pts], 1)
                
        # 3. Calculate Metrics
        gt_area = np.sum(gt_mask)
        pred_area = np.sum(pred_mask)
        img_area = h * w
        
        intersection = np.sum(np.logical_and(gt_mask, pred_mask))
        union = np.sum(np.logical_or(gt_mask, pred_mask))
        
        iou = intersection / union if union > 0 else 0.0
        recall = intersection / gt_area if gt_area > 0 else 0.0
        precision = intersection / pred_area if pred_area > 0 else 0.0
        
        gt_area_ratio = gt_area / img_area
        pred_area_ratio = pred_area / img_area
        
        # Top region recall (top 1/3)
        top_h = h // 3
        top_gt = gt_mask[:top_h, :]
        top_pred = pred_mask[:top_h, :]
        top_gt_area = np.sum(top_gt)
        top_intersection = np.sum(np.logical_and(top_gt, top_pred))
        top_region_recall = top_intersection / top_gt_area if top_gt_area > 0 else 1.0
        
        # 4. Decision Tree
        failure_type = determine_failure_type(iou, recall, precision, gt_area_ratio, pred_area_ratio, top_region_recall, gt_area, pred_area)
        
        # Check if bottom connected (New for Semantic Corridor Audit)
        bottom_h = max(1, int(h * 0.05))
        is_bottom_connected = bool(np.any(pred_mask[h - bottom_h:, :] == 1)) if pred_area > 0 else True
        
        # 5. Output record
        report_data.append({
            "filename": img_path.name,
            "iou": round(iou, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "gt_area_ratio": round(gt_area_ratio, 4),
            "pred_area_ratio": round(pred_area_ratio, 4),
            "top_region_recall": round(top_region_recall, 4),
            "failure_type": failure_type,
            "bottom_connected": is_bottom_connected
        })
        
        # 6. Synthesize Comparison Image if failed
        if failure_type != "pass":
            comp_img = synthesize_comparison(img, gt_mask, pred_mask)
            save_path = out_dir / failure_type / img_path.name
            cv2.imwrite(str(save_path), comp_img)
            
    # Write CSV
    csv_path = out_dir.parent / "failure_verification_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "iou", "precision", "recall", "gt_area_ratio", "pred_area_ratio", "top_region_recall", "failure_type", "bottom_connected"])
        writer.writeheader()
        for row in report_data:
            writer.writerow(row)
            
    # Write JSON Summary
    summary = defaultdict(int)
    for row in report_data:
        summary[row["failure_type"]] += 1
        
    json_path = out_dir.parent / "failure_verification_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    print("\n--- Failure Verification Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nReports saved to {csv_path} and {json_path}")
    print(f"Visualizations saved in {out_dir}/")

if __name__ == "__main__":
    main()
