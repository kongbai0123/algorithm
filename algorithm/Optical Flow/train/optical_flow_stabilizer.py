import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Optical Flow Temporal Stabilization for YOLO Segmentation")
    parser.add_argument("--model", type=str, default="runs/road_segmentation/road_surface_custom_v29_regularized2/weights/best.pt", help="Path to YOLO best.pt")
    parser.add_argument("--input", type=str, default="videos/3_Video Project.mp4", help="Path to video file or folder of images")
    parser.add_argument("--output", type=str, default="algorithm\Optical Flow/outputs/output_stabilized.mp4", help="Output video path")
    parser.add_argument("--alpha", type=float, default=0.85, help="Fusion weight for new frame (0.85 means 85% YOLO, 15% warped)")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.is_absolute():
        # 腳本在 train/ 底下，所以 parent.parent 就是專案根目錄
        model_path = (Path(__file__).resolve().parent.parent / model_path).resolve()
        
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(str(model_path))

    # Setup video capture or image list
    is_video = False
    img_paths = []
    
    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path(__file__).resolve().parent.parent / input_path).resolve()

    if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        cap = cv2.VideoCapture(str(input_path))
        is_video = True
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Loaded video: {args.input} ({total_frames} frames, {fps} FPS, {w}x{h})")
    else:
        # Assume folder of images
        img_dir = input_path
        img_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg')))
        if not img_paths:
            print(f"No images found in {input_path}")
            return
        is_video = False
        total_frames = len(img_paths)
        # Read first image to get dimensions
        first_img = cv2.imread(str(img_paths[0]))
        h, w = first_img.shape[:2]
        fps = 30.0  # Default FPS for image sequence output
        print(f"Loaded image sequence: {input_path} ({total_frames} images, {w}x{h})")
        
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Output will be side-by-side: Original (with Raw YOLO) | Fused Result
    out_w = w * 2
    # Create output directory if it doesn't exist
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (Path(__file__).resolve().parent.parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving output to {out_path}")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, h))

    prev_gray = None
    prev_mask = None
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    frame_idx = 0
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_idx >= len(img_paths):
                break
            frame = cv2.imread(str(img_paths[frame_idx]))
            frame_idx += 1
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Run YOLO inference
        results = model(frame, verbose=False, conf=args.conf)
        
        # Create raw YOLO mask
        yolo_mask = np.zeros((h, w), dtype=np.float32)
        if len(results) > 0 and results[0].masks is not None:
            for poly in results[0].masks.xy:
                if len(poly) >= 3:
                    pts = np.array(poly, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(yolo_mask, [pts], 1.0)
                    
        # 2. Optical Flow and Fusion
        if prev_gray is not None and prev_mask is not None:
            # Calculate flow from CURRENT to PREVIOUS for remap (backward flow)
            flow = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Create remap coordinates
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = map_x.astype(np.float32) + flow[..., 0]
            map_y = map_y.astype(np.float32) + flow[..., 1]
            
            # Warp previous mask to current frame
            warped_mask = cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR)
            
            # Fuse masks: Current YOLO + Warped Previous
            fused_mask = args.alpha * yolo_mask + (1.0 - args.alpha) * warped_mask
        else:
            # First frame or no previous frame
            fused_mask = yolo_mask
            
        # 3. Visualization
        # Left side: Raw YOLO prediction (Red)
        left_img = frame.copy()
        left_img[yolo_mask > 0.5] = left_img[yolo_mask > 0.5] * 0.5 + np.array([0, 0, 255]) * 0.5
        cv2.putText(left_img, "Raw YOLOv8-seg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Right side: Fused stable prediction (Green)
        right_img = frame.copy()
        right_img[fused_mask > 0.5] = right_img[fused_mask > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.putText(right_img, "Temporal Fused (Flow)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine side-by-side
        combined = np.hstack([left_img, right_img])
        
        # Write to video
        writer.write(combined)
        
        # Update previous frame data
        prev_gray = gray
        prev_mask = fused_mask
        pbar.update(1)
        
    # Cleanup
    pbar.close()
    if is_video:
        cap.release()
    writer.release()
    print(f"\nProcessing complete! Video saved to {args.output}")

if __name__ == "__main__":
    main()
