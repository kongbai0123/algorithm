from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run repeatable road perception benchmarks.")
    parser.add_argument("--videos", default="benchmark/videos", help="benchmark video directory")
    parser.add_argument("--methods", nargs="+", default=["classical"], help="methods to evaluate")
    parser.add_argument("--out", default="benchmark/reports", help="benchmark output directory")
    parser.add_argument("--weights", default=None, help="YOLO weights for yolo-seg methods")
    parser.add_argument("--max-frames", type=int, default=None, help="optional frame cap")
    args = parser.parse_args()

    video_dir = Path(args.videos)
    if not video_dir.is_absolute():
        video_dir = base_dir / video_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
    if not videos:
        raise FileNotFoundError(f"No benchmark videos found under: {video_dir}")

    rows: list[dict[str, object]] = []
    for method in args.methods:
        for video_path in videos:
            run_out = out_dir / method / video_path.stem
            command = [
                sys.executable,
                str(base_dir / "run_road_pipeline.py"),
                "--source",
                str(video_path),
                "--mode",
                "video",
                "--method",
                method,
                "--out",
                str(run_out),
                "--progress-every",
                "30",
            ]
            if args.max_frames is not None:
                command.extend(["--max-frames", str(args.max_frames)])
            if method.startswith("yolo"):
                if args.weights is None:
                    raise ValueError(f"{method} requires --weights")
                command.extend(["--weights", args.weights])

            started = time.perf_counter()
            subprocess.run(command, cwd=base_dir, check=True)
            elapsed = max(time.perf_counter() - started, 1e-6)
            summary = _load_summary(run_out / "video_summary.json")
            frame_count = int(summary.get("image_count", 0))
            summary_rows = summary.get("rows", [])
            stable_count = sum(1 for row in summary_rows if row.get("stability_label") == "stable")
            flicker_count = sum(1 for row in summary_rows if row.get("flicker") is True)
            iou_values = [
                float(row["mask_iou_prev"])
                for row in summary_rows
                if row.get("mask_iou_prev") is not None
            ]
            rows.append(
                {
                    "method": method,
                    "video": video_path.name,
                    "mean_area": summary.get("mean_road_area_ratio", 0.0),
                    "mean_smoothness": summary.get("mean_boundary_smoothness", 0.0),
                    "stable_rate": stable_count / frame_count if frame_count else 0.0,
                    "flicker_rate": flicker_count / max(frame_count - 1, 1) if frame_count > 1 else 0.0,
                    "mean_mask_iou_prev": sum(iou_values) / len(iou_values) if iou_values else 0.0,
                    "runtime_fps": frame_count / elapsed,
                }
            )

    report_path = out_dir / "benchmark_metrics.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "video",
                "mean_area",
                "mean_smoothness",
                "stable_rate",
                "flicker_rate",
                "mean_mask_iou_prev",
                "runtime_fps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"benchmark_csv={report_path}")


if __name__ == "__main__":
    main()
