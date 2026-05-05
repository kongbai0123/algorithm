from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import cv2

from optical_flow import (
    FarnebackConfig,
    HornSchunckConfig,
    LucasKanadeConfig,
    detect_road_from_path,
    farneback_flow,
    lucas_kanade_sparse_flow,
    multiresolution_horn_schunck,
    road_masked_flow_stats,
    sparse_points_to_flow,
)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base_dir / path).resolve()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["method", *[key for key in rows[0] if key != "method"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare optical-flow methods inside the detected road mask ROI.")
    parser.add_argument("--prev", required=True, help="previous image path")
    parser.add_argument("--curr", required=True, help="current image path")
    parser.add_argument("--out", default="outputs/flow_compare/flow_metrics.csv", help="CSV output path")
    parser.add_argument("--hs-iterations", type=int, default=120, help="Horn-Schunck iterations")
    args = parser.parse_args()

    prev_path = _resolve_path(args.prev, base_dir)
    curr_path = _resolve_path(args.curr, base_dir)
    out_path = _resolve_path(args.out, base_dir)
    if out_path.suffix.lower() != ".csv":
        out_path = out_path / "flow_metrics.csv"

    prev = cv2.imread(str(prev_path))
    curr = cv2.imread(str(curr_path))
    if prev is None:
        raise FileNotFoundError(f"Unable to read previous image: {prev_path}")
    if curr is None:
        raise FileNotFoundError(f"Unable to read current image: {curr_path}")
    if prev.shape != curr.shape:
        raise ValueError("prev and curr images must have identical shape")

    _, road_mask, _ = detect_road_from_path(curr_path)
    rows: list[dict[str, object]] = []

    hs_u, hs_v = multiresolution_horn_schunck(
        prev,
        curr,
        HornSchunckConfig(alpha=8.0, iterations=args.hs_iterations, pyramid_levels=4, warps_per_level=3),
    )
    rows.append({"method": "horn_schunck", **asdict(road_masked_flow_stats(hs_u, hs_v, road_mask))})

    fb_u, fb_v = farneback_flow(prev, curr, FarnebackConfig())
    rows.append({"method": "farneback", **asdict(road_masked_flow_stats(fb_u, fb_v, road_mask))})

    start, end, _ = lucas_kanade_sparse_flow(prev, curr, road_mask, LucasKanadeConfig())
    lk_u, lk_v, lk_valid = sparse_points_to_flow(start, end, road_mask.shape)
    rows.append({"method": "lucas_kanade", **asdict(road_masked_flow_stats(lk_u, lk_v, road_mask, lk_valid))})

    _write_rows(out_path, rows)
    for row in rows:
        print(
            f"{row['method']}: count={row['valid_pixel_count']} "
            f"mean_mag={float(row['mean_magnitude']):.4f} "
            f"angle={row['direction_angle_deg']} "
            f"consistency={float(row['consistency']):.3f}"
        )
    print(f"metrics_csv={out_path}")


if __name__ == "__main__":
    main()
