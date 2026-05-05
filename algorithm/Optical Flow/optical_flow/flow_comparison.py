from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import cv2

from .farneback_flow import FarnebackConfig, farneback_flow
from .flow_metrics import road_masked_flow_stats
from .horn_schunck import HornSchunckConfig, multiresolution_horn_schunck
from .lucas_kanade import LucasKanadeConfig, lucas_kanade_sparse_flow, sparse_points_to_flow
from .road_detection import detect_road_from_path


def write_flow_comparison_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = ["method", *[key for key in rows[0] if key != "method"]]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_optical_flow_methods(
    prev_path: str | Path,
    curr_path: str | Path,
    output_path: str | Path,
    hs_iterations: int = 120,
) -> list[dict[str, object]]:
    prev_image_path = Path(prev_path)
    curr_image_path = Path(curr_path)
    csv_path = Path(output_path)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path / "flow_metrics.csv"

    prev = cv2.imread(str(prev_image_path))
    curr = cv2.imread(str(curr_image_path))
    if prev is None:
        raise FileNotFoundError(f"Unable to read previous image: {prev_image_path}")
    if curr is None:
        raise FileNotFoundError(f"Unable to read current image: {curr_image_path}")
    if prev.shape != curr.shape:
        raise ValueError("prev and curr images must have identical shape")

    _, road_mask, _ = detect_road_from_path(curr_image_path)
    rows: list[dict[str, object]] = []

    hs_u, hs_v = multiresolution_horn_schunck(
        prev,
        curr,
        HornSchunckConfig(alpha=8.0, iterations=hs_iterations, pyramid_levels=4, warps_per_level=3),
    )
    rows.append({"method": "horn_schunck", **asdict(road_masked_flow_stats(hs_u, hs_v, road_mask))})

    fb_u, fb_v = farneback_flow(prev, curr, FarnebackConfig())
    rows.append({"method": "farneback", **asdict(road_masked_flow_stats(fb_u, fb_v, road_mask))})

    start, end, _ = lucas_kanade_sparse_flow(prev, curr, road_mask, LucasKanadeConfig())
    lk_u, lk_v, lk_valid = sparse_points_to_flow(start, end, road_mask.shape)
    rows.append({"method": "lucas_kanade", **asdict(road_masked_flow_stats(lk_u, lk_v, road_mask, lk_valid))})

    write_flow_comparison_csv(csv_path, rows)
    return rows
