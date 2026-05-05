from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import cv2

from .flow_backends import create_flow_estimator
from .flow_metrics import road_masked_flow_stats
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
    methods: list[str] | None = None,
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
    selected_methods = methods or ["horn_schunck", "farneback", "lucas_kanade"]

    for method in selected_methods:
        estimator = create_flow_estimator(method, hs_iterations=hs_iterations)
        flow = estimator.estimate(prev, curr, road_mask)
        stats = road_masked_flow_stats(flow.u, flow.v, road_mask, flow.valid_mask)
        rows.append({"method": flow.backend, "flow_backend": flow.backend, **asdict(stats)})

    write_flow_comparison_csv(csv_path, rows)
    return rows
