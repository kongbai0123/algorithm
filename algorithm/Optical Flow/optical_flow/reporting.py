from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .road_analysis import RoadMetrics


def metrics_row(
    filename: str,
    metrics: RoadMetrics,
    mask_path: str | Path,
    overlay_path: str | Path,
    metrics_overlay_path: str | Path,
) -> dict[str, Any]:
    row = asdict(metrics)
    row.update(
        {
            "filename": filename,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "metrics_overlay_path": str(metrics_overlay_path),
        }
    )
    return row


def write_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "filename",
        "road_area_ratio",
        "road_center_x",
        "road_center_offset_px",
        "road_center_offset_ratio",
        "boundary_smoothness",
        "valid_road",
        "stability_label",
        "mask_path",
        "overlay_path",
        "metrics_overlay_path",
    ]
    extra_fields = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extra_fields)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: str | Path, rows: list[dict[str, Any]], run_config: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_count = sum(1 for row in rows if row["valid_road"])
    area_values = [float(row["road_area_ratio"]) for row in rows]
    smoothness_values = [float(row["boundary_smoothness"]) for row in rows]
    payload = {
        "run_config": run_config,
        "image_count": len(rows),
        "valid_road_count": valid_count,
        "mean_road_area_ratio": sum(area_values) / len(area_values) if area_values else 0.0,
        "mean_boundary_smoothness": sum(smoothness_values) / len(smoothness_values) if smoothness_values else 0.0,
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
