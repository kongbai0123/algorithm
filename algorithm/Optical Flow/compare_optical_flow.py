from __future__ import annotations

import argparse
from pathlib import Path

from optical_flow import compare_optical_flow_methods


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base_dir / path).resolve()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare optical-flow methods inside the detected road mask ROI.")
    parser.add_argument("--prev", required=True, help="previous image path")
    parser.add_argument("--curr", required=True, help="current image path")
    parser.add_argument("--out", default="outputs/flow_compare/flow_metrics.csv", help="CSV output path")
    parser.add_argument("--hs-iterations", type=int, default=120, help="Horn-Schunck iterations")
    parser.add_argument(
        "--flow-method",
        action="append",
        dest="flow_methods",
        help="flow backend to compare; repeatable. Defaults to horn_schunck, farneback, lucas_kanade",
    )
    args = parser.parse_args()

    prev_path = _resolve_path(args.prev, base_dir)
    curr_path = _resolve_path(args.curr, base_dir)
    out_path = _resolve_path(args.out, base_dir)
    if out_path.suffix.lower() != ".csv":
        out_path = out_path / "flow_metrics.csv"

    rows = compare_optical_flow_methods(prev_path, curr_path, out_path, args.hs_iterations, args.flow_methods)
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
