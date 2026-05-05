from __future__ import annotations

import argparse
from pathlib import Path

from optical_flow import detect_fused_road_from_paths


def _resolve_input_path(value: str, base_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    fallback = base_dir / candidate
    return fallback.resolve()


def _resolve_output_path(value: str, base_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Detect road by fusing static road score and optical flow.")
    parser.add_argument("--prev", required=True, help="previous image path")
    parser.add_argument("--curr", required=True, help="current image path")
    parser.add_argument("--out", default="outputs/fusion_debug", help="output debug folder")
    args = parser.parse_args()

    prev_path = _resolve_input_path(args.prev, base_dir)
    curr_path = _resolve_input_path(args.curr, base_dir)
    output_dir = _resolve_output_path(args.out, base_dir)

    mask, score, _ = detect_fused_road_from_paths(prev_path, curr_path, output_dir)
    road_ratio = float((mask > 0).mean())
    print(f"road_ratio={road_ratio:.3f}")
    print(f"score_mean={float(score.mean()):.3f}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
