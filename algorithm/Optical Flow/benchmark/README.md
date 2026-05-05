# Benchmark Protocol

This folder defines a repeatable benchmark protocol for road surface perception experiments.

## Goal

Run the same input videos through multiple methods and compare:

- `mean_area`
- `mean_smoothness`
- `stable_rate`
- `runtime_fps`

## Expected Layout

```text
benchmark/
  videos/
  reports/
  report_template.md
  run_benchmark.py
```

Input videos are local benchmark assets and should be curated before publishing.

## Methods

Initial comparison targets:

- `classical`
- `fused`
- `yolo-seg`
- `yolo-seg-fused`
