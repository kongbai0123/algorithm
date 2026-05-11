from __future__ import annotations
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import json
import csv

class RuntimeProfiler:
    def __init__(self):
        self.timings = defaultdict(float)  # Store total seconds
        self.counts = defaultdict(int)
        self.frame_records = []
        
    @contextmanager
    def time_block(self, name: str):
        """Context manager to time a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name] += elapsed
            self.counts[name] += 1
            
    def record_block(self, name: str, elapsed: float):
        """Record elapsed time for a block manually."""
        self.timings[name] += elapsed
        self.counts[name] += 1
            
    def record_frame(self, frame_index: int, frame_timings: dict[str, float]):
        """Record timings for a single frame. Timings should be in seconds."""
        record = {"frame": frame_index}
        for name, sec in frame_timings.items():
            record[f"{name}_ms"] = sec * 1000
            
        # Calculate total
        total_sec = sum(frame_timings.values())
        record["total_frame_ms"] = total_sec * 1000
        record["fps"] = 1.0 / total_sec if total_sec > 0 else 0.0
        
        self.frame_records.append(record)
        
    def get_summary(self) -> dict[str, object]:
        """Get summarized timings."""
        summary = {}
        for name, total_time in self.timings.items():
            count = self.counts[name]
            summary[name] = {
                "total_sec": total_time,
                "count": count,
                "mean_ms": (total_time / count) * 1000 if count else 0.0
            }
        return summary
        
    def save_results(self, summary_path: Path, metrics_path: Path):
        """Save results to JSON and CSV."""
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=4)
            
        # Save frame metrics
        if self.frame_records:
            # Determine fieldnames
            keys = set()
            for r in self.frame_records:
                keys.update(r.keys())
            
            # Order fieldnames logically
            fieldnames = ["frame"]
            for k in sorted(keys):
                if k != "frame" and k != "total_frame_ms" and k != "fps":
                    fieldnames.append(k)
            fieldnames.extend(["total_frame_ms", "fps"])
            
            with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.frame_records:
                    row = {k: record.get(k, 0.0) for k in fieldnames}
                    writer.writerow(row)
