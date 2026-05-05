# Dataset Audit Checklist

Use this checklist before training or publishing a dataset split.

- Every label file uses YOLO segmentation polygon format.
- Only visible road surface is labeled.
- Occluded road regions are not hallucinated.
- Sidewalks are excluded unless the task definition changes.
- Train, validation, and test splits are separated by scene or video segment.
- Adjacent frames from the same continuous clip do not cross splits.
- Validation contains difficult cases such as shadows, rain, markings, parked vehicles, reflections, and damaged road.
- `road.yaml` paths match the actual dataset structure.
- A small visual spot-check has been performed before training.
