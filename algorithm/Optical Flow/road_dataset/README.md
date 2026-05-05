# Road Surface Dataset

This dataset is designed for single-class YOLO segmentation.

## Class

```text
0: road_surface
```

## Labeling Rules

- Label only visible road surface.
- Do not label hidden or occluded road regions.
- Shadows do not change the road_surface label.
- Lane markings and zebra crossings may be included when they lie on road surface.
- Do not label sidewalks unless the task definition changes.
- Do not label vehicles, pedestrians, sky, walls, buildings, or traffic lights.
- If a vehicle blocks the road, do not hallucinate the road behind it.

## Split Policy

Split by scene or video segment, not by random frames.

Example:

```text
scene_01 -> train
scene_02 -> train
scene_03 -> val
scene_04 -> test
```

Do not put adjacent frames from the same continuous clip into both train and val.
That would leak near-duplicate images into validation and overstate model quality.

## Structure

```text
road_dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  road.yaml
```

