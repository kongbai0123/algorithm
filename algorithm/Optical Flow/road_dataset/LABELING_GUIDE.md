# Road Surface Labeling Guide

## Class

```text
0: road_surface
```

## Include

- Visible asphalt road surface.
- Visible concrete road surface.
- Shadows that still fall on road surface.
- Lane markings when they lie on road surface.
- Zebra crossings when they lie on road surface.
- Wet road surface or puddles when the road surface remains visible.

## Exclude

- Sidewalks, unless the task definition changes.
- Vehicles, pedestrians, cyclists, poles, buildings, walls, sky, and traffic lights.
- Hidden road regions behind vehicles or other occluders.
- Road regions inferred from context but not visible in the image.

## Principle

Label what is visibly road surface in the image, not what you know exists behind objects.
