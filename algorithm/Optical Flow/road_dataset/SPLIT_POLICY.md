# Dataset Split Policy

## Rule

Split by scene, route segment, or video clip. Do not randomly split adjacent video frames across train, validation, and test sets.

## Recommended Layout

```text
scene_01 -> train
scene_02 -> train
scene_03 -> val
scene_04 -> test
```

## Reason

Adjacent frames from the same continuous clip are near-duplicates. If they appear in both train and validation sets, validation scores become overly optimistic and do not measure generalization.

## Validation Set Coverage

Include difficult conditions intentionally:

- Shadows.
- Night scenes.
- Rain or wet roads.
- Zebra crossings.
- Lane markings.
- Parked vehicles.
- Different camera heights.
- Strong reflections.
- Damaged road surface.
