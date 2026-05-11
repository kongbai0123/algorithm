# Annotation Rules: Traversable Corridor Estimation

This document defines the rules for annotating the "Traversable" class for Phase 4.0 Ontology Refactor. The goal is to move from material classification to **Navigable Path Estimation**.

## Definition of "Traversable"

An area should be annotated as `Traversable` ONLY if it meets the following criteria:

1. **Directionality**: The path has a clear direction for navigation.
2. **Continuity**: The foreground and background are continuous, forming a viable path.
3. **Vehicle Corridor**: It forms a corridor where a vehicle can physically move.
4. **Not just a Flat Texture**: A large flat area (like a plaza or parking lot) is NOT traversable unless it is part of a defined path.
5. **No Major Obstacles**: The path is not blocked by large obstacles.

## Expanded Rules for v29+

### Rule 6: Corridor Width Consistency
- **Do NOT**: Mark the entire flat area just because it has the same texture.
- **DO**: Maintain a reasonable vehicle corridor width. Focus on wheel tracks, path centerlines, and visual convergence directions.
- *Reason*: Prevents the model from equating "flat region" with "traversable".

### Rule 7: Handle Distant Ambiguity
- **Do NOT**: Guess polygons in distant, blurry areas (e.g., far end of a forest trail).
- **DO**: Only annotate the corridor where you have high confidence.
- *Reason*: Prevents the model from learning incorrect geometry priors.

### Rule 8: Obstacle Dominance
- **Do NOT**: Continue annotating traversable regions behind a major obstacle that blocks the path (e.g., fallen trees, large rocks, vehicles).
- *Reason*: Navigability is interrupted by obstacles.

### Rule 9: Ignore/Uncertain Zone Concept
- **Rule**: Avoid forced binary classification (road/non-road) in highly uncertain areas like extreme shadows, heavy glare, or distant blurs. Better to leave unlabeled than to provide false ground truth.
- *Reason*: False GT is more dangerous than missing GT.

### Rule 10: Temporal Annotation Mindset
- **Rule**: When annotating a single frame, consider if the corridor would still be logical in the next frame as the vehicle moves.
- *Reason*: Prepares the data for future optical flow temporal smoothing.

## Classes

| Class ID | Class Name | Description |
| :--- | :--- | :--- |
| 0 | `traversable` | Explicit navigable path forming a vehicle corridor. |

## Phase 4 Roadmap

### Phase 4.0a: Ontology Definition
Establish these rules and ensure semantic consistency.

### Phase 4.0b: Dataset Audit
Identify samples that violate these rules, focusing on:
1. Overly broad masks (Plazas, parking lots).
2. Semantic confusion.
3. Impossible geometry.

### Phase 4.0c: Selective Relabel
Prioritize fixing **High Influence Samples** rather than relabeling everything:
- Massive mask areas.
- Super wide asphalt/gravel areas with no corridor definition.
- Known failure samples from previous versions.

### Phase 4.0d: Single-Class Traversable Training
Train model `v29` on the audited and refactored dataset.

### Phase 4.0e: Temporal Stabilization
Integrate optical flow for temporal smoothing of the mask.
