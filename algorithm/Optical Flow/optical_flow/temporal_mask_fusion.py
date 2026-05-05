from __future__ import annotations

import cv2
import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    if a.shape != b.shape:
        raise ValueError("mask shapes must match")
    union = np.logical_or(a, b).sum()
    if int(union) == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def warp_mask_with_flow(mask: np.ndarray, u: np.ndarray, v: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.float32)
    if binary.shape != u.shape or binary.shape != v.shape:
        raise ValueError("mask, u, and v must share the same shape")

    height, width = binary.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    warped = cv2.remap(
        binary,
        grid_x - u.astype(np.float32),
        grid_y - v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.where(warped >= threshold, 255, 0).astype(np.uint8)


def fuse_masks(current_mask: np.ndarray, propagated_mask: np.ndarray, current_weight: float = 0.7) -> np.ndarray:
    if current_mask.shape != propagated_mask.shape:
        raise ValueError("mask shapes must match")
    if not 0.0 <= current_weight <= 1.0:
        raise ValueError("current_weight must be between 0 and 1")

    current = (np.asarray(current_mask) > 0).astype(np.float32)
    propagated = (np.asarray(propagated_mask) > 0).astype(np.float32)
    fused = current_weight * current + (1.0 - current_weight) * propagated
    return np.where(fused >= 0.5, 255, 0).astype(np.uint8)


def flicker_flag(previous_mask: np.ndarray | None, current_mask: np.ndarray, iou_threshold: float = 0.75) -> bool:
    if previous_mask is None:
        return False
    return mask_iou(previous_mask, current_mask) < iou_threshold
