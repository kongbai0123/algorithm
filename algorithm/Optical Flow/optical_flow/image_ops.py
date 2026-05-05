from __future__ import annotations

import cv2
import numpy as np


def as_float_gray(image: np.ndarray) -> np.ndarray:
    """Return a 2-D float32 image in [0, 1]."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {array.shape}")

    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("Image must not be empty")
    if array.max() > 1.0 or array.min() < 0.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    if levels < 1:
        raise ValueError("levels must be >= 1")

    pyramid = [as_float_gray(image)]
    for _ in range(1, levels):
        current = pyramid[-1]
        if min(current.shape) < 8:
            break
        smoothed = cv2.GaussianBlur(current, (5, 5), 0)
        downsampled = cv2.resize(
            smoothed,
            (max(1, current.shape[1] // 2), max(1, current.shape[0] // 2)),
            interpolation=cv2.INTER_AREA,
        )
        pyramid.append(downsampled.astype(np.float32, copy=False))
    return pyramid


def resize_flow(u: np.ndarray, v: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    target_h, target_w = shape
    source_h, source_w = u.shape
    if v.shape != u.shape:
        raise ValueError("u and v must have the same shape")

    scale_x = target_w / source_w
    scale_y = target_h / source_h
    resized_u = cv2.resize(u, (target_w, target_h), interpolation=cv2.INTER_LINEAR) * scale_x
    resized_v = cv2.resize(v, (target_w, target_h), interpolation=cv2.INTER_LINEAR) * scale_y
    return resized_u.astype(np.float32), resized_v.astype(np.float32)


def warp_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    frame = as_float_gray(image)
    if frame.shape != u.shape or frame.shape != v.shape:
        raise ValueError("image, u, and v must share the same height and width")

    height, width = frame.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + u.astype(np.float32, copy=False)
    map_y = grid_y + v.astype(np.float32, copy=False)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def spatial_temporal_gradients(frame1: np.ndarray, frame2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = as_float_gray(frame1)
    second = as_float_gray(frame2)
    if first.shape != second.shape:
        raise ValueError("frame1 and frame2 must have the same shape")

    average = 0.5 * (first + second)
    ix = cv2.Sobel(average, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    iy = cv2.Sobel(average, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    it = second - first
    return ix.astype(np.float32), iy.astype(np.float32), it.astype(np.float32)

