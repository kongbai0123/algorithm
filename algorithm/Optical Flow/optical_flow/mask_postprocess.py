from __future__ import annotations

import cv2
import numpy as np


def postprocess_road_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    min_area_ratio: float = 0.01,
    bottom_roi_ratio: float = 0.35,
    close_kernel_size: int = 17,
    open_kernel_size: int = 5,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), connectivity=8)
    if component_count <= 1:
        return cleaned

    min_area = int(height * width * min_area_ratio)
    bottom_band = np.zeros((height, width), dtype=np.uint8)
    bottom_band[int(height * (1.0 - bottom_roi_ratio)) :, :] = 1
    result = np.zeros((height, width), dtype=np.uint8)

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        if not np.any(component & (bottom_band > 0)):
            continue
        result[component] = 255

    if np.any(result):
        return result

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def select_road_mask_from_yolo_result(
    result,
    image_shape: tuple[int, int] | tuple[int, int, int],
    min_area_ratio: float = 0.01,
    bottom_roi_ratio: float = 0.35,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    mask_data = masks.data.detach().cpu().numpy()
    class_ids = None
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.cls is not None:
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    combined = np.zeros((height, width), dtype=np.uint8)
    for index, mask in enumerate(mask_data):
        if class_ids is not None and index < len(class_ids) and class_ids[index] != 0:
            continue
        resized = cv2.resize((mask > 0.5).astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, resized)

    return postprocess_road_mask(
        combined,
        image_shape,
        min_area_ratio=min_area_ratio,
        bottom_roi_ratio=bottom_roi_ratio,
    )

