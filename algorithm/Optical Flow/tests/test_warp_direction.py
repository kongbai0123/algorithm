from __future__ import annotations
import numpy as np
import pytest
from optical_flow.image_ops import warp_bilinear

def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)

def create_synthetic_case(shift_x: int, shift_y: int, shape=(100, 100)):
    # Create a square mask in the center of the previous frame
    prev_mask = np.zeros(shape, dtype=np.uint8)
    prev_mask[30:70, 30:70] = 255
    
    # Create current mask by shifting the square
    curr_mask = np.zeros(shape, dtype=np.uint8)
    curr_mask[30+shift_y:70+shift_y, 30+shift_x:70+shift_x] = 255
    
    # Flow estimated from current to previous (backward flow)
    # For a pixel at (X, Y) in current, its source in previous was (X - shift_x, Y - shift_y)
    # So the flow vector points to (-shift_x, -shift_y)
    u = np.full(shape, -shift_x, dtype=np.float32)
    v = np.full(shape, -shift_y, dtype=np.float32)
    
    return prev_mask, curr_mask, u, v

@pytest.mark.parametrize("name, shift_x, shift_y", [
    ("right", 10, 0),
    ("left", -10, 0),
    ("down", 0, 10),
    ("up", 0, -10),
    ("diagonal", 10, 10),
    ("zero", 0, 0),
])
def test_warp_direction(name, shift_x, shift_y):
    prev_mask, curr_mask, u, v = create_synthetic_case(shift_x, shift_y)
    
    # Warp prev_mask to current using the flow
    # warp_bilinear does: map_x = grid_x + u
    warped = warp_bilinear(prev_mask, u, v)
    
    # Convert to binary for IoU calculation
    warped_mask = np.where(warped >= 0.5, 255, 0).astype(np.uint8)
    
    iou = mask_iou(curr_mask, warped_mask)
    print(f"Test {name}: IoU = {iou:.4f}")
    assert iou > 0.95
