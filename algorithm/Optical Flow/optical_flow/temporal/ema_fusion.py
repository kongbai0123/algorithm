from __future__ import annotations
import numpy as np
from .base import TemporalFusion

class EMATemporalFusion(TemporalFusion):
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        
    def fuse(self, curr_mask: np.ndarray, prev_mask: np.ndarray, flow: np.ndarray | None = None) -> np.ndarray:
        curr_score = (curr_mask > 0).astype(np.float32)
        prev_score = (prev_mask > 0).astype(np.float32)
        
        fused = self.alpha * curr_score + (1.0 - self.alpha) * prev_score
        fused_mask = np.where(fused >= 0.5, 255, 0).astype(np.uint8)
        return fused_mask
