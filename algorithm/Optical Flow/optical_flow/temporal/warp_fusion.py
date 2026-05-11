from __future__ import annotations
import numpy as np
from ..image_ops import warp_bilinear
from .base import TemporalFusion
from .ema_fusion import EMATemporalFusion

class WarpTemporalFusion(TemporalFusion):
    def __init__(self, alpha: float = 0.7, fallback_to_ema: bool = True):
        self.alpha = alpha
        self.fallback_to_ema = fallback_to_ema
        self.ema_fallback = EMATemporalFusion(alpha=alpha)
        
    def fuse(self, curr_mask: np.ndarray, prev_mask: np.ndarray, flow: np.ndarray | None = None) -> np.ndarray:
        """
        Fuse current mask with previous mask using optical flow compensation.
        
        Args:
            curr_mask: Binary mask or score map from current frame.
            prev_mask: Binary mask or score map from previous frame.
            flow: Optical flow result object or tuple (u, v).
            
        Returns:
            Fused binary mask.
        """
        if flow is None:
            if self.fallback_to_ema:
                return self.ema_fallback.fuse(curr_mask, prev_mask)
            else:
                return curr_mask
                
        # Try to extract u and v from flow
        try:
            u = flow.u
            v = flow.v
        except AttributeError:
            # Maybe it's a tuple (u, v)
            if isinstance(flow, tuple) and len(flow) == 2:
                u, v = flow
            else:
                # Fallback if flow format is unknown
                if self.fallback_to_ema:
                    return self.ema_fallback.fuse(curr_mask, prev_mask)
                return curr_mask
                
        # Convert masks to float scores in [0, 1]
        curr_score = (curr_mask > 0).astype(np.float32)
        prev_score = (prev_mask > 0).astype(np.float32)
        
        # Warp prev_score using flow
        try:
            warped_prev_score = warp_bilinear(prev_score, u, v)
        except Exception:
            # Fallback if warp fails (e.g., shape mismatch)
            if self.fallback_to_ema:
                return self.ema_fallback.fuse(curr_mask, prev_mask)
            return curr_mask
            
        # Fuse scores
        fused = self.alpha * curr_score + (1.0 - self.alpha) * warped_prev_score
        
        # Threshold to create binary mask
        fused_mask = np.where(fused >= 0.5, 255, 0).astype(np.uint8)
        
        return fused_mask
