from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class TemporalFusion(ABC):
    @abstractmethod
    def fuse(self, curr_mask: np.ndarray, prev_mask: np.ndarray, flow: np.ndarray | None = None) -> np.ndarray:
        """
        Fuse current mask with previous mask.
        
        Args:
            curr_mask: Mask from current frame.
            prev_mask: Mask from previous frame.
            flow: Optional optical flow between frames.
            
        Returns:
            Fused mask.
        """
        pass
