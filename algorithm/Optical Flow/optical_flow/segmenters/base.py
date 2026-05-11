from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class RoadSegmenter(ABC):
    @abstractmethod
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the road from a given frame.
        
        Args:
            frame: BGR image array.
            
        Returns:
            Binary mask or score map of the road.
        """
        pass
