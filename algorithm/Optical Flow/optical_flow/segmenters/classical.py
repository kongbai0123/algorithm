from __future__ import annotations
import numpy as np
from ..road_detection import RoadDetectionConfig, detect_road
from .base import RoadSegmenter

class ClassicalRoadSegmenter(RoadSegmenter):
    def __init__(self, config: RoadDetectionConfig | None = None):
        self.config = config or RoadDetectionConfig()
        self.config.validate()
        
    def segment(self, frame: np.ndarray) -> np.ndarray:
        mask, _ = detect_road(frame, self.config)
        return mask
