from __future__ import annotations
import numpy as np
from ..yolo_segmentation import YoloRoadSegmenter, YoloSegmentationConfig
from .base import RoadSegmenter

class YoloSegmenterWrapper(RoadSegmenter):
    def __init__(self, config: YoloSegmentationConfig):
        self.segmenter = YoloRoadSegmenter(config)
        
    def segment(self, frame: np.ndarray) -> np.ndarray:
        return self.segmenter.segment(frame)
