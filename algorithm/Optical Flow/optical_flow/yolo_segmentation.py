from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mask_postprocess import select_road_mask_from_yolo_result


@dataclass(frozen=True)
class YoloSegmentationConfig:
    weights: str
    imgsz: int = 640
    conf: float = 0.25
    min_area_ratio: float = 0.01
    bottom_roi_ratio: float = 0.35

    def validate(self) -> None:
        if not self.weights:
            raise ValueError("weights must be provided for YOLO segmentation")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if not 0.0 < self.conf < 1.0:
            raise ValueError("conf must be between 0 and 1")
        if not 0.0 <= self.min_area_ratio < 1.0:
            raise ValueError("min_area_ratio must be between 0 and 1")
        if not 0.0 < self.bottom_roi_ratio <= 1.0:
            raise ValueError("bottom_roi_ratio must be between 0 and 1")


class YoloRoadSegmenter:
    def __init__(self, config: YoloSegmentationConfig) -> None:
        config.validate()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("YOLO segmentation requires the optional 'ultralytics' package.") from exc

        self.config = config
        self.model = YOLO(config.weights)

    def segment(self, image: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            source=image,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            verbose=False,
        )[0]
        return select_road_mask_from_yolo_result(
            result,
            image.shape,
            min_area_ratio=self.config.min_area_ratio,
            bottom_roi_ratio=self.config.bottom_roi_ratio,
        )
