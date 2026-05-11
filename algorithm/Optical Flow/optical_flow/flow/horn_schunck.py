from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..horn_schunck import HornSchunckConfig, multiresolution_horn_schunck
from .base import FlowResult, FlowEstimator

@dataclass(frozen=True)
class HornSchunckEstimator:
    config: HornSchunckConfig = HornSchunckConfig(alpha=8.0, iterations=120, pyramid_levels=4, warps_per_level=3)
    backend: str = "horn_schunck"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        u, v = multiresolution_horn_schunck(prev_frame, curr_frame, self.config)
        return FlowResult(u=u, v=v, valid_mask=None, backend=self.backend)
