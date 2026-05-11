from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..farneback_flow import FarnebackConfig, farneback_flow
from .base import FlowResult, FlowEstimator

@dataclass(frozen=True)
class FarnebackEstimator:
    config: FarnebackConfig = FarnebackConfig()
    backend: str = "farneback"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        u, v = farneback_flow(prev_frame, curr_frame, self.config)
        return FlowResult(u=u, v=v, valid_mask=None, backend=self.backend)
