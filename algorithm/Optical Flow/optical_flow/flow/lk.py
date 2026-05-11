from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..lucas_kanade import LucasKanadeConfig, lucas_kanade_sparse_flow, sparse_points_to_flow
from .base import FlowResult, FlowEstimator

@dataclass(frozen=True)
class LucasKanadeEstimator:
    config: LucasKanadeConfig = LucasKanadeConfig()
    backend: str = "lucas_kanade"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        start, end, _ = lucas_kanade_sparse_flow(prev_frame, curr_frame, mask, self.config)
        u, v, valid = sparse_points_to_flow(start, end, prev_frame.shape[:2])
        return FlowResult(u=u, v=v, valid_mask=valid, backend=self.backend)
