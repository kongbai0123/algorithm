from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .farneback_flow import FarnebackConfig, farneback_flow
from .horn_schunck import HornSchunckConfig, multiresolution_horn_schunck
from .lucas_kanade import LucasKanadeConfig, lucas_kanade_sparse_flow, sparse_points_to_flow


@dataclass(frozen=True)
class FlowResult:
    u: np.ndarray
    v: np.ndarray
    valid_mask: np.ndarray | None
    backend: str


class FlowEstimator(Protocol):
    backend: str

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        ...


@dataclass(frozen=True)
class HornSchunckEstimator:
    config: HornSchunckConfig = HornSchunckConfig(alpha=8.0, iterations=120, pyramid_levels=4, warps_per_level=3)
    backend: str = "horn_schunck"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        u, v = multiresolution_horn_schunck(prev_frame, curr_frame, self.config)
        return FlowResult(u=u, v=v, valid_mask=None, backend=self.backend)


@dataclass(frozen=True)
class FarnebackEstimator:
    config: FarnebackConfig = FarnebackConfig()
    backend: str = "farneback"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        u, v = farneback_flow(prev_frame, curr_frame, self.config)
        return FlowResult(u=u, v=v, valid_mask=None, backend=self.backend)


@dataclass(frozen=True)
class LucasKanadeEstimator:
    config: LucasKanadeConfig = LucasKanadeConfig()
    backend: str = "lucas_kanade"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        start, end, _ = lucas_kanade_sparse_flow(prev_frame, curr_frame, mask, self.config)
        u, v, valid = sparse_points_to_flow(start, end, prev_frame.shape[:2])
        return FlowResult(u=u, v=v, valid_mask=valid, backend=self.backend)


@dataclass(frozen=True)
class PWCNetEstimator:
    checkpoint: str | None = None
    device: str = "cpu"
    backend: str = "pwcnet"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        raise NotImplementedError(
            "PWC-Net backend is reserved for benchmark integration. "
            "Provide a concrete adapter and checkpoint before enabling this backend."
        )


def create_flow_estimator(name: str, hs_iterations: int = 120) -> FlowEstimator:
    normalized = name.lower().replace("-", "_")
    if normalized in {"horn_schunck", "hs"}:
        return HornSchunckEstimator(
            HornSchunckConfig(alpha=8.0, iterations=hs_iterations, pyramid_levels=4, warps_per_level=3)
        )
    if normalized == "farneback":
        return FarnebackEstimator()
    if normalized in {"lucas_kanade", "lk"}:
        return LucasKanadeEstimator()
    if normalized in {"pwc", "pwcnet", "pwc_net"}:
        return PWCNetEstimator()
    raise ValueError(f"Unknown flow backend: {name}")
