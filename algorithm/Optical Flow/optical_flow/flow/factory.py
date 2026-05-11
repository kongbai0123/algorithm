from __future__ import annotations
from .base import FlowEstimator, PWCNetEstimator
from .horn_schunck import HornSchunckEstimator
from .farneback import FarnebackEstimator
from .lk import LucasKanadeEstimator
from ..horn_schunck import HornSchunckConfig

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
