from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

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
class PWCNetEstimator:
    checkpoint: str | None = None
    device: str = "cpu"
    backend: str = "pwcnet"

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray | None = None) -> FlowResult:
        raise NotImplementedError(
            "PWC-Net backend is reserved for benchmark integration. "
            "Provide a concrete adapter and checkpoint before enabling this backend."
        )
