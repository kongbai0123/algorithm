from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ExponentialSmoother:
    alpha: float = 0.7
    value: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

    def update(self, current: float) -> float:
        if self.value is None:
            self.value = float(current)
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * float(current)
        return self.value


@dataclass
class MajorityVoteSmoother:
    window_size: int = 5
    required_true: int = 4
    values: deque[bool] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 1 <= self.required_true <= self.window_size:
            raise ValueError("required_true must be between 1 and window_size")

    def update(self, current: bool) -> bool:
        self.values.append(bool(current))
        while len(self.values) > self.window_size:
            self.values.popleft()
        return sum(self.values) >= self.required_true
