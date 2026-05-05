from __future__ import annotations

import numpy as np


def endpoint_error(
    estimated_u: np.ndarray,
    estimated_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> float:
    eu = np.asarray(estimated_u, dtype=np.float32)
    ev = np.asarray(estimated_v, dtype=np.float32)
    tu = np.asarray(target_u, dtype=np.float32)
    tv = np.asarray(target_v, dtype=np.float32)
    if eu.shape != ev.shape or eu.shape != tu.shape or eu.shape != tv.shape:
        raise ValueError("all flow arrays must share the same shape")
    return float(np.mean(np.sqrt((eu - tu) ** 2 + (ev - tv) ** 2)))


def average_angular_error(
    estimated_u: np.ndarray,
    estimated_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    eu = np.asarray(estimated_u, dtype=np.float32)
    ev = np.asarray(estimated_v, dtype=np.float32)
    tu = np.asarray(target_u, dtype=np.float32)
    tv = np.asarray(target_v, dtype=np.float32)
    if eu.shape != ev.shape or eu.shape != tu.shape or eu.shape != tv.shape:
        raise ValueError("all flow arrays must share the same shape")

    dot = eu * tu + ev * tv
    norm = np.sqrt(eu * eu + ev * ev + epsilon) * np.sqrt(tu * tu + tv * tv + epsilon)
    cosine = np.clip(dot / (norm + epsilon), -1.0, 1.0)
    return float(np.mean(np.arccos(cosine)))

