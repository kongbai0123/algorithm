from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from optical_flow import HornSchunckConfig, endpoint_error, multiresolution_horn_schunck


def make_translation_pair(shift_x: int = 4, shift_y: int = 2, size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros((size, size), dtype=np.float32)
    cv2.rectangle(first, (24, 28), (70, 68), 1.0, thickness=-1)
    cv2.circle(first, (48, 48), 12, 0.35, thickness=-1)

    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    second = cv2.warpAffine(first, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return first, second


def main() -> None:
    shift_x, shift_y = 4, 2
    frame1, frame2 = make_translation_pair(shift_x, shift_y)
    config = HornSchunckConfig(alpha=1.2, iterations=250, pyramid_levels=4, warps_per_level=4)
    u, v = multiresolution_horn_schunck(frame1, frame2, config)

    target_u = np.full_like(u, shift_x, dtype=np.float32)
    target_v = np.full_like(v, shift_y, dtype=np.float32)
    moving_region = frame1 > 0.05
    epe = endpoint_error(u[moving_region], v[moving_region], target_u[moving_region], target_v[moving_region])

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    y, x = np.mgrid[0 : frame1.shape[0] : 6, 0 : frame1.shape[1] : 6]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(frame1, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Frame 1")
    axes[1].imshow(frame2, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Frame 2")
    axes[2].imshow(frame1, cmap="gray", vmin=0, vmax=1)
    axes[2].quiver(x, y, u[::6, ::6], v[::6, ::6], color="tab:red", angles="xy", scale_units="xy", scale=1)
    axes[2].set_title(f"MR-HS flow, EPE={epe:.3f}")
    for axis in axes:
        axis.axis("off")
    fig.savefig(output_dir / "synthetic_flow.png", dpi=150)
    print(f"moving-region EPE: {epe:.4f}")
    print(f"mean u/v in moving region: {u[moving_region].mean():.3f}, {v[moving_region].mean():.3f}")


if __name__ == "__main__":
    main()

