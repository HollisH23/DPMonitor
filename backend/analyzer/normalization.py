"""Skeleton normalization for the CTR-GCN pipeline.

The CTR-GCN model is sensitive to two nuisance factors that change with
camera placement but have nothing to do with the *movement* itself:

* **Translation** — where in the frame the patient is standing.
* **Scale**       — how far they are from the camera.

We strip both with a two-step transform applied per frame:

1. **Re-centre to hip midpoint.**  Subtract ``(left_hip + right_hip) / 2``
   from every joint. The hip midpoint is the most stable proxy for the
   centre of mass on a single-person stream and is always visible in a
   well-framed exercise pose.
2. **Rescale by spine length.**  Divide by the euclidean distance from
   the hip midpoint to the shoulder midpoint. This collapses the
   depth-dependent scale of the skeleton onto a unit "torso" so that a
   tall patient near the camera and a short one far away produce
   approximately the same coordinates for the same posture.

Both steps are applied in numpy so the function is cheap enough to run
inline on every 30 FPS frame. Degenerate cases (missing hips, zero spine
length) fall back to returning the input unchanged rather than blowing
up the live stream — the rest of the pipeline can still operate.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .mediapipe_graph import LANDMARK_NAMES

_LEFT_HIP_IDX = LANDMARK_NAMES.index("left_hip")
_RIGHT_HIP_IDX = LANDMARK_NAMES.index("right_hip")
_LEFT_SHOULDER_IDX = LANDMARK_NAMES.index("left_shoulder")
_RIGHT_SHOULDER_IDX = LANDMARK_NAMES.index("right_shoulder")

# Below this spine length the rescale would explode small numerical noise
# into very large coordinates; fall back to "no scale" instead.
_MIN_SPINE_LENGTH = 1e-3


def normalize_pose(frame_VC: np.ndarray) -> np.ndarray:
    """Return a (V, C) array re-centred at the hip and rescaled by spine.

    The input is the per-frame ``(V, C)`` stack the analyzer already
    builds (33 landmarks × 3 channels). Output is the same shape, with
    the hip midpoint sitting at the origin and the typical joint
    magnitude ≈ 1 (because the spine ≈ 1 by construction).
    """
    if frame_VC.ndim != 2 or frame_VC.shape[1] < 2:
        return frame_VC

    hip_mid, ok = _midpoint(frame_VC, _LEFT_HIP_IDX, _RIGHT_HIP_IDX)
    if not ok:
        return frame_VC

    centred = frame_VC - hip_mid

    shoulder_mid, sh_ok = _midpoint(centred, _LEFT_SHOULDER_IDX, _RIGHT_SHOULDER_IDX)
    if not sh_ok:
        return centred

    spine_length = float(np.linalg.norm(shoulder_mid))
    if spine_length < _MIN_SPINE_LENGTH:
        # Degenerate: shoulder ≈ hip (patient bent forward off-frame,
        # or one shoulder dropped to 0). Skip the rescale.
        return centred
    return centred / spine_length


def normalize_window(frames_TVC: np.ndarray) -> np.ndarray:
    """Apply :func:`normalize_pose` to each frame in a ``(T, V, C)`` stack."""
    if frames_TVC.ndim != 3:
        return frames_TVC
    out = np.empty_like(frames_TVC)
    for t in range(frames_TVC.shape[0]):
        out[t] = normalize_pose(frames_TVC[t])
    return out


def _midpoint(
    arr_VC: np.ndarray, idx_a: int, idx_b: int,
) -> Tuple[np.ndarray, bool]:
    """Midpoint of two landmarks, with a presence/validity flag.

    A landmark is "missing" when its (x, y) are exactly ``(0, 0)`` (the
    upstream code zero-fills absent points). When either side is missing
    we fall back to the present one; if both are missing we report
    failure so the caller can short-circuit.
    """
    a = arr_VC[idx_a]
    b = arr_VC[idx_b]
    a_present = not (a[0] == 0.0 and a[1] == 0.0)
    b_present = not (b[0] == 0.0 and b[1] == 0.0)
    if a_present and b_present:
        return 0.5 * (a + b), True
    if a_present:
        return a.copy(), True
    if b_present:
        return b.copy(), True
    return np.zeros_like(a), False


__all__ = ["normalize_pose", "normalize_window"]
