"""Kinematic metrics: Range-of-Motion and tremor (velocity/acceleration).

The CTR-GCN model gives us a *qualitative* signal — "does this look like
good form?" — but clinicians and physios want *quantitative* numbers
that translate directly to recovery milestones. This module is the
geometric counterpart to the learned model, computed from the same
sliding-window buffer of normalised keypoints:

* :func:`joint_angle_series` — per-frame angle (in degrees) of the
  triplet ``(a, vertex, b)``; e.g. the knee angle from hip → knee → ankle.
* :func:`range_of_motion` — peak-to-trough swing of an angle series.
  This is the canonical clinical ROM number.
* :func:`tremor_metrics` — RMS of velocity and acceleration of a 1D
  signal. Smooth motion → ~0; jitter → large numbers. Cheap, robust to
  scale, and matches what the rehab literature reports as "movement
  smoothness".

All functions accept numpy arrays so they're trivially composable with
the analyzer's existing buffer.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# Triplets used to define each clinical joint angle. The vertex is the
# joint whose angle we report; the other two are the limb segments
# meeting at it.
JOINT_ANGLE_TRIPLETS: Dict[str, tuple[str, str, str]] = {
    "left_knee":   ("left_hip",      "left_knee",      "left_ankle"),
    "right_knee":  ("right_hip",     "right_knee",     "right_ankle"),
    "left_elbow":  ("left_shoulder", "left_elbow",     "left_wrist"),
    "right_elbow": ("right_shoulder","right_elbow",    "right_wrist"),
    "left_hip":    ("left_shoulder", "left_hip",       "left_knee"),
    "right_hip":   ("right_shoulder","right_hip",      "right_knee"),
}


def joint_angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle at ``vertex`` in degrees, between rays vertex→a and vertex→b.

    Returns ``nan`` if either segment has zero length (a "missing"
    landmark situation), letting callers ignore the frame without
    polluting downstream statistics with a fake 0.
    """
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    v2 = np.asarray(b, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def joint_angle_series(
    window_TVC: np.ndarray,
    landmark_names: tuple[str, ...],
    triplet: tuple[str, str, str],
) -> np.ndarray:
    """Compute the angle at ``triplet[1]`` for every frame in the window.

    Frames where any required landmark is missing produce ``nan`` so
    downstream consumers can call ``np.nanmax`` / ``np.nanmin`` safely.
    """
    if window_TVC.ndim != 3:
        raise ValueError(f"window_TVC must be (T, V, C), got {window_TVC.shape}")
    ia = landmark_names.index(triplet[0])
    iv = landmark_names.index(triplet[1])
    ib = landmark_names.index(triplet[2])
    T = window_TVC.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        out[t] = joint_angle(window_TVC[t, ia], window_TVC[t, iv], window_TVC[t, ib])
    return out


def range_of_motion(angle_series: np.ndarray) -> Dict[str, float]:
    """Min / max / range for an angle series, ignoring NaNs."""
    if angle_series.size == 0 or np.all(np.isnan(angle_series)):
        return {"min_deg": float("nan"), "max_deg": float("nan"), "range_deg": float("nan")}
    mn = float(np.nanmin(angle_series))
    mx = float(np.nanmax(angle_series))
    return {"min_deg": mn, "max_deg": mx, "range_deg": mx - mn}


def tremor_metrics(signal: np.ndarray) -> Dict[str, float]:
    """RMS of velocity and acceleration of a 1D signal.

    For a perfectly smooth sinusoid sampled densely the discrete first
    and second differences are O(1/T) and O(1/T²) respectively — close
    enough to 0 that the tests use a small absolute threshold rather
    than exact equality.
    """
    if signal.size < 3:
        return {"velocity_rms": 0.0, "acceleration_rms": 0.0}
    sig = signal.astype(np.float64)
    # Ignore NaN segments by linear-interpolating across them; this keeps
    # the derivatives well-defined when a frame is briefly missing.
    if np.isnan(sig).any():
        idx = np.arange(sig.size)
        mask = np.isnan(sig)
        if mask.all():
            return {"velocity_rms": 0.0, "acceleration_rms": 0.0}
        sig[mask] = np.interp(idx[mask], idx[~mask], sig[~mask])
    v = np.diff(sig)
    a = np.diff(v)
    return {
        "velocity_rms": float(np.sqrt(np.mean(v * v))),
        "acceleration_rms": float(np.sqrt(np.mean(a * a))),
    }


__all__ = [
    "JOINT_ANGLE_TRIPLETS",
    "joint_angle",
    "joint_angle_series",
    "range_of_motion",
    "tremor_metrics",
]
