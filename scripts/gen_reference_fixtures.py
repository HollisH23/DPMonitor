#!/usr/bin/env python3
"""Generate golden reference fixtures for the Swift parity tests.

The iOS app re-implements four pieces of numerics that already exist in
Python/JS:

* ``analyzer.normalization.normalize_pose``          -> PoseNormalizer.swift
* ``analyzer.normalization.apply_occlusion_carryforward`` -> OcclusionHandler.swift
* ``analyzer.kinematics.{joint_angle, range_of_motion, tremor_metrics}``
                                                     -> Kinematics.swift
* ``frontend/src/lib/poseSmoothing.js`` (EMA)        -> PoseSmoother.swift

Rather than hard-coding hand-computed expectations into the XCTest files
(which drift silently the moment the Python changes), this script runs the
*actual* Python reference implementations over deterministic pseudo-random
inputs and dumps inputs + outputs to JSON. The Swift tests load that JSON
and assert agreement to < 1e-4.

Run it whenever the Python numerics change::

    python scripts/gen_reference_fixtures.py

Output: ios/DPMonitorTests/Fixtures/reference_fixtures.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _REPO_ROOT / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyzer.kinematics import (  # noqa: E402
    JOINT_ANGLE_TRIPLETS,
    joint_angle,
    range_of_motion,
    tremor_metrics,
)
from analyzer.mediapipe_graph import LANDMARK_NAMES, NUM_NODE  # noqa: E402
from analyzer.normalization import (  # noqa: E402
    apply_occlusion_carryforward,
    normalize_pose,
)
from analyzer import centering as _centering  # noqa: E402
from analyzer.centering import evaluate_centering  # noqa: E402

SEED = 20260803
OUT_PATH = _REPO_ROOT / "ios" / "DPMonitorTests" / "Fixtures" / "reference_fixtures.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(x) -> Any:
    """JSON-safe float conversion; NaN becomes ``None``."""
    v = float(x)
    return None if v != v else v


def _mat(a: np.ndarray) -> List[List[Any]]:
    return [[_f(v) for v in row] for row in np.asarray(a)]


def _vec(a: np.ndarray) -> List[Any]:
    return [_f(v) for v in np.asarray(a).ravel()]


def _plausible_pose(rng: np.random.Generator) -> np.ndarray:
    """A (33, 3) pose with a realistic torso so normalisation is well-conditioned.

    Purely random coordinates give a near-degenerate spine roughly 1 time in
    20, which would exercise only the fallback branch. We plant anatomically
    sane hips and shoulders and jitter everything else.
    """
    pose = rng.uniform(0.2, 0.8, size=(NUM_NODE, 3)).astype(np.float32)
    idx = {n: i for i, n in enumerate(LANDMARK_NAMES)}
    pose[idx["left_hip"]] = [0.45, 0.60, 0.02]
    pose[idx["right_hip"]] = [0.55, 0.60, -0.02]
    pose[idx["left_shoulder"]] = [0.43, 0.32, 0.03]
    pose[idx["right_shoulder"]] = [0.57, 0.32, -0.03]
    # Small per-sample jitter so cases aren't identical.
    pose += rng.normal(0.0, 0.01, size=pose.shape).astype(np.float32)
    return pose.astype(np.float32)


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def build_normalize_cases(rng: np.random.Generator) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    for i in range(6):
        pose = _plausible_pose(rng)
        cases.append({
            "name": f"plausible_{i}",
            "note": "well-conditioned torso; exercises the normal path",
            "input": _mat(pose),
            "expected": _mat(normalize_pose(pose)),
        })

    # Degenerate 1: both hips zero-filled -> function returns input unchanged.
    pose = _plausible_pose(rng)
    pose[LANDMARK_NAMES.index("left_hip")] = [0.0, 0.0, 0.0]
    pose[LANDMARK_NAMES.index("right_hip")] = [0.0, 0.0, 0.0]
    cases.append({
        "name": "both_hips_missing",
        "note": "both hips at (0,0) -> _midpoint reports failure -> input returned as-is",
        "input": _mat(pose),
        "expected": _mat(normalize_pose(pose)),
    })

    # Degenerate 2: one hip missing -> midpoint falls back to the present hip.
    pose = _plausible_pose(rng)
    pose[LANDMARK_NAMES.index("right_hip")] = [0.0, 0.0, 0.0]
    cases.append({
        "name": "right_hip_missing",
        "note": "single-hip fallback: midpoint == left_hip",
        "input": _mat(pose),
        "expected": _mat(normalize_pose(pose)),
    })

    # Degenerate 3: shoulders collapse onto the hips -> spine < 1e-3 -> centre only.
    pose = _plausible_pose(rng)
    hip_mid = 0.5 * (
        pose[LANDMARK_NAMES.index("left_hip")] + pose[LANDMARK_NAMES.index("right_hip")]
    )
    pose[LANDMARK_NAMES.index("left_shoulder")] = hip_mid
    pose[LANDMARK_NAMES.index("right_shoulder")] = hip_mid
    cases.append({
        "name": "degenerate_spine",
        "note": "spine length < 1e-3 -> centre but do not rescale",
        "input": _mat(pose),
        "expected": _mat(normalize_pose(pose)),
    })

    # Degenerate 4: both shoulders zero-filled -> return the centred pose.
    pose = _plausible_pose(rng)
    pose[LANDMARK_NAMES.index("left_shoulder")] = [0.0, 0.0, 0.0]
    pose[LANDMARK_NAMES.index("right_shoulder")] = [0.0, 0.0, 0.0]
    cases.append({
        "name": "both_shoulders_missing",
        "note": "shoulder midpoint unavailable -> centred pose returned",
        "input": _mat(pose),
        "expected": _mat(normalize_pose(pose)),
    })

    return cases


def build_occlusion_cases(rng: np.random.Generator) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    threshold = 0.5

    for i in range(4):
        normed = normalize_pose(_plausible_pose(rng))
        prev = normalize_pose(_plausible_pose(rng))
        vis = rng.uniform(0.0, 1.0, size=NUM_NODE).astype(np.float32)
        out = apply_occlusion_carryforward(normed, vis, prev, threshold)
        cases.append({
            "name": f"mixed_visibility_{i}",
            "normed": _mat(normed),
            "visibility": _vec(vis),
            "prev": _mat(prev),
            "threshold": threshold,
            "expected": _mat(out),
        })

    # No previous frame -> pass-through (first frame of a session).
    normed = normalize_pose(_plausible_pose(rng))
    vis = np.zeros(NUM_NODE, dtype=np.float32)  # everything "occluded"
    cases.append({
        "name": "no_previous_frame",
        "note": "first frame: nothing to carry forward, input passes through",
        "normed": _mat(normed),
        "visibility": _vec(vis),
        "prev": None,
        "threshold": threshold,
        "expected": _mat(apply_occlusion_carryforward(normed, vis, None, threshold)),
    })

    # All joints visible -> untouched.
    normed = normalize_pose(_plausible_pose(rng))
    prev = normalize_pose(_plausible_pose(rng))
    vis = np.full(NUM_NODE, 0.99, dtype=np.float32)
    cases.append({
        "name": "all_visible",
        "note": "no joint below threshold -> identity",
        "normed": _mat(normed),
        "visibility": _vec(vis),
        "prev": _mat(prev),
        "threshold": threshold,
        "expected": _mat(apply_occlusion_carryforward(normed, vis, prev, threshold)),
    })

    return cases


def build_joint_angle_cases(rng: np.random.Generator) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    # Analytically known angles first — these catch sign/axis mistakes that
    # random data can mask.
    known = [
        ("right_angle_90", [1, 0, 0], [0, 0, 0], [0, 1, 0], 90.0),
        ("straight_180", [1, 0, 0], [0, 0, 0], [-1, 0, 0], 180.0),
        ("collapsed_0", [1, 0, 0], [0, 0, 0], [2, 0, 0], 0.0),
        ("diagonal_45", [1, 0, 0], [0, 0, 0], [1, 1, 0], 45.0),
    ]
    for name, a, v, b, analytic in known:
        A, V, B = map(lambda t: np.array(t, dtype=np.float64), (a, v, b))
        cases.append({
            "name": name,
            "a": _vec(A), "vertex": _vec(V), "b": _vec(B),
            "expected_deg": _f(joint_angle(A, V, B)),
            "analytic_deg": analytic,
        })

    # Zero-length segment -> NaN (encoded as null).
    cases.append({
        "name": "degenerate_zero_segment",
        "note": "vertex coincides with a -> NaN, must not become 0",
        "a": [0.0, 0.0, 0.0], "vertex": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0],
        "expected_deg": _f(joint_angle(
            np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]))),
        "analytic_deg": None,
    })

    for i in range(8):
        A, V, B = (rng.uniform(-1, 1, 3) for _ in range(3))
        cases.append({
            "name": f"random_{i}",
            "a": _vec(A), "vertex": _vec(V), "b": _vec(B),
            "expected_deg": _f(joint_angle(A, V, B)),
            "analytic_deg": None,
        })

    return cases


def _synthetic_angle_series(rng: np.random.Generator, n: int = 64) -> np.ndarray:
    """A rep-like flexion signal: sinusoid + noise, in degrees."""
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return (120.0 - 45.0 * (1.0 - np.cos(t)) + rng.normal(0.0, 1.2, n)).astype(np.float64)


def build_rom_cases(rng: np.random.Generator) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for i in range(4):
        s = _synthetic_angle_series(rng)
        cases.append({
            "name": f"rep_like_{i}",
            "series": _vec(s),
            "expected": {k: _f(v) for k, v in range_of_motion(s).items()},
        })

    # With NaN holes (dropped landmarks) — nan{min,max} must ignore them.
    s = _synthetic_angle_series(rng)
    s[[3, 4, 17, 40]] = np.nan
    cases.append({
        "name": "with_nan_holes",
        "note": "nanmin/nanmax must skip missing frames",
        "series": _vec(s),
        "expected": {k: _f(v) for k, v in range_of_motion(s).items()},
    })

    # All-NaN -> all NaN out.
    s = np.full(16, np.nan)
    cases.append({
        "name": "all_nan",
        "series": _vec(s),
        "expected": {k: _f(v) for k, v in range_of_motion(s).items()},
    })

    # Empty.
    cases.append({
        "name": "empty",
        "series": [],
        "expected": {k: _f(v) for k, v in range_of_motion(np.array([])).items()},
    })

    return cases


def build_tremor_cases(rng: np.random.Generator) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    # Smooth: derivatives near zero.
    t = np.linspace(0.0, 2.0 * np.pi, 128)
    smooth = 30.0 * np.sin(t)
    cases.append({
        "name": "smooth_sinusoid",
        "signal": _vec(smooth),
        "expected": {k: _f(v) for k, v in tremor_metrics(smooth).items()},
    })

    # Jittery.
    for i in range(3):
        s = _synthetic_angle_series(rng) + rng.normal(0.0, 4.0, 64)
        cases.append({
            "name": f"jittery_{i}",
            "signal": _vec(s),
            "expected": {k: _f(v) for k, v in tremor_metrics(s).items()},
        })

    # NaN interpolation branch.
    s = _synthetic_angle_series(rng)
    s[[5, 6, 7, 30]] = np.nan
    cases.append({
        "name": "with_nan_interpolated",
        "note": "NaNs are linearly interpolated before differencing",
        "signal": _vec(s),
        "expected": {k: _f(v) for k, v in tremor_metrics(s).items()},
    })

    # Too short -> zeros.
    cases.append({
        "name": "too_short",
        "signal": [10.0, 11.0],
        "expected": {k: _f(v) for k, v in tremor_metrics(np.array([10.0, 11.0])).items()},
    })

    return cases


# --- EMA smoother -----------------------------------------------------------
# Port of frontend/src/lib/poseSmoothing.js, kept here as the single source of
# truth for the golden values so Swift and JS provably agree.

def _ema_reference(
    frames: List[np.ndarray],
    alpha: float = 0.6,
    occlusion_threshold: float = 0.5,
    occluded_alpha: float = 0.15,
) -> List[np.ndarray]:
    """Each frame is a (V, 4) array of (x, y, z, visibility)."""
    prev: Optional[np.ndarray] = None
    outputs: List[np.ndarray] = []
    for raw in frames:
        if prev is None:
            prev = raw.copy()
            outputs.append(prev.copy())
            continue
        out = np.empty_like(raw)
        for v in range(raw.shape[0]):
            a = occluded_alpha if raw[v, 3] < occlusion_threshold else alpha
            out[v] = a * raw[v] + (1.0 - a) * prev[v]
        prev = out
        outputs.append(out.copy())
    return outputs


def build_ema_cases(rng: np.random.Generator) -> List[Dict[str, Any]]:
    n_frames = 10
    frames: List[np.ndarray] = []
    for _ in range(n_frames):
        xyz = rng.uniform(-1.0, 1.0, size=(NUM_NODE, 3))
        # Bias visibility so both alpha branches are exercised every frame.
        vis = rng.uniform(0.0, 1.0, size=(NUM_NODE, 1))
        frames.append(np.hstack([xyz, vis]).astype(np.float64))

    outputs = _ema_reference(frames)
    return [{
        "name": "ema_sequence",
        "alpha": 0.6,
        "occlusion_threshold": 0.5,
        "occluded_alpha": 0.15,
        "note": "visibility is smoothed with the same alpha as the coordinates",
        "frames": [_mat(f) for f in frames],
        "expected": [_mat(o) for o in outputs],
    }]


def _screen_pose(
    *,
    hip_x: float = 0.50,
    shoulder_x: float = 0.50,
    hip_y: float = 0.60,
    shoulder_y: float = 0.32,
    nose_y: float = 0.10,
    knee_y: float = 0.80,
    ankle_y: float = 0.93,
    nose_vis: float = 0.9,
    vis: float = 0.9,
    drop: tuple = (),
) -> Dict[str, List[float]]:
    """Build a screen-space landmark dict with controllable framing.

    Defaults describe a well-framed standing patient: torso ratio 0.50
    (nose-to-hip, per the reference), hips and shoulders on the centre
    line, head and feet inside frame. Each keyword nudges exactly one
    property so a case can isolate a single check.

    ``hip_x`` values are chosen to give clean whole percentages, because
    the reference's detail lines format them with ``:.0%`` and a value
    landing exactly on a .5 boundary would round differently in Swift.
    """
    half_w = 0.09  # half the shoulder/hip width on screen
    pose: Dict[str, List[float]] = {}

    def put(name: str, x: float, y: float, v: float = None) -> None:
        if name in drop:
            return
        pose[name] = [float(x), float(y), 0.0, float(vis if v is None else v)]

    put("nose", hip_x, nose_y, nose_vis)
    put("left_shoulder", shoulder_x - half_w, shoulder_y)
    put("right_shoulder", shoulder_x + half_w, shoulder_y)
    put("left_hip", hip_x - half_w, hip_y)
    put("right_hip", hip_x + half_w, hip_y)
    put("left_knee", hip_x - half_w, knee_y)
    put("right_knee", hip_x + half_w, knee_y)
    put("left_ankle", hip_x - half_w, ankle_y)
    put("right_ankle", hip_x + half_w, ankle_y)
    return pose


def build_centering_cases() -> List[Dict[str, Any]]:
    """One case per branch of ``evaluate_centering``, plus the boundaries.

    Every threshold comparison is strict on one side, so a value sitting
    exactly on a boundary is the cheapest way to catch a ``<`` that should
    have been ``<=`` in the Swift port. Those cases are deliberate.

    Note the torso ratio is measured **nose to hip**, not shoulder to hip
    — using the shoulder would roughly halve it and quietly invalidate the
    0.12/0.70 thresholds.
    """
    cases: List[Dict[str, Any]] = []

    def add(name: str, pose: Dict[str, List[float]], note: str) -> None:
        cases.append({
            "name": name,
            "note": note,
            "points": pose,
            "expected": evaluate_centering(pose).to_json(),
        })

    # --- Happy path ---------------------------------------------------
    add("centered", _screen_pose(), "all checks pass")
    add("centered_off_axis_but_inside", _screen_pose(hip_x=0.34, shoulder_x=0.30),
        "inside both safe zones, still centered")

    # --- Horizontal position (hips) -----------------------------------
    add("hip_too_far_left", _screen_pose(hip_x=0.18, shoulder_x=0.18),
        "hip x below 0.30")
    add("hip_too_far_right", _screen_pose(hip_x=0.85, shoulder_x=0.85),
        "hip x above 0.70")

    # --- Boundaries ---------------------------------------------------
    # Straddled, never sat on exactly.
    #
    # Python computes in float64; Swift stores landmarks as Float. For a
    # value like 0.30 the two representations differ by ~1e-8, so at the
    # exact threshold the two sides can legitimately land on opposite
    # branches — (0.21+0.39)/2 is 0.30000000000000004 in float64 but
    # 0.29999999 in float32. Asserting agreement there tests floating
    # point, not the algorithm. A ±0.002 margin is ~4 orders of magnitude
    # above float32 epsilon at this scale, so it pins the inclusive/
    # exclusive semantics without depending on sub-epsilon behaviour.
    # Exact-threshold behaviour is pinned separately, in Python only,
    # by `CenteringTests` where float64 exactness is meaningful.
    add("hip_just_inside_min", _screen_pose(hip_x=0.302, shoulder_x=0.302),
        "just inside 0.30 — must NOT flag")
    add("hip_just_outside_min", _screen_pose(hip_x=0.298, shoulder_x=0.298),
        "just outside 0.30 — must flag")
    add("hip_just_inside_max", _screen_pose(hip_x=0.698, shoulder_x=0.698),
        "just inside 0.70 — must NOT flag")
    add("hip_just_outside_max", _screen_pose(hip_x=0.702, shoulder_x=0.702),
        "just outside 0.70 — must flag")
    add("shoulder_just_inside_min", _screen_pose(hip_x=0.40, shoulder_x=0.252),
        "just inside 0.25 — must NOT flag")
    add("shoulder_just_outside_min", _screen_pose(hip_x=0.40, shoulder_x=0.248),
        "just outside 0.25 — must flag")
    add("shoulder_just_inside_max", _screen_pose(hip_x=0.60, shoulder_x=0.748),
        "just inside 0.75 — must NOT flag")
    add("shoulder_just_outside_max", _screen_pose(hip_x=0.60, shoulder_x=0.752),
        "just outside 0.75 — must flag")

    # --- Shoulder alignment -------------------------------------------
    add("shoulders_shifted_left", _screen_pose(hip_x=0.45, shoulder_x=0.20),
        "shoulders outside band while hips are inside")
    add("shoulders_shifted_right", _screen_pose(hip_x=0.55, shoulder_x=0.80),
        "shoulders outside band while hips are inside")

    # --- Head clipping ------------------------------------------------
    add("head_clipped_high", _screen_pose(nose_y=0.01), "nose y under 0.03")
    add("head_just_inside_threshold", _screen_pose(nose_y=0.032),
        "just inside 0.03 — must NOT flag")
    add("head_just_outside_threshold", _screen_pose(nose_y=0.028),
        "just outside 0.03 — must flag")
    add("head_low_visibility", _screen_pose(nose_vis=0.2),
        "nose visibility under 0.3 counts as clipped")
    add("head_visibility_just_inside", _screen_pose(nose_vis=0.32),
        "just inside the 0.3 gate — must NOT flag")
    add("head_visibility_just_below", _screen_pose(nose_vis=0.28),
        "just under the visibility gate — must flag")

    # --- Feet clipping ------------------------------------------------
    add("feet_clipped", _screen_pose(knee_y=0.99), "knee y past 0.97")
    add("feet_just_inside_threshold", _screen_pose(knee_y=0.968),
        "just inside 0.97 — must NOT flag")
    add("feet_just_outside_threshold", _screen_pose(knee_y=0.972),
        "just outside 0.97 — must flag")
    add("one_knee_clipped", _screen_pose(knee_y=0.80),
        "max() of the two knees drives the check")

    # --- Distance (nose-to-hip torso ratio) ---------------------------
    add("too_far", _screen_pose(nose_y=0.50, hip_y=0.58, knee_y=0.70),
        "torso ratio 0.08 < 0.12")
    add("too_close", _screen_pose(nose_y=0.05, hip_y=0.80, knee_y=0.95),
        "torso ratio 0.75 > 0.70")
    add("distance_just_inside_min", _screen_pose(nose_y=0.475, hip_y=0.60),
        "ratio 0.125 — just inside, must NOT flag")
    add("distance_just_outside_min", _screen_pose(nose_y=0.485, hip_y=0.60),
        "ratio 0.115 — just outside, must flag")
    add("distance_just_inside_max", _screen_pose(nose_y=0.055, hip_y=0.75,
                                                 knee_y=0.90),
        "ratio 0.695 — just inside, must NOT flag")
    add("distance_just_outside_max", _screen_pose(nose_y=0.045, hip_y=0.75,
                                                  knee_y=0.90),
        "ratio 0.705 — just outside, must flag")

    # --- Priority ordering (hips win the headline) --------------------
    add("off_centre_and_too_far",
        _screen_pose(hip_x=0.10, shoulder_x=0.10, nose_y=0.50, hip_y=0.58,
                     knee_y=0.70),
        "hip issue is listed first, so it becomes `status`")
    add("everything_wrong",
        _screen_pose(hip_x=0.90, shoulder_x=0.90, nose_y=0.00, hip_y=0.80,
                     knee_y=0.99),
        "every check fails; details must carry all of them in order")

    # --- Degenerate ----------------------------------------------------
    add("no_landmarks", {}, "empty dict -> not detected")
    add("hips_missing", _screen_pose(drop=("left_hip", "right_hip")),
        "required landmark absent -> not detected")
    add("nose_missing", _screen_pose(drop=("nose",)),
        "required landmark absent -> not detected")
    add("knees_missing", _screen_pose(drop=("left_knee", "right_knee")),
        "required landmark absent -> not detected")

    return cases


def build_centering_thresholds() -> Dict[str, Any]:
    """Ship the constants so the Swift port cannot drift from Python."""
    return {
        "hip_x_min": _centering.HIP_X_MIN,
        "hip_x_max": _centering.HIP_X_MAX,
        "shoulder_x_min": _centering.SHOULDER_X_MIN,
        "shoulder_x_max": _centering.SHOULDER_X_MAX,
        "head_clip_y": _centering.HEAD_CLIP_Y,
        "nose_min_visibility": _centering.NOSE_MIN_VISIBILITY,
        "knee_clip_y": _centering.KNEE_CLIP_Y,
        "torso_ratio_min": _centering.TORSO_RATIO_MIN,
        "torso_ratio_max": _centering.TORSO_RATIO_MAX,
    }


def build_tensor_layout_case() -> Dict[str, Any]:
    """Golden (1,3,64,33,1) flattening so FrameBuffer's index math is pinned.

    Mirrors ``CTRGCNAnalyzer._make_tensor``: numpy (T, V, C) -> permute to
    (C, T, V) -> unsqueeze to (1, C, T, V, 1), C-contiguous.
    """
    T, V, C = 64, NUM_NODE, 3
    # Deterministic ramp so any transposition bug shows up as an exact mismatch.
    flat_TVC = np.arange(T * V * C, dtype=np.float32).reshape(T, V, C)
    tensor = np.transpose(flat_TVC, (2, 0, 1)).reshape(1, C, T, V, 1)
    contiguous = np.ascontiguousarray(tensor).ravel()
    # Spot-check indices rather than shipping 6336 floats.
    probes = [0, 1, 33, 34, 2111, 2112, 2113, 4223, 4224, 6335]
    return {
        "shape": [1, C, T, V, 1],
        "element_count": int(T * V * C),
        "strides": [T * V * C, T * V, V, 1, 1],
        "source_layout": "(T, V, C) row-major",
        "note": "dstIdx = c*T*V + t*V + v ; srcIdx = t*(V*C) + v*C + c",
        "source_flat_TVC": _vec(flat_TVC),
        "probe_indices": probes,
        "probe_values": [_f(contiguous[i]) for i in probes],
    }


# ---------------------------------------------------------------------------

def main() -> int:
    rng = np.random.default_rng(SEED)

    fixtures: Dict[str, Any] = {
        "_generated_by": "scripts/gen_reference_fixtures.py",
        "_seed": SEED,
        "_tolerance": 1e-4,
        "landmark_names": list(LANDMARK_NAMES),
        "num_node": NUM_NODE,
        "joint_angle_triplets": {k: list(v) for k, v in JOINT_ANGLE_TRIPLETS.items()},
        "normalize": build_normalize_cases(rng),
        "occlusion": build_occlusion_cases(rng),
        "joint_angle": build_joint_angle_cases(rng),
        "range_of_motion": build_rom_cases(rng),
        "tremor": build_tremor_cases(rng),
        "ema": build_ema_cases(rng),
        "centering": build_centering_cases(),
        "centering_thresholds": build_centering_thresholds(),
        "tensor_layout": build_tensor_layout_case(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixtures, indent=1) + "\n")

    size_kb = OUT_PATH.stat().st_size / 1024.0
    print(f"wrote {OUT_PATH}  ({size_kb:.1f} KB)")
    for key in ("normalize", "occlusion", "joint_angle", "range_of_motion",
                "tremor", "ema", "centering"):
        print(f"  {key:18s} {len(fixtures[key])} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
