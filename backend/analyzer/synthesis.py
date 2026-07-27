"""Post-workout synthesis: turn the per-frame session log into the four
headline metrics + chart series the Summary Box renders.

The session log is a list of plain dicts, one per analysed frame:

    {
        "frame_index": int,
        "t_ms":        float,
        "similarity":  float | None,   # cosine-similarity score, if set
        "angles":      { "<joint>": <degrees>, ... },
    }

When the session ends we want four things (Phase 4 Task 8):

* **Overall Accuracy** — mean cosine-similarity if any references were
  scored, otherwise the analyzer's running quality score scaled to 0–100.
* **Rep count by angle peaks** — count completed cycles by looking for
  local minima in the most-active joint's angle series, using a small
  hysteresis to ignore noise jitter.
* **Per-rep peak ROM** — for each rep slice, ``max - min`` of the angle
  series. This is what a physio reports as "how deep did they go".
* **Fatigue indicator** — variance of the per-rep tremor (acceleration
  RMS). If late reps tremble more than early reps, fatigue is high.

Plus two chart series for the Summary Box:

* ``rom_curve`` — the chosen primary joint's angle over time, with the
  rep boundaries marked.
* ``stability_trend`` — one stability value (``1 − normalised tremor``)
  per detected rep.

All of this is computed in plain numpy so it's trivial to unit-test
without spinning up torch or the live socket.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .kinematics import range_of_motion, tremor_metrics


# Minimum drop (in degrees) below the running mean for a sample to be
# called a "trough". 8° is empirically wide enough to ignore camera
# jitter but narrow enough to catch shallow rehab reps (e.g., assisted
# knee extension).
_TROUGH_HYSTERESIS_DEG = 8.0


def pick_primary_joint(log: List[Dict[str, Any]]) -> Optional[str]:
    """Choose the joint with the largest range across the session.

    The "primary" joint is whichever moved most — that's the one whose
    rep cycle the user actually cares about (e.g., the knee for squats,
    the elbow for arm raises). Falls back to ``None`` for an empty log.
    """
    if not log:
        return None
    angles_by_joint: Dict[str, List[float]] = {}
    for entry in log:
        for joint, deg in (entry.get("angles") or {}).items():
            if deg is None or (isinstance(deg, float) and np.isnan(deg)):
                continue
            angles_by_joint.setdefault(joint, []).append(float(deg))
    if not angles_by_joint:
        return None
    best_joint = None
    best_range = -1.0
    for joint, values in angles_by_joint.items():
        if len(values) < 3:
            continue
        rng = max(values) - min(values)
        if rng > best_range:
            best_range = rng
            best_joint = joint
    return best_joint


def detect_rep_troughs(
    angle_series: np.ndarray, hysteresis_deg: float = _TROUGH_HYSTERESIS_DEG,
) -> List[int]:
    """Return indices of trough samples — one per completed rep cycle.

    Algorithm: walk the series tracking a running "armed" flag that flips
    on whenever the signal dips ``hysteresis_deg`` below the most recent
    peak, then commit a trough at the next sample where the signal
    starts rising again. This is the standard hysteresis-based peak
    detector adapted to inverted signals (rehab reps usually *decrease*
    a flexion angle then come back up).
    """
    if angle_series.size < 3:
        return []
    sig = angle_series.astype(np.float64)
    # Replace nans by linear interpolation so the pass is uninterrupted.
    if np.isnan(sig).any():
        idx = np.arange(sig.size)
        mask = np.isnan(sig)
        if mask.all():
            return []
        sig[mask] = np.interp(idx[mask], idx[~mask], sig[~mask])

    troughs: List[int] = []
    running_peak = sig[0]
    armed = False
    candidate_idx = -1
    candidate_val = float("inf")
    for i, v in enumerate(sig):
        if v > running_peak:
            running_peak = v
            # Reset the candidate search once we crest above a previous peak.
            if not armed:
                candidate_idx = -1
                candidate_val = float("inf")
        # Look for a sufficient drop to "arm" the trough detector.
        if not armed and (running_peak - v) >= hysteresis_deg:
            armed = True
            candidate_idx = i
            candidate_val = v
        elif armed:
            if v < candidate_val:
                candidate_idx = i
                candidate_val = v
            # Once the signal recovers far enough above the candidate, commit it.
            elif (v - candidate_val) >= hysteresis_deg:
                troughs.append(candidate_idx)
                armed = False
                running_peak = v
                candidate_idx = -1
                candidate_val = float("inf")
    return troughs


def _rep_slices(n_samples: int, troughs: List[int]) -> List[Tuple[int, int]]:
    """Split a series into [start, end) windows, one per detected rep."""
    if not troughs:
        return []
    bounds = [0] + troughs + [n_samples]
    slices: List[Tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a >= 3:  # ignore tiny tail slivers
            slices.append((a, b))
    return slices


def synthesize(
    log: List[Dict[str, Any]],
    *,
    quality_samples: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute the Summary Box payload from a session log.

    Output schema (stable — consumed by the React Summary Box)::

        {
          "overall_accuracy":     float | None,   # 0–100
          "rep_count_by_angle":   int,
          "per_rep_rom":          [ {rep, min_deg, max_deg, range_deg}, ... ],
          "fatigue_index":        float | None,   # variance of per-rep tremor
          "primary_joint":        str | None,
          "charts": {
            "rom_curve":        [ {t_ms, angle, is_trough}, ... ],
            "stability_trend": [ {rep, stability}, ... ],
          },
        }
    """
    out: Dict[str, Any] = {
        "overall_accuracy": None,
        "rep_count_by_angle": 0,
        "per_rep_rom": [],
        "fatigue_index": None,
        "primary_joint": None,
        "charts": {"rom_curve": [], "stability_trend": []},
    }
    if not log:
        return out

    # ---- Overall accuracy -------------------------------------------
    sims = [e["similarity"] for e in log if e.get("similarity") is not None]
    if sims:
        out["overall_accuracy"] = float(np.mean(sims))
    elif quality_samples:
        # Fall back to scaled mean quality so the Summary Box always has
        # *something* to show even before a reference movement is loaded.
        out["overall_accuracy"] = float(np.mean(quality_samples) * 100.0)

    # ---- Primary joint + angle series -------------------------------
    primary = pick_primary_joint(log)
    out["primary_joint"] = primary
    if primary is None:
        return out

    times = np.array([float(e.get("t_ms", 0.0)) for e in log], dtype=np.float64)
    angles = np.array(
        [float((e.get("angles") or {}).get(primary, np.nan)) for e in log],
        dtype=np.float64,
    )

    # ---- Rep detection + per-rep ROM --------------------------------
    troughs = detect_rep_troughs(angles)
    out["rep_count_by_angle"] = len(troughs)
    trough_set = set(troughs)
    slices = _rep_slices(angles.size, troughs)

    per_rep_rom: List[Dict[str, Any]] = []
    per_rep_tremor: List[float] = []
    stability_trend: List[Dict[str, Any]] = []
    for r, (a, b) in enumerate(slices, start=1):
        segment = angles[a:b]
        rom = range_of_motion(segment)
        per_rep_rom.append({
            "rep": r,
            "min_deg": rom["min_deg"],
            "max_deg": rom["max_deg"],
            "range_deg": rom["range_deg"],
        })
        tremor = tremor_metrics(segment)
        accel = tremor.get("acceleration_rms", 0.0)
        per_rep_tremor.append(accel)
        # Stability ∈ [0, 1]; clamps to 0 if accel is wildly large.
        stability = max(0.0, 1.0 - min(1.0, accel / 5.0))
        stability_trend.append({"rep": r, "stability": round(stability, 4)})

    out["per_rep_rom"] = per_rep_rom
    out["charts"]["stability_trend"] = stability_trend

    # ---- Fatigue indicator ------------------------------------------
    if len(per_rep_tremor) >= 2:
        out["fatigue_index"] = float(np.var(per_rep_tremor))

    # ---- ROM curve (subsampled if very long) ------------------------
    # The UI canvas can comfortably render a few hundred points; if the
    # session is longer than that we stride-down the series rather than
    # ship every frame across the wire. Ceil division guarantees the
    # output never exceeds ``max_points``.
    max_points = 400
    stride = max(1, -(-angles.size // max_points))
    curve: List[Dict[str, Any]] = []
    for i in range(0, angles.size, stride):
        v = angles[i]
        if np.isnan(v):
            continue
        curve.append({
            "t_ms": float(times[i]),
            "angle": float(v),
            "is_trough": bool(i in trough_set),
        })
    out["charts"]["rom_curve"] = curve

    return out


__all__ = ["synthesize", "pick_primary_joint", "detect_rep_troughs"]
