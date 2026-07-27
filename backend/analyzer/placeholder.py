"""Placeholder counting/quality analyzer.

Deterministic for a given seed + identical input stream. Intentionally
simple: the point of the MVP is to prove the *interface* works end-to-
end; real XAI models drop in later behind the same shape.

Algorithm sketch (used for every exercise type at MVP — exercise-
specific tuning is a future hook):

1. Track the y-coordinate of the hip midpoint over time.
2. Smooth it with a short moving average to suppress jitter.
3. Treat a full down-then-up excursion past a minimum threshold as one
   rep, using hysteresis to avoid double-counting jitter near extrema.
4. Quality is a function of (a) trajectory smoothness (jerk of the
   smoothed signal) and (b) optional joint-angle bounds passed in by
   the client.

This implementation is fully deterministic on its own — it consumes
neither `random` nor `numpy.random` — so reproducibility comes for
free without globally seeding the process. Future analyzers that DO
sample randomness should call `analyzer.seed.apply_global_seed(self.seed)`
in their own `reset()`.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from .base import AnalyzerFrame, AnalyzerResult, AnalyzerSummary, BaseAnalyzer


# MediaPipe Pose landmark names we care about for hip-midpoint tracking.
_LEFT_HIP = "left_hip"
_RIGHT_HIP = "right_hip"


def _hip_y(frame: AnalyzerFrame) -> Optional[float]:
    p = frame.points
    if _LEFT_HIP in p and _RIGHT_HIP in p:
        try:
            return 0.5 * (p[_LEFT_HIP][1] + p[_RIGHT_HIP][1])
        except (IndexError, TypeError):
            return None
    return None


class PlaceholderAnalyzer(BaseAnalyzer):
    name = "placeholder-v1"
    supported_exercises = ("squat", "lunge", "shoulder_raise", "knee_extension", "custom")

    # Smoothing window (frames). At 30 FPS, 5 frames ≈ 165 ms.
    _SMOOTH_WINDOW = 5
    # Minimum vertical excursion (normalised) to count a rep.
    _MIN_EXCURSION = 0.06
    # Quality drops when smoothed jerk exceeds this.
    _JERK_FLAG = 0.04

    def reset(self) -> None:
        # Note: this analyzer has no stochastic component, so we don't
        # touch `random` / `numpy.random` here. Determinism is implicit.
        self._rep_count: int = 0
        self._hip_window: Deque[float] = deque(maxlen=self._SMOOTH_WINDOW)
        self._smoothed_history: List[float] = []
        self._last_extreme: Optional[float] = None
        self._direction: int = 0  # +1 = going down, -1 = going up, 0 = unknown
        self._compensation_events: int = 0
        self._frames_seen: int = 0
        self._quality_running: float = 1.0
        # Per-frame quality samples used by the summary's stability score.
        self._quality_samples: List[float] = []

    # ------------------------------------------------------------------

    def analyze_frame(self, frame: AnalyzerFrame) -> AnalyzerResult:
        self._frames_seen += 1
        hip_y = _hip_y(frame)
        feedback: List[str] = []

        if hip_y is None:
            # Cannot evaluate this frame — emit a low-information result.
            self._quality_samples.append(self._quality_running)
            return AnalyzerResult(
                frame_index=frame.frame_index,
                count=self._rep_count,
                quality_score=self._quality_running,
                is_compensatory=False,
                feedback=["Move into full-body view"],
            )

        # 1) Smooth.
        self._hip_window.append(hip_y)
        smoothed = float(np.mean(self._hip_window))
        self._smoothed_history.append(smoothed)

        # 2) Rep detection via local extrema with hysteresis.
        if self._last_extreme is None:
            self._last_extreme = smoothed
        else:
            delta = smoothed - self._last_extreme
            # NB: in image coords y grows downward, so "going down" in the
            # world means y is *increasing*.
            if self._direction >= 0 and delta > 0:
                self._direction = 1
                self._last_extreme = max(self._last_extreme, smoothed)
            elif self._direction == 1 and delta < -self._MIN_EXCURSION:
                # Just turned upward after a sufficient descent — half rep.
                self._direction = -1
                self._last_extreme = smoothed
            elif self._direction == -1 and delta < 0:
                self._last_extreme = min(self._last_extreme, smoothed)
            elif self._direction == -1 and delta > self._MIN_EXCURSION:
                # Returned to top — full rep complete.
                self._rep_count += 1
                self._direction = 1
                self._last_extreme = smoothed
                feedback.append("Rep counted")

        # 3) Quality: penalise jerk.
        is_compensatory = False
        if len(self._smoothed_history) >= 3:
            jerk = abs(
                self._smoothed_history[-1]
                - 2 * self._smoothed_history[-2]
                + self._smoothed_history[-3]
            )
            if jerk > self._JERK_FLAG:
                is_compensatory = True
                self._compensation_events += 1
                feedback.append("Slow down — keep movement smooth")
                self._quality_running = max(0.0, self._quality_running - 0.05)
            else:
                # Gentle recovery toward 1.0.
                self._quality_running = min(1.0, self._quality_running + 0.01)

        # 4) Joint-angle hints from the client (if any).
        for joint, angle in frame.angles.items():
            if "knee" in joint and angle < 70:
                feedback.append("Knees collapsing inward")
                is_compensatory = True
            elif "back" in joint and angle < 150:
                feedback.append("Keep your chest up")
                is_compensatory = True

        self._quality_samples.append(self._quality_running)

        return AnalyzerResult(
            frame_index=frame.frame_index,
            count=self._rep_count,
            quality_score=self._quality_running,
            is_compensatory=is_compensatory,
            feedback=feedback,
            diagnostics={
                "smoothed_hip_y": round(smoothed, 4),
                "direction": self._direction,
            },
        )

    # ------------------------------------------------------------------

    def generate_summary(self) -> AnalyzerSummary:
        if self._quality_samples:
            quality = float(np.mean(self._quality_samples))
        else:
            quality = 0.0

        if len(self._smoothed_history) >= 3:
            # Stability ≈ 1 - normalised variance of first differences.
            diffs = np.diff(self._smoothed_history)
            spread = float(np.std(diffs))
            # Tuned so that a perfectly smooth signal ≈ 1.0 and noisy ≈ 0.
            stability = max(0.0, 1.0 - min(1.0, spread * 50.0))
        else:
            stability = 0.0

        return AnalyzerSummary(
            rep_count=self._rep_count,
            overall_stability_score=stability,
            quality_score=quality,
            compensation_events=self._compensation_events,
            progress_trend={
                "frames_analyzed": self._frames_seen,
                "quality_curve_samples": len(self._quality_samples),
            },
            random_seed=self.seed,
        )
