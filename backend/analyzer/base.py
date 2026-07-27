"""Abstract analyzer interface.

The UI never imports a concrete analyzer; it asks the registry for one
by name. Any future XAI model just has to inherit `BaseAnalyzer` and
return the same dataclass shapes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Wire-level data contracts (mirrored on the frontend).
# ---------------------------------------------------------------------------


@dataclass
class AnalyzerFrame:
    """One frame's worth of normalised keypoints + timing info.

    `points` follows MediaPipe Pose's 33-landmark layout, with each value
    a 4-tuple `(x, y, z, visibility)` and x/y/z normalised into [0,1].
    """

    frame_index: int
    timestamp_ms: float
    points: Dict[str, List[float]]
    # Optional pre-computed joint angles from the client (cheap to do in JS).
    angles: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    """Per-frame output streamed to the UI."""

    frame_index: int
    count: int
    quality_score: float  # 0.0–1.0
    is_compensatory: bool
    feedback: List[str] = field(default_factory=list)
    # Optional: per-frame diagnostic payload (consumed only by Report view).
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "count": self.count,
            "quality_score": round(self.quality_score, 4),
            "is_compensatory": self.is_compensatory,
            "feedback": self.feedback,
            "diagnostics": self.diagnostics,
        }


@dataclass
class AnalyzerSummary:
    """Whole-session output produced at COMPLETED state."""

    rep_count: int
    overall_stability_score: float  # 0.0–1.0
    quality_score: float  # 0.0–1.0
    compensation_events: int
    progress_trend: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "rep_count": self.rep_count,
            "overall_stability_score": round(self.overall_stability_score, 4),
            "quality_score": round(self.quality_score, 4),
            "compensation_events": self.compensation_events,
            "progress_trend": self.progress_trend,
            "random_seed": self.random_seed,
        }


# ---------------------------------------------------------------------------
# Abstract base.
# ---------------------------------------------------------------------------


class BaseAnalyzer(ABC):
    """Strict, minimal interface every counting/quality model implements."""

    #: Human-friendly name used in logs/UI.
    name: str = "base"

    #: Set by subclasses if they accept an exercise_type hint.
    supported_exercises: tuple[str, ...] = ()

    def __init__(self, *, seed: int, exercise_type: str = "custom") -> None:
        self.seed = seed
        self.exercise_type = exercise_type
        # Concrete classes are expected to seed their own RNGs in `reset()`.
        self.reset()

    # ---- Lifecycle ------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of a session."""

    # ---- Core inference -------------------------------------------------

    @abstractmethod
    def analyze_frame(self, frame: AnalyzerFrame) -> AnalyzerResult:
        """Process one frame; return per-frame result.

        Must be cheap enough to run inline on every 30-FPS frame.
        """

    @abstractmethod
    def generate_summary(self) -> AnalyzerSummary:
        """Aggregate everything seen so far into a session summary."""

    # ---- Helpers --------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Static metadata for logs / report headers."""
        return {
            "analyzer": self.name,
            "exercise_type": self.exercise_type,
            "random_seed": self.seed,
            "supported_exercises": list(self.supported_exercises),
        }
