"""Pre-session body centering assistant.

Faithful port of the desktop reference at ``Final/centering_logic.py`` ::
``evaluate_centering``. The iOS app re-implements this again in Swift
(``ios/DPMonitor/Core/CenteringEvaluator.swift``); this module is the
shared source of truth that both are pinned against by the fixtures in
``scripts/gen_reference_fixtures.py``.

Why framing deserves its own check
----------------------------------
Every downstream number degrades silently when framing is bad:

* A patient at the edge of frame has landmarks clamped at the image
  boundary, which reads to the GCN as a held pose.
* Clipped feet mean knee and ankle joints are extrapolated, so the knee
  ROM a physio reads is fiction.
* Standing too far away shrinks the skeleton until landmark noise is a
  significant fraction of the spine length — and normalisation then
  amplifies exactly that noise.

None of these raise an error. They produce confident, wrong numbers.

Coordinate space and handedness
-------------------------------
Inputs are MediaPipe **screen-space** normalised landmarks: ``x``/``y`` in
``[0, 1]``, ``y`` growing downward. This is Branch A in the iOS pipeline —
deliberately NOT the world landmarks, which are hip-centred and so carry
no information about where in the frame the patient stands.

All direction words are **frame-relative**, exactly as in the reference:
"Patient too far LEFT" means the patient's body is toward x = 0, not that
they should move left. This is mirroring-agnostic, which is why the
evaluator takes no `mirrored` flag: the on-screen arrow the overlay draws
sits next to the patient's own on-screen image, so it reads correctly
whether or not the preview is mirrored.

Deviations from the reference, and why
--------------------------------------
1. Input is the ``{name: [x, y, z, visibility]}`` dict the rest of this
   package already passes around, rather than an indexed MediaPipe list.
   Landmark selection is identical.
2. A ``not_detected`` result is returned when a required landmark is
   absent. The reference indexes the list directly and would raise; its
   caller (``workers.py``) handles the no-pose case one level up. Folding
   that into the evaluator keeps the iOS call site from needing a second
   code path — the strings are copied from ``workers.py`` verbatim.
3. A ``status_code`` field is added alongside the reference's human
   ``status`` string, so the Swift enum and the UI can switch on a stable
   token instead of parsing English.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Thresholds (verbatim from the reference) -------------------------
HIP_X_MIN = 0.30
HIP_X_MAX = 0.70
SHOULDER_X_MIN = 0.25
SHOULDER_X_MAX = 0.75
HEAD_CLIP_Y = 0.03
#: The reference gates the head check on nose visibility < 0.3. Note this
#: is NOT the 0.5 occlusion threshold used elsewhere in the pipeline —
#: head framing is a coarser judgement than joint carry-forward.
NOSE_MIN_VISIBILITY = 0.3
KNEE_CLIP_Y = 0.97
TORSO_RATIO_MIN = 0.12
TORSO_RATIO_MAX = 0.70

# --- Colours (BGR, verbatim from the reference) -----------------------
COLOR_CENTERED = (0, 220, 0)
COLOR_POSITION = (0, 165, 255)
COLOR_ADVISORY = (0, 200, 255)
COLOR_CLIPPED = (0, 0, 255)
COLOR_NOT_DETECTED = (50, 50, 200)

# --- Status codes -----------------------------------------------------
# Plain strings so the JSON fixture, the Python and the Swift enum all
# agree on one spelling with no mapping layer in between.
STATUS_CENTERED = "centered"
STATUS_MOVE_RIGHT = "move_right"     # patient is too far LEFT in frame
STATUS_MOVE_LEFT = "move_left"       # patient is too far RIGHT in frame
STATUS_TOO_CLOSE = "too_close"
STATUS_TOO_FAR = "too_far"
STATUS_HEAD_CLIPPED = "head_clipped"
STATUS_FEET_CLIPPED = "feet_clipped"
STATUS_NOT_DETECTED = "not_detected"

# Severity drives the overlay colour on iOS, where a BGR tuple would be
# meaningless. Derived from the reference's colour choices.
SEVERITY_OK = "ok"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

_SEVERITY_BY_COLOR = {
    COLOR_CENTERED: SEVERITY_OK,
    COLOR_POSITION: SEVERITY_WARNING,
    COLOR_ADVISORY: SEVERITY_WARNING,
    COLOR_CLIPPED: SEVERITY_CRITICAL,
    COLOR_NOT_DETECTED: SEVERITY_CRITICAL,
}

_REQUIRED = (
    "nose",
    "left_hip", "right_hip",
    "left_shoulder", "right_shoulder",
    "left_knee", "right_knee",
)


@dataclass
class CenteringResult:
    """Outcome of one centering evaluation."""

    #: Human-readable primary message (the reference's ``status``).
    status: str = "Patient has left the field of view"
    #: Stable token for switching on. See ``STATUS_*``.
    status_code: str = STATUS_NOT_DETECTED
    is_centered: bool = False
    color: Tuple[int, int, int] = COLOR_NOT_DETECTED
    severity: str = SEVERITY_CRITICAL
    details: List[str] = field(default_factory=lambda: ["Pose not recognised."])
    hip_center_x: Optional[float] = None
    shoulder_center_x: Optional[float] = None
    torso_height_ratio: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "status_code": self.status_code,
            "is_centered": self.is_centered,
            "color": list(self.color),
            "severity": self.severity,
            "details": list(self.details),
            "hip_center_x": self.hip_center_x,
            "shoulder_center_x": self.shoulder_center_x,
            "torso_height_ratio": self.torso_height_ratio,
        }


def _pct(value: float) -> str:
    """Reproduce Python's ``f"{value:.0%}"`` for the detail lines."""
    return f"{value:.0%}"


def evaluate_centering(
    points: Dict[str, Sequence[float]],
) -> CenteringResult:
    """Evaluate framing and return guidance.

    Parameters
    ----------
    points
        MediaPipe screen-space landmarks, ``{name: [x, y, z, visibility]}``
        with x/y normalised to ``[0, 1]``.
    """
    if not points or any(
        points.get(name) is None or len(points[name]) < 2 for name in _REQUIRED
    ):
        # Matches workers.py's no-pose branch.
        return CenteringResult()

    def x(name: str) -> float:
        return float(points[name][0])

    def y(name: str) -> float:
        return float(points[name][1])

    def vis(name: str) -> float:
        p = points[name]
        return float(p[3]) if len(p) > 3 else 1.0

    issues: List[Tuple[str, Tuple[int, int, int], str]] = []

    # --- Horizontal centering (hip midpoint) --------------------------
    hip_cx = (x("left_hip") + x("right_hip")) / 2.0
    if hip_cx < HIP_X_MIN:
        issues.append(("Patient too far LEFT", COLOR_POSITION, STATUS_MOVE_RIGHT))
    elif hip_cx > HIP_X_MAX:
        issues.append(("Patient too far RIGHT", COLOR_POSITION, STATUS_MOVE_LEFT))

    # --- Shoulder centering -------------------------------------------
    shoulder_cx = (x("left_shoulder") + x("right_shoulder")) / 2.0
    if shoulder_cx < SHOULDER_X_MIN:
        issues.append(("Shoulders shifted LEFT", COLOR_ADVISORY, STATUS_MOVE_RIGHT))
    elif shoulder_cx > SHOULDER_X_MAX:
        issues.append(("Shoulders shifted RIGHT", COLOR_ADVISORY, STATUS_MOVE_LEFT))

    # --- Head clipping -------------------------------------------------
    if y("nose") < HEAD_CLIP_Y or vis("nose") < NOSE_MIN_VISIBILITY:
        issues.append(
            ("Patient HEAD may be cut off", COLOR_CLIPPED, STATUS_HEAD_CLIPPED))

    # --- Feet / knees clipping -----------------------------------------
    if max(y("left_knee"), y("right_knee")) > KNEE_CLIP_Y:
        issues.append(
            ("Patient FEET may be cut off", COLOR_CLIPPED, STATUS_FEET_CLIPPED))

    # --- Too close / too far -------------------------------------------
    # NOTE: the reference measures nose-to-hip, not shoulder-to-hip. Using
    # the shoulder would roughly halve the ratio and silently invalidate
    # every threshold below.
    torso_height = abs((y("left_hip") + y("right_hip")) / 2.0 - y("nose"))
    if torso_height < TORSO_RATIO_MIN:
        issues.append(
            ("Patient is TOO FAR from camera", COLOR_ADVISORY, STATUS_TOO_FAR))
    elif torso_height > TORSO_RATIO_MAX:
        issues.append(
            ("Patient is TOO CLOSE to camera", COLOR_ADVISORY, STATUS_TOO_CLOSE))

    hip_center_x = round(hip_cx, 4)
    shoulder_center_x = round(shoulder_cx, 4)
    torso_height_ratio = round(torso_height, 4)

    if not issues:
        return CenteringResult(
            status="Patient is CENTERED",
            status_code=STATUS_CENTERED,
            is_centered=True,
            color=COLOR_CENTERED,
            severity=SEVERITY_OK,
            details=[
                f"Hip center: {_pct(hip_cx)} (ideal ~50%)",
                f"Shoulder center: {_pct(shoulder_cx)}",
            ],
            hip_center_x=hip_center_x,
            shoulder_center_x=shoulder_center_x,
            torso_height_ratio=torso_height_ratio,
        )

    primary_msg, primary_color, primary_code = issues[0]
    detail_lines = [msg for msg, _, _ in issues]
    detail_lines.append(f"Hip center: {_pct(hip_cx)}")
    return CenteringResult(
        status=primary_msg,
        status_code=primary_code,
        is_centered=False,
        color=primary_color,
        severity=_SEVERITY_BY_COLOR.get(primary_color, SEVERITY_WARNING),
        details=detail_lines,
        hip_center_x=hip_center_x,
        shoulder_center_x=shoulder_center_x,
        torso_height_ratio=torso_height_ratio,
    )


__all__ = [
    "evaluate_centering",
    "CenteringResult",
    "HIP_X_MIN", "HIP_X_MAX",
    "SHOULDER_X_MIN", "SHOULDER_X_MAX",
    "HEAD_CLIP_Y", "NOSE_MIN_VISIBILITY",
    "KNEE_CLIP_Y",
    "TORSO_RATIO_MIN", "TORSO_RATIO_MAX",
    "STATUS_CENTERED", "STATUS_MOVE_LEFT", "STATUS_MOVE_RIGHT",
    "STATUS_TOO_CLOSE", "STATUS_TOO_FAR", "STATUS_HEAD_CLIPPED",
    "STATUS_FEET_CLIPPED", "STATUS_NOT_DETECTED",
    "SEVERITY_OK", "SEVERITY_WARNING", "SEVERITY_CRITICAL",
]
