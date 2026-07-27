"""CTR-GCN-backed analyzer (replaces the placeholder for live tracking).

Bridges the per-frame MediaPipe stream (33 normalised landmarks) to the
pre-trained CTR-GCN action recognition model living under
``ctrgcn/ctrgcn.py``. The bridge does six things, in order:

1. **Stack** the incoming MediaPipe ``points`` dict into a ``(V=33, C=3)``
   numpy array.
2. **Normalise** the pose (hip midpoint → origin, divide by spine
   length) so camera distance and frame position can't change the
   model's input distribution.
3. **Buffer** the most recent ``window_size`` normalised frames in a
   deque. Until the buffer is full the analyzer emits a neutral result
   so the UI keeps streaming during the warm-up.
4. **Stride trigger**: once the window is full, run a CTR-GCN forward
   pass every ``inference_stride`` frames (not every single frame).
   This is the difference between "responsive" and "GPU on fire" on
   modest hardware.
5. **Reshape & score**: turn ``(T, V, 3)`` into the strict
   ``(N=1, C=3, T, V, M=1)`` tensor the model expects, softmax the
   logits, and translate them into ``quality_score`` (0–1) and
   ``is_compensatory`` plus an optional cosine-similarity score
   against a stored reference movement.
6. **Kinematic side-band**: compute ROM and tremor metrics on the
   current window from the same buffer, surfaced via the per-frame
   ``diagnostics`` payload so the report view can graph them later.

Rep counting is intentionally kept as a deterministic geometric heuristic
(hip-midpoint peak detection) — CTR-GCN is an action classifier, not a
counter, so coupling rep counting to its raw logits would conflate two
concerns. The classifier drives quality; the heuristic drives count.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .base import AnalyzerFrame, AnalyzerResult, AnalyzerSummary, BaseAnalyzer
from .kinematics import (
    JOINT_ANGLE_TRIPLETS,
    joint_angle,
    joint_angle_series,
    range_of_motion,
    tremor_metrics,
)
from .mediapipe_graph import LANDMARK_NAMES, NUM_NODE
from .normalization import normalize_pose
from .seed import apply_global_seed
from .similarity import similarity_score
from .synthesis import synthesize

logger = logging.getLogger("rehab")


# Default class layout if no fine-tuned head is supplied: the first
# logit indexes "good form", the second "compensatory form". With a
# real checkpoint this can be re-mapped via ``class_index_good_form``.
_DEFAULT_NUM_CLASS = 2
_DEFAULT_GOOD_FORM_INDEX = 0
_DEFAULT_BAD_FORM_INDEX = 1


class CTRGCNAnalyzer(BaseAnalyzer):
    """Plug-and-play CTR-GCN analyzer for the live monitoring pipeline."""

    name = "ctrgcn-v1"
    supported_exercises = (
        "squat", "lunge", "shoulder_raise", "knee_extension",
        "chest_expansion", "custom",
    )

    # ---- Construction --------------------------------------------------

    def __init__(
        self,
        *,
        seed: int,
        exercise_type: str = "custom",
        window_size: int = 64,
        inference_stride: int = 5,
        num_class: int = _DEFAULT_NUM_CLASS,
        weights_path: Optional[str] = None,
        reference_feature: Optional[np.ndarray] = None,
        device: str = "cpu",
        class_index_good_form: int = _DEFAULT_GOOD_FORM_INDEX,
        class_index_bad_form: int = _DEFAULT_BAD_FORM_INDEX,
        normalize: bool = True,
    ) -> None:
        # Late-bind torch + the model so test environments that opt out of
        # the heavy dep can still import the analyzer module for
        # introspection.
        import torch  # noqa: F401  (importable from the test surface)

        self.window_size = int(window_size)
        self.inference_stride = max(1, int(inference_stride))
        self.num_class = int(num_class)
        self.device = device
        self._good_idx = class_index_good_form
        self._bad_idx = class_index_bad_form
        self._weights_path = weights_path
        self._reference_feature = (
            np.asarray(reference_feature, dtype=np.float32)
            if reference_feature is not None
            else None
        )
        self._normalize = bool(normalize)
        super().__init__(seed=seed, exercise_type=exercise_type)

    # ---- Lifecycle -----------------------------------------------------

    def reset(self) -> None:
        # Seed BEFORE building the model so its randomly-initialised
        # weights are reproducible.
        apply_global_seed(self.seed)
        self._build_model()

        self._buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self._raw_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self._latest_logits: Optional[np.ndarray] = None
        self._latest_features: Optional[np.ndarray] = None
        self._latest_similarity: Optional[float] = None
        self._frames_since_inference: int = 0
        self._inference_calls: int = 0
        self._quality_running: float = 1.0
        self._quality_samples: List[float] = []
        self._compensation_events: int = 0

        # Heuristic rep counter (mirrors PlaceholderAnalyzer).
        self._rep_count: int = 0
        self._smoothed_history: List[float] = []
        self._hip_window: Deque[float] = deque(maxlen=5)
        self._last_extreme: Optional[float] = None
        self._direction: int = 0
        self._frames_seen: int = 0

        # Continuous session log (Phase 3 Task 7): the analyzer keeps a
        # cheap in-memory list of {frame_index, t_ms, similarity, angles}
        # per analysed frame, which feeds the post-workout synthesis at
        # ``generate_summary()`` time.
        self._session_log: List[Dict[str, Any]] = []

    def _build_model(self) -> None:
        # Imported here so the (heavy) ctrgcn import doesn't pay its cost
        # at module-load time.
        import torch
        from ctrgcn.ctrgcn import Model  # type: ignore

        self._torch = torch
        self._model = Model(
            num_class=self.num_class,
            num_point=NUM_NODE,
            num_person=1,
            graph="analyzer.mediapipe_graph.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=3,
            drop_out=0.0,
            adaptive=True,
        )
        if self._weights_path:
            path = Path(self._weights_path)
            if path.is_file():
                state = torch.load(str(path), map_location=self.device)
                # Accept either a raw state_dict or a wrapper {"model": ...}.
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                missing, unexpected = self._model.load_state_dict(state, strict=False)
                logger.info(
                    "ctrgcn weights loaded from %s (missing=%d unexpected=%d)",
                    path, len(missing), len(unexpected),
                )
            else:
                logger.warning(
                    "ctrgcn weights_path %s not found; using freshly initialised weights.",
                    path,
                )
        self._model.to(self.device)
        self._model.eval()

    # ---- Public helpers (testable without torch) -----------------------

    def buffer_tensor_shape(self) -> Tuple[int, int, int, int, int]:
        """Return the shape that the next forward pass would receive."""
        return (1, 3, self.window_size, NUM_NODE, 1)

    def set_reference_feature(self, feature: Optional[np.ndarray]) -> None:
        """Install / clear the cosine-similarity reference vector at runtime."""
        self._reference_feature = (
            np.asarray(feature, dtype=np.float32) if feature is not None else None
        )

    def _stack_frame(self, frame: AnalyzerFrame) -> np.ndarray:
        """Flatten one frame's points dict to a (33, 3) np.float32 array."""
        out = np.zeros((NUM_NODE, 3), dtype=np.float32)
        for i, name in enumerate(LANDMARK_NAMES):
            p = frame.points.get(name)
            if not p:
                continue
            # ``points`` is ``[x, y, z, visibility]`` in MediaPipe order.
            out[i, 0] = float(p[0]) if len(p) > 0 else 0.0
            out[i, 1] = float(p[1]) if len(p) > 1 else 0.0
            out[i, 2] = float(p[2]) if len(p) > 2 else 0.0
        return out

    def _make_tensor(self, frames_TVC: np.ndarray):
        """``(T, V, 3)`` numpy → ``(1, 3, T, V, 1)`` torch tensor."""
        torch = self._torch
        # (T, V, C) → (C, T, V) → (1, C, T, V, 1)
        t = torch.from_numpy(frames_TVC).float().permute(2, 0, 1).contiguous()
        t = t.unsqueeze(0).unsqueeze(-1)  # add N=1 and M=1 dims
        return t.to(self.device)

    # ---- Core inference ------------------------------------------------

    def analyze_frame(self, frame: AnalyzerFrame) -> AnalyzerResult:
        self._frames_seen += 1
        feedback: List[str] = []

        # 1) Rep counting (geometric heuristic on raw, un-normalised hips).
        hip_y = self._hip_y(frame)
        if hip_y is not None:
            self._hip_window.append(hip_y)
            smoothed = float(np.mean(self._hip_window))
            self._update_rep_counter(smoothed, feedback)

        # 2) Stack + normalise + buffer.
        raw = self._stack_frame(frame)
        self._raw_buffer.append(raw)
        normed = normalize_pose(raw) if self._normalize else raw
        self._buffer.append(normed)
        self._frames_since_inference += 1

        # 3) Quality from CTR-GCN, triggered on stride once the window is full.
        is_compensatory = False
        ran_inference = False
        if (
            len(self._buffer) == self.window_size
            and self._frames_since_inference >= self.inference_stride
        ):
            quality, is_compensatory = self._infer_quality()
            ran_inference = True
            self._frames_since_inference = 0
            self._inference_calls += 1
            # EMA toward the model's estimate to suppress per-frame jitter.
            self._quality_running = (
                0.7 * self._quality_running + 0.3 * quality
            )
            if is_compensatory:
                self._compensation_events += 1
                feedback.append("Form drift detected — slow down and reset")

        # 4) Joint-angle bounds (cheap client-side hints, retained).
        for joint, angle in frame.angles.items():
            if "knee" in joint and angle < 70:
                feedback.append("Knees collapsing inward")
                is_compensatory = True
            elif "back" in joint and angle < 150:
                feedback.append("Keep your chest up")
                is_compensatory = True

        self._quality_samples.append(self._quality_running)

        # Append the per-frame log entry (Phase 3 Task 7). Keep it cheap:
        # one dict, a few floats, no numpy allocation per frame.
        frame_angles = self._latest_frame_angles(raw)
        self._session_log.append({
            "frame_index": frame.frame_index,
            "t_ms": float(frame.timestamp_ms),
            "similarity": self._latest_similarity,
            "angles": frame_angles,
        })

        diagnostics: Dict[str, Any] = {
            "window_full": len(self._buffer) == self.window_size,
            "ran_inference_this_frame": ran_inference,
            "inference_calls": self._inference_calls,
            "latest_logits": (
                self._latest_logits.tolist() if self._latest_logits is not None else None
            ),
            "similarity_score": self._latest_similarity,
        }
        if ran_inference:
            diagnostics["kinematics"] = self._window_kinematics_summary()

        return AnalyzerResult(
            frame_index=frame.frame_index,
            count=self._rep_count,
            quality_score=self._quality_running,
            is_compensatory=is_compensatory,
            feedback=feedback,
            diagnostics=diagnostics,
        )

    def _infer_quality(self) -> Tuple[float, bool]:
        torch = self._torch
        # (T, V, 3) stacked from window order.
        arr = np.stack(list(self._buffer), axis=0)
        with torch.no_grad():
            x = self._make_tensor(arr)
            # Capture the pre-FC feature vector for similarity scoring.
            features, logits = self._model(x, return_features=True)
            probs = torch.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy().reshape(-1)
        self._latest_logits = logits.detach().cpu().numpy().reshape(-1)
        self._latest_features = features.detach().cpu().numpy().reshape(-1)
        self._latest_similarity = similarity_score(
            self._latest_features, self._reference_feature,
        )
        good = float(probs_np[self._good_idx]) if self._good_idx < probs_np.size else 0.0
        bad = float(probs_np[self._bad_idx]) if self._bad_idx < probs_np.size else 0.0
        is_compensatory = bad > good
        return good, is_compensatory

    def extract_features(self) -> Optional[np.ndarray]:
        """Return the latest pre-FC feature vector, or ``None`` if not yet run.

        Exposed so the calling layer (e.g. a "record this as the target
        movement" admin tool) can grab a feature vector without parsing
        diagnostics payloads.
        """
        if self._latest_features is None:
            return None
        return self._latest_features.copy()

    # ---- Rep counter (shared logic, factored out for clarity) ---------

    @staticmethod
    def _hip_y(frame: AnalyzerFrame) -> Optional[float]:
        lh = frame.points.get("left_hip")
        rh = frame.points.get("right_hip")
        if lh and rh and len(lh) > 1 and len(rh) > 1:
            return 0.5 * (float(lh[1]) + float(rh[1]))
        return None

    def _update_rep_counter(self, smoothed: float, feedback: List[str]) -> None:
        self._smoothed_history.append(smoothed)
        if self._last_extreme is None:
            self._last_extreme = smoothed
            return
        delta = smoothed - self._last_extreme
        MIN_EXC = 0.06
        if self._direction >= 0 and delta > 0:
            self._direction = 1
            self._last_extreme = max(self._last_extreme, smoothed)
        elif self._direction == 1 and delta < -MIN_EXC:
            self._direction = -1
            self._last_extreme = smoothed
        elif self._direction == -1 and delta < 0:
            self._last_extreme = min(self._last_extreme, smoothed)
        elif self._direction == -1 and delta > MIN_EXC:
            self._rep_count += 1
            self._direction = 1
            self._last_extreme = smoothed
            feedback.append("Rep counted")

    # ---- Per-frame angle snapshot --------------------------------------

    def _latest_frame_angles(self, raw_frame_VC: np.ndarray) -> Dict[str, float]:
        """Compute every clinical joint angle for the current raw frame.

        Used by the session log so the synthesis layer has dense angle
        coverage even on frames where the model didn't fire. Cheap:
        six 3-vector dot products per frame.
        """
        out: Dict[str, float] = {}
        for joint, triplet in JOINT_ANGLE_TRIPLETS.items():
            try:
                ia = LANDMARK_NAMES.index(triplet[0])
                iv = LANDMARK_NAMES.index(triplet[1])
                ib = LANDMARK_NAMES.index(triplet[2])
            except ValueError:
                continue
            angle = joint_angle(raw_frame_VC[ia], raw_frame_VC[iv], raw_frame_VC[ib])
            # Only log finite angles so downstream code can `np.isnan` clean.
            if angle == angle:  # i.e. not NaN
                out[joint] = float(angle)
        return out

    # ---- Kinematics ----------------------------------------------------

    def _window_kinematics_summary(self) -> Dict[str, Any]:
        """ROM + tremor for each tracked joint over the current window.

        Uses the *un-normalised* buffer so angles are in real degrees
        (normalised coords still give the same angle but it's clearer to
        compute on the canonical input). The output dict has one entry
        per joint defined in ``JOINT_ANGLE_TRIPLETS``.
        """
        if len(self._raw_buffer) < 3:
            return {}
        window = np.stack(list(self._raw_buffer), axis=0)
        summary: Dict[str, Any] = {}
        for joint, triplet in JOINT_ANGLE_TRIPLETS.items():
            angles = joint_angle_series(window, LANDMARK_NAMES, triplet)
            rom = range_of_motion(angles)
            tremor = tremor_metrics(angles)
            summary[joint] = {**rom, **tremor}
        return summary

    # ---- Summary -------------------------------------------------------

    def generate_summary(self) -> AnalyzerSummary:
        quality = float(np.mean(self._quality_samples)) if self._quality_samples else 0.0
        if len(self._smoothed_history) >= 3:
            spread = float(np.std(np.diff(self._smoothed_history)))
            stability = max(0.0, 1.0 - min(1.0, spread * 50.0))
        else:
            stability = 0.0
        progress_trend: Dict[str, Any] = {
            "frames_analyzed": self._frames_seen,
            "quality_curve_samples": len(self._quality_samples),
            "analyzer": self.name,
            "window_size": self.window_size,
            "inference_stride": self.inference_stride,
            "inference_calls": self._inference_calls,
        }
        if self._latest_similarity is not None:
            progress_trend["latest_similarity_score"] = self._latest_similarity
        if self._raw_buffer:
            progress_trend["final_window_kinematics"] = self._window_kinematics_summary()

        # Phase 4 Task 8: Post-workout synthesis fed to the React Summary
        # Box. Keys are stable and consumed by SummaryBox.jsx; do not
        # rename without updating the frontend.
        progress_trend["summary_box"] = synthesize(
            self._session_log,
            quality_samples=self._quality_samples,
        )

        return AnalyzerSummary(
            rep_count=self._rep_count,
            overall_stability_score=stability,
            quality_score=quality,
            compensation_events=self._compensation_events,
            progress_trend=progress_trend,
            random_seed=self.seed,
        )
