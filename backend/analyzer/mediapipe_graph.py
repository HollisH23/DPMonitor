"""MediaPipe Pose graph adapter for CTR-GCN.

CTR-GCN's ``Model`` resolves its graph by dotted path through
``import_class(...)``; the chosen class must expose a ``Graph(**graph_args)``
constructor and an ``A`` attribute shaped ``(3, num_node, num_node)``. The
upstream repo ships ``ctrgcn.graph.ntu_rgb_d.Graph`` with 25 NTU joints,
but our pose-estimation pipeline runs MediaPipe Pose which emits 33
landmarks. Re-using NTU's topology would force a lossy remapping and
discard hand/foot/face detail, so we build a dedicated 33-node graph
here that mirrors MediaPipe's actual skeletal connectivity.

The edge set is the natural skeleton:

* face arc (nose → eyes → ears) + mouth bridge,
* upper body (shoulders ↔, shoulders → elbows → wrists, wrists → hand tips),
* torso (shoulders ↔ hips, hip cross-bar),
* lower body (hips → knees → ankles → heels → foot indices, ankle → toe).

All edge tuples are zero-indexed against MediaPipe's canonical landmark
order (see ``LANDMARK_NAMES`` for the lookup).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Pulled in lazily so the analyzer module can be imported in environments
# that haven't installed the (heavy) CTR-GCN package or its torch dep.
from ctrgcn.graph import tools  # type: ignore  # noqa: E402


# MediaPipe Pose 33-landmark canonical order.
LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)
NUM_NODE: int = len(LANDMARK_NAMES)  # 33


def _idx(name: str) -> int:
    return LANDMARK_NAMES.index(name)


# Edges expressed by landmark name for readability — converted to indices below.
_EDGE_PAIRS_BY_NAME: List[Tuple[str, str]] = [
    # Face
    ("nose", "left_eye_inner"), ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"), ("left_eye_outer", "left_ear"),
    ("nose", "right_eye_inner"), ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"), ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
    # Anchor the mouth to the rest of the head so the graph stays connected.
    ("nose", "mouth_left"), ("nose", "mouth_right"),
    # Shoulders & arms
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    # Hand tips
    ("left_wrist", "left_pinky"), ("left_wrist", "left_index"), ("left_wrist", "left_thumb"),
    ("right_wrist", "right_pinky"), ("right_wrist", "right_index"), ("right_wrist", "right_thumb"),
    # Torso
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Anchor head to torso through the shoulder line.
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    # Legs
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
]


# Zero-indexed directed edge lists, NTU convention: child → parent for inward.
_INWARD: List[Tuple[int, int]] = [(_idx(a), _idx(b)) for a, b in _EDGE_PAIRS_BY_NAME]
_OUTWARD: List[Tuple[int, int]] = [(b, a) for (a, b) in _INWARD]
_SELF_LINK: List[Tuple[int, int]] = [(i, i) for i in range(NUM_NODE)]


class Graph:
    """MediaPipe-33 spatial graph in the shape CTR-GCN expects."""

    def __init__(self, labeling_mode: str = "spatial") -> None:
        self.num_node = NUM_NODE
        self.self_link = _SELF_LINK
        self.inward = _INWARD
        self.outward = _OUTWARD
        self.neighbor = _INWARD + _OUTWARD
        self.A = self._build_adjacency(labeling_mode)

    def _build_adjacency(self, labeling_mode: str) -> np.ndarray:
        if labeling_mode != "spatial":
            raise ValueError(
                f"Only 'spatial' labeling is supported for the MediaPipe graph; "
                f"got {labeling_mode!r}."
            )
        return tools.get_spatial_graph(NUM_NODE, _SELF_LINK, _INWARD, _OUTWARD)


__all__ = ["Graph", "LANDMARK_NAMES", "NUM_NODE"]
