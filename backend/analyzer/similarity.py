"""Movement-similarity scoring against a reference exercise.

Phase 3 task 6: compare the patient's current CTR-GCN feature vector
``F_current`` against a stored "textbook" vector ``F_target`` using
cosine similarity, and map the result onto the intuitive 0–100 scale
the UI displays.

Cosine similarity already lives in ``[-1, 1]``; rehab movements rarely
go below 0 (they're variations of the same posture, not antipodal),
so we treat anything ≤ 0 as 0 and linearly remap ``[0, 1] → [0, 100]``.
This keeps the score easy to reason about for clinicians:

* 100 — pixel-perfect match against the reference.
* 80–95 — typical "good form" range with mild patient-specific drift.
* < 50 — substantial deviation; warrants clinician attention.

The reference vector is a numpy array supplied at session start (loaded
from disk by the caller). Storing references on disk is intentionally
out of scope of this module — it stays pure: vectors in, score out.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]``, returns 0.0 on zero vectors."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def similarity_score(
    current: np.ndarray, target: Optional[np.ndarray],
) -> Optional[float]:
    """0–100 similarity score, or ``None`` if no target reference is set.

    Negative similarity is clipped to 0 because in this domain it means
    "completely different posture", not "opposite posture". Returning
    ``None`` (rather than a default like 50) when no reference is loaded
    lets the UI distinguish "no clinician reference yet" from "matches
    poorly".
    """
    if target is None:
        return None
    sim = cosine_similarity(current, target)
    sim = max(0.0, sim)  # clip negatives — see module docstring
    return float(round(100.0 * sim, 2))


__all__ = ["cosine_similarity", "similarity_score"]
