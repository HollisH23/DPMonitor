"""Deterministic seeding for the analyzer layer.

Why this matters for clinical/academic use:
  * Reviewers must be able to reproduce a session's quality scores and
    rep count *exactly* from the persisted trajectory.
  * We therefore expose `apply_global_seed` (process-level) and a
    `deterministic_context` context manager that scopes the seed to a
    single analyzer run without leaking state to other threads.

We seed every randomness source the analyzer layer actually consumes:
`random`, `numpy`, and — once the CTR-GCN analyzer is wired in —
`torch` (CPU + CUDA, deterministic cuDNN). Torch is imported lazily
so that test environments that opt out of the heavy dep can still use
this module for the lightweight analyzers.
"""
from __future__ import annotations

import contextlib
import random
from typing import Any, Iterator, Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is in requirements.txt
    np = None  # type: ignore


def _seed_torch(seed: int) -> bool:
    """Seed torch CPU + CUDA RNGs if torch is importable. Returns True if seeded.

    Kept as a soft dependency: importing the analyzer layer must not crash
    when torch isn't installed (the placeholder analyzer doesn't need it).
    """
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - torch is optional at import time
        return False
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic so repeated inference matches bit-for-bit.
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return True


def _get_torch_state() -> Optional[Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch.get_rng_state()


def _set_torch_state(state: Any) -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return
    torch.set_rng_state(state)


def apply_global_seed(seed: int) -> None:
    """Seed every randomness source the analyzer relies on."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    _seed_torch(seed)


@contextlib.contextmanager
def deterministic_context(seed: int) -> Iterator[int]:
    """Temporarily fix RNG state, restoring whatever was there before.

    Usage::

        with deterministic_context(1337):
            result = analyzer.analyze_frame(frame)

    The original `random`, `numpy` and (if available) `torch` RNG states
    are restored on exit, so callers that interleave deterministic
    analyzer runs with other randomised work won't see surprising
    coupling.
    """
    py_state = random.getstate()
    np_state: Optional[object] = None
    if np is not None:
        np_state = np.random.get_state()
    torch_state = _get_torch_state()

    apply_global_seed(seed)
    try:
        yield seed
    finally:
        random.setstate(py_state)
        if np is not None and np_state is not None:
            np.random.set_state(np_state)  # type: ignore[arg-type]
        if torch_state is not None:
            _set_torch_state(torch_state)
