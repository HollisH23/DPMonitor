"""Analyzer package: plug-and-play AI / counting interface.

UI code is shielded from analyzer internals; it only consumes
`quality_score`, `count`, and `feedback` from these classes.

The CTR-GCN analyzer pulls in PyTorch + the ctrgcn sub-package; we
import it lazily so environments that haven't yet installed torch
can still use the placeholder analyzer (and run the lightweight
tests against it).
"""
from .base import AnalyzerFrame, AnalyzerResult, AnalyzerSummary, BaseAnalyzer
from .placeholder import PlaceholderAnalyzer
from .seed import apply_global_seed, deterministic_context

__all__ = [
    "AnalyzerFrame",
    "AnalyzerResult",
    "AnalyzerSummary",
    "BaseAnalyzer",
    "PlaceholderAnalyzer",
    "CTRGCNAnalyzer",
    "apply_global_seed",
    "deterministic_context",
    "get_analyzer",
]


def __getattr__(name: str):  # noqa: D401
    """Lazy import for the heavy CTR-GCN analyzer.

    Keeps `from analyzer import PlaceholderAnalyzer` cheap on import,
    while still exposing `from analyzer import CTRGCNAnalyzer` for
    callers that actually want it.
    """
    if name == "CTRGCNAnalyzer":
        from .ctrgcn_analyzer import CTRGCNAnalyzer as _CTRGCNAnalyzer
        return _CTRGCNAnalyzer
    raise AttributeError(f"module 'analyzer' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Registry — single place to flip the active analyzer.
# ---------------------------------------------------------------------------


def get_analyzer(name: str, *, seed: int, exercise_type: str = "custom", **kwargs):
    """Resolve an analyzer by short name.

    Used by the WebSocket consumer so the active model is configurable
    from Django settings (`REHAB_ANALYZER`) without hard-coding a class
    import in the consumer.
    """
    name = (name or "").lower()
    if name in ("placeholder", "placeholder-v1"):
        return PlaceholderAnalyzer(seed=seed, exercise_type=exercise_type)
    if name in ("ctrgcn", "ctrgcn-v1"):
        from .ctrgcn_analyzer import CTRGCNAnalyzer
        return CTRGCNAnalyzer(seed=seed, exercise_type=exercise_type, **kwargs)
    raise ValueError(f"Unknown analyzer {name!r}; expected 'placeholder' or 'ctrgcn'.")
