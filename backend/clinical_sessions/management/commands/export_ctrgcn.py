"""Operator-facing command to bake the CTR-GCN graph into a deploy artefact.

Usage::

    python manage.py export_ctrgcn --out ./artifacts/ctrgcn
    python manage.py export_ctrgcn --out ./artifacts/ctrgcn \\
        --weights ./checkpoints/ctrgcn.pt --window 64

The command builds a ``CTRGCNAnalyzer`` (so the exported model shares
exactly the same weights and shape as the live pipeline), then delegates
to :func:`analyzer.export.export_ctrgcn`. ONNX is tried first; if any
op (notably ``einsum``) fails the converter, the command silently
pivots to TorchScript — both formats live happily next to a Django
edge-deployment.
"""
from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Export the live CTR-GCN graph to ONNX (or TorchScript on fallback)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--out", required=True,
            help="Output path (no extension; .onnx or .pt is appended).",
        )
        parser.add_argument(
            "--weights", default=None,
            help="Optional fine-tuned CTR-GCN checkpoint to load before export.",
        )
        parser.add_argument(
            "--window", type=int, default=None,
            help="Window length (frames) used for the dummy input.",
        )
        parser.add_argument(
            "--opset", type=int, default=16,
            help="ONNX opset version to target (default: 16).",
        )

    def handle(self, *args, **opts) -> None:
        from analyzer import get_analyzer
        from analyzer.export import export_ctrgcn

        window = opts["window"] or getattr(settings, "REHAB_CTRGCN_WINDOW", 64)
        weights = opts["weights"] or getattr(settings, "REHAB_CTRGCN_WEIGHTS", None) or None

        try:
            analyzer = get_analyzer(
                "ctrgcn",
                seed=settings.REHAB_RANDOM_SEED,
                exercise_type="custom",
                window_size=window,
                weights_path=weights,
            )
        except Exception as exc:  # torch import or model build failed
            raise CommandError(f"Failed to build CTR-GCN analyzer: {exc}") from exc

        path, fmt = export_ctrgcn(analyzer, opts["out"], onnx_opset=opts["opset"])
        self.stdout.write(self.style.SUCCESS(f"Exported as {fmt}: {path}"))
