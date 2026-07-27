"""Export a built CTR-GCN model to ONNX, falling back to TorchScript.

Plan task 8: the live edge deployment needs a deploy artefact that
isn't a Python pickle of the training graph. We try ONNX first because
it's the most portable, but CTR-GCN's adaptive topology refinement
involves ``einsum`` calls (see ``ctrgcn.ctrgcn.CTRGC.forward``) which
some ONNX opset versions can't represent. When that conversion raises,
we pivot to TorchScript tracing, which natively handles every op
PyTorch can run.

The function returns the absolute path of the artefact that was
actually written, plus the format that succeeded. Callers — typically a
Django management command — surface that to the operator.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Tuple

logger = logging.getLogger("rehab")


def export_ctrgcn(
    analyzer,
    output_path: str,
    *,
    onnx_opset: int = 16,
) -> Tuple[Path, Literal["onnx", "torchscript"]]:
    """Export ``analyzer._model`` to ONNX or, on failure, TorchScript.

    ``analyzer`` is an already-built ``CTRGCNAnalyzer`` — we reuse its
    model and the shape it tells us its forward pass expects, so the
    exported artefact matches the live inference shape bit-for-bit.
    """
    import torch  # local import to keep the analyzer package torch-free at import

    model = analyzer._model
    model.eval()
    shape = analyzer.buffer_tensor_shape()
    dummy = torch.zeros(*shape, dtype=torch.float32, device=analyzer.device)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- ONNX first ---------------------------------------------------
    try:
        onnx_path = out.with_suffix(".onnx")
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=onnx_opset,
        )
        logger.info("Exported CTR-GCN to ONNX at %s", onnx_path)
        return onnx_path, "onnx"
    except Exception as exc:  # ONNX may reject einsum-heavy ops
        logger.warning(
            "ONNX export failed (%s); falling back to TorchScript.", exc,
        )

    # --- TorchScript fallback ----------------------------------------
    ts_path = out.with_suffix(".pt")
    traced = torch.jit.trace(model, dummy, strict=False)
    traced.save(str(ts_path))
    logger.info("Exported CTR-GCN to TorchScript at %s", ts_path)
    return ts_path, "torchscript"


__all__ = ["export_ctrgcn"]
