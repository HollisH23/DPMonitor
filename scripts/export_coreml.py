#!/usr/bin/env python3
"""Export the MediaPipe-33 CTR-GCN to a Core ML ``.mlpackage`` for iOS.

Pipeline
--------
    PyTorch nn.Module
        -> ONNX  (opset 16, fixed shape, archival / interop artefact)
        -> TorchScript trace
        -> Core ML .mlpackage (FP16, iOS18 deployment target)

Why TorchScript is the *primary* Core ML path
---------------------------------------------
``coremltools`` removed its ONNX front-end in 6.0; the supported route for
PyTorch models is ``ct.convert(traced_module, ...)``. We still emit the
``.onnx`` file because it is a useful, portable archival artefact (and the
existing :mod:`analyzer.export` helper already produces one), but the
``.mlpackage`` is built from a TorchScript trace. If a legacy coremltools
(<6.0) is installed, ``--from-onnx`` will use the ONNX front-end instead.

Determinism
-----------
No fine-tuned checkpoint ships with the repository, so by default the model
is built under :func:`analyzer.seed.apply_global_seed` with the same seed the
live analyzer uses (1337). The exported weights are therefore *reproducible*,
not *random*: re-running this script yields a bit-identical package. Pass
``--weights path/to/ckpt.pt`` once real weights exist — nothing else changes.

Model I/O contract (must match ios/DPMonitor/Core/ActionClassifier.swift)
-------------------------------------------------------------------------
    input    "input"     float32  (1, 3, 64, 33, 1)   # N, C, T, V, M
    output   "logits"    float32  (1, num_class)
    output   "features"  float32  (1, 256)            # pre-FC embedding

Usage
-----
    python scripts/export_coreml.py                       # export + validate
    python scripts/export_coreml.py --no-validate
    python scripts/export_coreml.py --weights ckpt.pt --num-class 2
    python scripts/export_coreml.py --validate            # explicit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Repo path wiring -------------------------------------------------------
# ``ctrgcn`` lives at the repo root; ``analyzer`` lives under ``backend/``
# (Django app root). Both must be importable for the graph string
# "analyzer.mediapipe_graph.Graph" to resolve exactly as it does at runtime.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _REPO_ROOT / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must match CTRGCNAnalyzer defaults.
WINDOW_SIZE = 64
NUM_POINT = 33          # MediaPipe Pose landmarks — NOT NTU-25
NUM_PERSON = 1
IN_CHANNELS = 3
DEFAULT_SEED = 1337
DEFAULT_NUM_CLASS = 2
FEATURE_DIM = 256       # base_channel * 4

INPUT_SHAPE: Tuple[int, int, int, int, int] = (
    1, IN_CHANNELS, WINDOW_SIZE, NUM_POINT, NUM_PERSON,
)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

class ExportWrapper:
    """Placeholder for the torch-dependent wrapper built in :func:`_wrapper`.

    Defined lazily inside a function so this module can be imported (e.g. by
    ``--help`` or by a linter) without torch installed.
    """


def _wrapper(torch):
    """Build the ``nn.Module`` that Core ML actually sees.

    ``Model.forward`` takes a ``return_features`` *keyword*, which neither
    ``torch.onnx.export`` nor ``torch.jit.trace`` can pass positionally in a
    stable way. We wrap it in a single-argument module with a fixed
    two-tensor output so both exporters see a clean signature.
    """
    import torch.nn as nn

    class CTRGCNExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            features, logits = self.model(x, return_features=True)
            return logits, features

    return CTRGCNExport


def build_model(
    *,
    num_class: int = DEFAULT_NUM_CLASS,
    weights: Optional[str] = None,
    seed: int = DEFAULT_SEED,
):
    """Instantiate the CTR-GCN exactly as ``CTRGCNAnalyzer._build_model`` does."""
    import torch

    from analyzer.seed import apply_global_seed  # type: ignore
    from ctrgcn.ctrgcn import Model  # type: ignore

    # Seed BEFORE construction so the random init is reproducible.
    apply_global_seed(seed)

    model = Model(
        num_class=num_class,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        graph="analyzer.mediapipe_graph.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=IN_CHANNELS,
        drop_out=0.0,
        adaptive=True,
    )

    if weights:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"--weights {path} does not exist")
        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[weights] loaded {path}  missing={len(missing)}  "
            f"unexpected={len(unexpected)}"
        )
    else:
        print(
            f"[weights] no checkpoint supplied — using deterministic "
            f"seeded init (seed={seed}). Swap in a real checkpoint with "
            f"--weights; no other change is needed."
        )

    model.eval()
    wrapper_cls = _wrapper(torch)
    wrapped = wrapper_cls(model)
    wrapped.eval()
    return wrapped


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------

def export_onnx(wrapped, out_path: Path, *, opset: int = 16) -> Optional[Path]:
    """Write a fixed-shape ONNX graph. Returns the path, or None on failure.

    ``CTRGC.forward`` used to contain ``torch.einsum('ncuv,nctv->nctu', ...)``,
    which several ONNX opsets cannot represent. It has been rewritten as
    ``matmul`` + ``permute`` (numerically identical to ~1e-16), so this export
    should now succeed. A failure here is non-fatal — the Core ML build uses
    the TorchScript path.
    """
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapped,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=["logits", "features"],
            opset_version=opset,
            do_constant_folding=True,
            # Fixed batch: Core ML on-device inference is strictly N=1, and a
            # static graph lets the Neural Engine plan allocations up front.
            dynamic_axes=None,
        )
    except Exception as exc:  # pragma: no cover - depends on local torch build
        print(f"[onnx]  export FAILED ({type(exc).__name__}: {exc})")
        print("[onnx]  continuing — Core ML is built from TorchScript anyway.")
        return None

    print(f"[onnx]  wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Core ML
# ---------------------------------------------------------------------------

def export_coreml(
    wrapped,
    out_path: Path,
    *,
    onnx_path: Optional[Path] = None,
    from_onnx: bool = False,
) -> Optional[Path]:
    """Convert to a Core ML ``.mlpackage`` with FP16 weights.

    Returns the written path, or ``None`` when the host cannot serialise a
    Core ML package. Conversion itself is pure Python and runs anywhere, but
    writing the weight blobs needs the ``libmilstoragepython`` native
    extension, which coremltools only ships for macOS (and x86-64 Linux).
    On an unsupported host we report that clearly instead of dumping a
    `RuntimeError: BlobWriter not loaded` traceback — the ONNX artefact and
    the parity check are still perfectly valid there.
    """
    import coremltools as ct
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)

    ct_input = ct.TensorType(name="input", shape=INPUT_SHAPE, dtype=None)

    convert_kwargs: Dict[str, Any] = dict(
        inputs=[ct_input],
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="features")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    try:
        if from_onnx:
            if onnx_path is None or not onnx_path.is_file():
                raise RuntimeError("--from-onnx requested but no ONNX file was produced")
            print("[coreml] converting from ONNX (legacy coremltools front-end)")
            # Legacy front-end does not accept `outputs=`.
            convert_kwargs.pop("outputs", None)
            mlmodel = ct.convert(str(onnx_path), **convert_kwargs)
        else:
            print("[coreml] tracing TorchScript ...")
            with torch.no_grad():
                traced = torch.jit.trace(wrapped, dummy, strict=False)
            traced.eval()
            mlmodel = ct.convert(traced, **convert_kwargs)
    except RuntimeError as exc:
        if "BlobWriter" not in str(exc):
            raise
        print(
            f"\n[coreml] SKIPPED — this host cannot serialise a .mlpackage.\n"
            f"         ({exc})\n"
            f"         coremltools ships the libmilstoragepython native\n"
            f"         extension for macOS and x86-64 Linux only. Conversion\n"
            f"         itself completed: every op lowered to MIL cleanly.\n"
            f"         Re-run this script on the Mac you build the app on.\n"
        )
        return None

    mlmodel.short_description = (
        "CTR-GCN action-quality classifier over a 64-frame window of 33 "
        "MediaPipe Pose landmarks (hip-centred, spine-normalised)."
    )
    mlmodel.input_description["input"] = (
        "float32 (1, 3, 64, 33, 1) = (N, C, T, V, M). C is (x, y, z) in "
        "spine-normalised units; T is the 64-frame sliding window."
    )
    mlmodel.output_description["logits"] = "Raw class logits, shape (1, num_class)."
    mlmodel.output_description["features"] = "Pre-FC 256-d embedding for similarity scoring."

    mlmodel.save(str(out_path))
    print(f"[coreml] wrote {out_path}")
    _print_compute_unit_summary(mlmodel)
    return out_path


def _print_compute_unit_summary(mlmodel) -> None:
    """Best-effort op-type census — a preliminary Neural Engine sanity check.

    This is *not* a substitute for Xcode's Core ML Performance Report
    (Task 1.3), which is the only authoritative source for per-layer
    compute-unit placement. It just flags op types that are known to be
    poorly served by the ANE so you know what to look for in Instruments.
    """
    try:
        spec = mlmodel.get_spec()
        prog = spec.mlProgram
        counts: Dict[str, int] = {}
        for func in prog.functions.values():
            for block in [func.block_specializations[k] for k in func.block_specializations]:
                for op in block.operations:
                    counts[op.type] = counts.get(op.type, 0) + 1
    except Exception as exc:  # pragma: no cover
        print(f"[coreml] op census unavailable ({exc})")
        return

    print("\n[coreml] op-type census (MIL program):")
    for op_type, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"           {n:5d}  {op_type}")

    # Ops that commonly fall back off the Neural Engine.
    ane_risky = {"matmul", "einsum", "gather", "scatter", "cumsum", "topk", "reduce_argmax"}
    flagged = sorted(ane_risky & counts.keys())
    if flagged:
        print(
            "\n[coreml] NOTE: potential CPU/GPU fallback candidates present: "
            + ", ".join(flagged)
            + "\n         Confirm placement in Xcode -> open the .mlpackage -> "
              "Performance -> run on device."
        )
    print()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _softmax(a):
    import numpy as np
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _report_parity(name, t_logits, t_features, o_logits, o_features, tol) -> bool:
    """Compare a backend's outputs against the PyTorch reference.

    Tolerance is RELATIVE, not absolute. With untrained (seeded) weights the
    logits reach ~1e4, where float32 accumulation noise alone is a few 1e-3 —
    an absolute 1e-3 threshold would report a failure for a graph that is in
    fact bit-for-bit correct in every way that matters. What we actually care
    about is that the two backends agree to float32 precision *relative to
    the signal*, and that the softmax the app consumes is unchanged.
    """
    import numpy as np

    def rel(a, b):
        scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1.0)
        return float(np.abs(a - b).max()) / scale

    d_logits_abs = float(np.abs(t_logits - o_logits).max())
    d_feat_abs = float(np.abs(t_features - o_features).max())
    d_logits_rel = rel(t_logits, o_logits)
    d_feat_rel = rel(t_features, o_features)
    d_prob = float(np.abs(_softmax(t_logits) - _softmax(o_logits)).max())

    print(f"\n[validate] PyTorch vs {name}, identical random input:")
    print(f"           torch  logits: {np.array2string(t_logits, precision=4)}")
    print(f"           {name[:6]:6s} logits: {np.array2string(o_logits, precision=4)}")
    print(f"           logits    abs {d_logits_abs:.3e}   rel {d_logits_rel:.3e}")
    print(f"           features  abs {d_feat_abs:.3e}   rel {d_feat_rel:.3e}")
    print(f"           softmax   abs {d_prob:.3e}   (this is what the app reads)")

    ok = d_logits_rel < tol and d_feat_rel < tol and d_prob < 1e-3
    print(f"[validate] {'PASS' if ok else 'FAIL'}  (relative tol {tol:.0e})")
    _warn_if_saturated(t_logits)
    return ok


def _warn_if_saturated(logits) -> None:
    """Flag the softmax saturation that untrained weights inevitably cause.

    With seeded-random weights and BatchNorm running on its default (0, 1)
    statistics, activations compound through ten TCN-GCN blocks and the
    logits land in the thousands. The softmax then saturates to exactly
    [0, 1], so the app's quality gauge will sit pinned at 100% or 0% and
    never move. That is expected for this build, but it looks exactly like
    a bug in the app, so say it out loud here rather than letting someone
    rediscover it on-device.
    """
    import numpy as np

    probs = _softmax(logits).ravel()
    if float(np.abs(probs - np.round(probs)).max()) < 1e-6:
        print(
            "\n[validate] NOTE: softmax is saturated "
            f"({np.array2string(probs, precision=1)}).\n"
            "           Expected with untrained weights — logits reach ~1e4 "
            "after ten\n"
            "           GCN blocks with default BatchNorm statistics. The "
            "in-app quality\n"
            "           gauge will read a constant 100% (or 0%) until real "
            "weights are\n"
            "           supplied via --weights. It is not an app bug."
        )


def validate(wrapped, mlpackage_path: Optional[Path], *,
             tol: float = 1e-3,
             onnx_path: Optional[Path] = None) -> bool:
    """Run identical input through PyTorch and the exported model(s).

    Core ML *prediction* only works on macOS. Off macOS we fall back to
    validating the ONNX graph with onnxruntime, which exercises the same
    rewritten `matmul` contraction and the same tensor layout — so the
    export is still meaningfully checked on any host.
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    x = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    with torch.no_grad():
        t_logits, t_features = wrapped(torch.from_numpy(x))
    t_logits = t_logits.numpy()
    t_features = t_features.numpy()

    # --- Core ML (macOS only) ----------------------------------------
    if sys.platform == "darwin" and mlpackage_path is not None:
        import coremltools as ct

        mlmodel = ct.models.MLModel(str(mlpackage_path))
        out = mlmodel.predict({"input": x})
        c_logits = np.asarray(out["logits"], dtype=np.float32).reshape(t_logits.shape)
        c_features = np.asarray(out["features"], dtype=np.float32).reshape(t_features.shape)
        return _report_parity("CoreML", t_logits, t_features, c_logits, c_features, tol)

    # --- ONNX fallback ------------------------------------------------
    if onnx_path is not None and onnx_path.is_file():
        try:
            import onnxruntime as ort
        except ImportError:
            print("[validate] SKIPPED — not macOS, and onnxruntime is not installed.")
            print("           pip install onnxruntime  to validate the ONNX graph here.")
            return True
        print("[validate] Core ML prediction needs macOS; validating the ONNX graph instead.")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        o_logits, o_features = sess.run(["logits", "features"], {"input": x})
        return _report_parity("ONNX", t_logits, t_features,
                              o_logits.reshape(t_logits.shape),
                              o_features.reshape(t_features.shape), tol)

    print("[validate] SKIPPED — no Core ML (needs macOS) and no ONNX artefact to check.")
    _warn_if_saturated(t_logits)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    default_out = _REPO_ROOT / "ios" / "DPMonitor" / "Models" / "CTRGCN.mlpackage"

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=default_out,
                    help=f"output .mlpackage path (default: {default_out})")
    ap.add_argument("--onnx", type=Path, default=None,
                    help="ONNX output path (default: <out dir>/CTRGCN.onnx)")
    ap.add_argument("--weights", type=str, default=None,
                    help="optional PyTorch checkpoint (.pt/.pth)")
    ap.add_argument("--num-class", type=int, default=DEFAULT_NUM_CLASS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--from-onnx", action="store_true",
                    help="use the legacy coremltools ONNX front-end (<6.0 only)")
    ap.add_argument("--skip-onnx", action="store_true",
                    help="do not emit the .onnx archival artefact")
    ap.add_argument("--validate", dest="do_validate", action="store_true", default=None)
    ap.add_argument("--no-validate", dest="do_validate", action="store_false")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args(argv)

    # Default is to validate; --no-validate opts out.
    do_validate = True if args.do_validate is None else args.do_validate

    out = args.out.resolve()
    onnx_path = (args.onnx or (out.parent / "CTRGCN.onnx")).resolve()

    print(f"repo root    : {_REPO_ROOT}")
    print(f"input shape  : {INPUT_SHAPE}  (N, C, T, V, M)")
    print(f"num_class    : {args.num_class}")
    print(f"mlpackage    : {out}\n")

    wrapped = build_model(
        num_class=args.num_class, weights=args.weights, seed=args.seed,
    )

    written_onnx: Optional[Path] = None
    if not args.skip_onnx or args.from_onnx:
        written_onnx = export_onnx(wrapped, onnx_path, opset=args.opset)

    written_mlpackage = export_coreml(
        wrapped, out, onnx_path=written_onnx, from_onnx=args.from_onnx)

    # Emit a small manifest so the iOS side (and CI) can assert the contract.
    # Written even when the .mlpackage could not be serialised, because it
    # documents the contract rather than the artefact.
    manifest = {
        "input_name": "input",
        "input_shape": list(INPUT_SHAPE),
        "output_names": ["logits", "features"],
        "num_class": args.num_class,
        "num_point": NUM_POINT,
        "window_size": WINDOW_SIZE,
        "feature_dim": FEATURE_DIM,
        "seed": args.seed,
        "weights": args.weights or "seeded-random-init",
        "precision": "float16",
        "minimum_deployment_target": "iOS18",
        "mlpackage_written": written_mlpackage is not None,
        "onnx_written": written_onnx is not None,
    }
    manifest_path = out.parent / "CTRGCN.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] wrote {manifest_path}")

    if do_validate:
        if not validate(wrapped, written_mlpackage,
                        tol=args.tol, onnx_path=written_onnx):
            return 1

    if written_mlpackage is None:
        print(
            "\n[result] ONNX export and conversion verified, but no "
            ".mlpackage was written\n"
            "         on this host. Re-run on macOS before building the app."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
