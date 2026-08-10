# Bundled model assets

Two binaries belong in this directory. Neither is checked into git — both
are generated or downloaded, and both are large.

## 1. `CTRGCN.mlpackage`

Generated from the PyTorch model:

```bash
python scripts/export_coreml.py
```

The script writes `CTRGCN.mlpackage`, `CTRGCN.onnx` and
`CTRGCN.manifest.json` here. Xcode compiles the `.mlpackage` into
`CTRGCN.mlmodelc` at build time; `ActionClassifier` loads that by name.

Contract (asserted by `scripts/export_coreml.py --validate`):

| feature    | shape              | notes                              |
|------------|--------------------|------------------------------------|
| `input`    | `(1, 3, 64, 33, 1)` | N, C, T, V, M — float32            |
| `logits`   | `(1, num_class)`    | raw, softmaxed on device           |
| `features` | `(1, 256)`          | pre-FC embedding, similarity input |

## 2. `pose_landmarker_full.task`

The MediaPipe BlazePose bundle. Download it once:

```bash
curl -L -o ios/DPMonitor/Models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

`full` is the Tasks-API equivalent of the legacy Solutions
`modelComplexity: 2` the web app uses, so the landmarks match. Swap
`PoseExtractorConfig.modelName` to `pose_landmarker_lite` if the thermal
budget demands it — accuracy drops, particularly on the extremities.

Both files are ordinary bundle resources; XcodeGen picks them up
automatically because this directory is inside the target's source path.
