# DPMonitor iOS — offline edge action analysis

A fully on-device rehabilitation movement analyser for iPhone 16 Pro Max.
Pose extraction, EMA smoothing, spatial normalisation, CTR-GCN inference,
kinematics and storage all run locally. **No backend, no cloud, no network
request.**

---

## Build

```bash
brew install xcodegen cocoapods

cd ios
xcodegen generate          # writes DPMonitor.xcodeproj from project.yml
pod install                # links MediaPipeTasksVision
open DPMonitor.xcworkspace # ← the WORKSPACE, not the project
```

Regenerate the project whenever files are added or removed. Nothing lives
in a hand-maintained `.pbxproj`.

### Bundled assets (both required, neither in git)

```bash
# 1. Core ML model — from the PyTorch source of truth
python scripts/export_coreml.py

# 2. MediaPipe BlazePose Full bundle
curl -L -o ios/DPMonitor/Models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

See [`DPMonitor/Models/README.md`](DPMonitor/Models/README.md) for the full
I/O contract. The app degrades gracefully if either is absent: the HUD shows
"Model not bundled" rather than crashing.

### Swift Package alternative

Google ships MediaPipe Tasks for iOS through CocoaPods only. If you must
avoid CocoaPods, add a community SPM mirror to `project.yml`:

```yaml
packages:
  MediaPipeTasksVision:
    url: https://github.com/<mirror>/mediapipe-spm
    from: "0.10.14"

targets:
  DPMonitor:
    dependencies:
      - package: MediaPipeTasksVision
```

`PoseExtractor` is wrapped in `#if canImport(MediaPipeTasksVision)`, so the
project compiles either way — it just throws
`PoseExtractorError.mediaPipeUnavailable` at runtime if the framework is
missing.

---

## Architecture

```
CMSampleBuffer  (camera delegate queue, 30 FPS)
  └─ PoseExtractor  ─ MediaPipe PoseLandmarker (33 landmarks)
      │
      ├─ Branch A  screen-relative (x, y ∈ [0,1])
      │    ├─ RepCounter          hip-y hysteresis  ← needs screen space
      │    └─ main ─ SkeletonOverlayView            CADisplayLink @120 Hz
      │
      └─ Branch B  world landmarks (metres)
           ├─ PoseSmoother        EMA, α 0.6 / 0.15 when occluded
           ├─ PoseNormalizer      hip-centre + spine-scale (vDSP)
           ├─ OcclusionHandler    carry-forward below visibility 0.5
           ├─ FrameBuffer         64-frame window → (1,3,64,33,1)
           ├─ every 5th frame ─ ActionClassifier (Core ML, ANE)
           ├─ Kinematics          ROM, tremor
           └─ main ─ HUDView
```

### Centering assistant

The camera and pose extraction start when the screen appears, **not** when
recording starts — the framing assistant has to see the patient in order to
guide them, and a black preview behind "stand in the middle" would be
useless. `PipelineCore.isRecording` is what separates "frames are flowing"
from "frames count towards a session"; without it the rep counter would tick
up while the patient was still walking into position.

* **Visibility** — guides show automatically whenever a session is not
  recording. During a session they are opt-in via the "Framing guides"
  toggle, and render in a dimmed compact form so they do not compete with
  the skeleton and HUD.
* **Gating** — being off-centre **warns, never blocks**. A hard gate would
  lock out exactly the patients who most need the app: someone in a
  wheelchair, in a small room, or standing with assistance may be physically
  unable to hit the ideal box. The alert states the cost and offers
  "Start anyway".
* **Handedness** — messages are frame-relative ("Patient too far LEFT" means
  their body is toward x = 0), copied verbatim from the desktop reference so
  a clinician comparing a JSONL recording to a screenshot sees the same
  words. The on-screen arrow points toward frame centre, which reads
  correctly whether or not the preview is mirrored.

### Threads

| Queue            | Owns                                                    |
|------------------|---------------------------------------------------------|
| camera delegate  | `CMSampleBuffer` → `MPImage` → `detectAsync`             |
| `pipelineQueue`  | every piece of numeric state, exclusively                |
| `inferenceQueue` | `MLModel.prediction` — never blocks camera or main       |
| main             | all `@Published` mutations, overlay, HUD                 |

`isInferenceInFlight` prevents overlapping forward passes. Without it a slow
pass under thermal throttling would queue work faster than it drains and the
UI would fall further behind every second.

### Why two branches

The rep counter *must* use screen-space coordinates. World landmarks are
hip-centred, so their hip y is ≈ 0 on every frame and carries no rep signal
at all. Conversely the model *must* use world landmarks: it was configured
for absolute spatial coordinates, and screen-normalised x/y paired with an
unrelated z scale corrupts the depth channel.

---

## Parity with the Python pipeline

Four numeric components are re-implementations, so they are pinned against
golden values produced by the actual Python reference:

```bash
python scripts/gen_reference_fixtures.py   # → ios/DPMonitorTests/Fixtures/
```

| Swift                  | Python / JS reference                          |
|------------------------|------------------------------------------------|
| `PoseNormalizer`       | `analyzer/normalization.py :: normalize_pose`  |
| `OcclusionHandler`     | `… :: apply_occlusion_carryforward`            |
| `Kinematics`           | `analyzer/kinematics.py`                       |
| `PoseSmoother`         | `frontend/src/lib/poseSmoothing.js`            |
| `Synthesis`            | `analyzer/synthesis.py`                        |
| `CenteringEvaluator`   | `analyzer/centering.py`                        |
| `FrameBuffer` layout   | `CTRGCNAnalyzer._make_tensor`                  |

`backend/analyzer/centering.py` is itself a port of the desktop original at
`../Final/centering_logic.py`, and is verified against it byte-for-byte
(status string, BGR colour, detail lines, rounded metrics) over 20,000
randomised inputs.

> **Threshold fixtures straddle, they do not sit on, the boundary.**
> Landmarks are `Float` on iOS and float64 in Python; at a threshold the two
> differ by ~1e-8 and can take opposite branches — `(0.21 + 0.39) / 2` is
> `0.30000000000000004` in float64 but `0.29999999` in `Float`. The centering
> fixtures therefore test ±0.002 either side. Do not tighten them to exact
> boundary values; that asserts something floating point cannot provide.
> Exact inclusive/exclusive semantics are pinned in Python only, by
> `CenteringTests`.

Tolerance is **1e-4**. Regenerate the fixtures whenever the Python changes,
then run the tests and decide whether a red result was intended.

```bash
xcodebuild test -workspace DPMonitor.xcworkspace -scheme DPMonitor \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro Max'
```

### Known deviations, and why

1. **Kinematics are computed from world landmarks, not screen coordinates.**
   The Python pipeline computes joint angles from raw MediaPipe screen
   coordinates, whose z channel is in an arbitrary scale. World landmarks are
   true metres, so on-device angles are *more* accurate — but absolute ROM
   degrees will not match the web app exactly. Compare rep counts and quality
   scores for cross-platform parity; treat ROM as platform-relative.

2. **Session preset is 720p30, not `.photo`.** BlazePose downsamples to
   256×256 internally, so a `.photo`-preset feed buys no accuracy and costs
   significant heat.

3. **`ActionClassifier` uses the generic `MLModel` API**, not the
   Xcode-generated `CTRGCN` class. The generated class only exists once the
   `.mlpackage` is in the target, so depending on it would make the whole
   project fail to compile before the export script has run.

4. **`modelComplexity: .full` maps to bundling `pose_landmarker_full.task`.**
   The MediaPipe Tasks API selects complexity by model bundle, not by an
   options flag — that was the legacy Solutions API the web app uses.

---

## Model export

```bash
python scripts/export_coreml.py            # export + validate
python scripts/export_coreml.py --no-validate
python scripts/export_coreml.py --weights path/to/ckpt.pt
```

The script builds the model exactly as `CTRGCNAnalyzer._build_model` does
(33-node MediaPipe graph, `num_class=2`, adaptive), traces it, and converts
to a FP16 `.mlpackage` targeting iOS 18.

**On weights:** no fine-tuned checkpoint ships with the repository. The model
is built under `apply_global_seed(1337)`, so the bundled weights are
*reproducible*, not *random* — re-running the export gives a bit-identical
package. No code changes are needed when real weights arrive; pass
`--weights`.

> **The uncalibrated build cannot produce a form score at all.** This was
> measured, not assumed. With untrained weights the logits reach ~1e4 after
> ten GCN blocks running on default BatchNorm statistics, so the softmax
> saturates to exactly `[0, 1]` on *every* window. Left alone that pins the
> gauge at 0%, marks all 32 windows of a 220-frame session compensatory, and
> paints the skeleton permanently red — indistinguishable from a broken app.
>
> The app therefore reads `weights` from `CTRGCN.manifest.json` at launch. If
> it is `"seeded-random-init"` (or the manifest is missing), `ActionClassifier`
> reports `isCalibrated == false` and the UI shows the form score as `—` with
> an "Uncalibrated model" badge, a neutral white skeleton, and no drift
> alerts. Inference still runs — it exercises the whole Core ML path and the
> feature embedding stays usable for similarity.
>
> **Everything geometric remains valid and is still shown**: rep count, range
> of motion, stability and tremor never touch the model.

**On ONNX:** `coremltools` removed its ONNX front-end in 6.0, so the
`.mlpackage` is built from a TorchScript trace. The `.onnx` file is still
emitted as a portable archival artefact. The `torch.einsum` in
`CTRGC.forward` that used to block ONNX export has been rewritten as
`matmul` + `permute` (verified equivalent to 8.9e-16).

**Exit codes:** `0` success · `1` parity failure · `2` conversion succeeded
but no `.mlpackage` could be written on this host. Code 2 happens off macOS:
serialising a Core ML package needs the `libmilstoragepython` native
extension, which coremltools ships for macOS and x86-64 Linux only. On such a
host the script still exports and validates the ONNX graph, so the export is
meaningfully checked — but you must re-run on your Mac before building.

**Validation tolerance is relative, not absolute.** With untrained weights the
logits are ~1e4, where float32 accumulation noise alone is a few 1e-3; an
absolute 1e-3 threshold reports failure on a graph that is in fact correct.
Measured PyTorch↔ONNX agreement: `rel 2.7e-07` on logits, `2.0e-07` on
features, and `0.0` on the softmax the app actually reads.

### NPU profiling (manual)

The export script prints a MIL op-type census and flags op types that
commonly fall off the Neural Engine. That is a *preliminary* check only. For
authoritative per-layer placement:

1. Open `CTRGCN.mlpackage` in Xcode.
2. Select the **Performance** tab → **+** → choose a connected iPhone 16 Pro Max.
3. Read the per-layer compute-unit breakdown; note any CPU/GPU fallback.

---

## Verification

### Airplane-mode test

1. Enable Airplane Mode (and turn Wi-Fi off in Control Centre).
2. Install and launch the app.
3. Run a full session: start → 10 reps → stop → results → history.
4. Everything must work. There is no offline banner because there is no
   online path — the target links no networking framework.

### Thermal test

1. Run continuously for 15 minutes on an iPhone 16 Pro Max.
2. Watch the HUD thermal badge and `ProcessInfo.thermalState`.
3. Must not reach `.critical` or trigger screen dimming.

`SessionAnalyzer.applyThermalState` widens the inference stride (5 → 10 → 20)
as thermal pressure rises. Overlay and rep counting stay at full rate; only
the quality score updates more slowly. That is a far better failure mode than
letting iOS throttle the whole app.

### Accuracy parity

Record a reference exercise, process it on both the web app and iOS, and
compare **rep counts** and **quality scores**. See deviation 1 above before
comparing ROM degrees.

---

## Privacy

* Camera frames are consumed in memory and never written to disk.
* No photo-library, microphone or location permission is requested.
* Session history is stored in a local Core Data store with
  `FileProtectionType.completeUntilFirstUserAuthentication`.
* No CloudKit container is configured, deliberately.
* There is no share or export path in the UI.
