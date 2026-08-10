# iOS MVP — Standalone Edge AI Action Analysis for iPhone 16 Pro Max

## Goal

Build a **100% offline** native iOS app that performs real-time skeleton extraction, EMA smoothing, spatial normalization, and CTR-GCN inference entirely on-device. No backend, no cloud, no network requests.

## Critical Findings from Codebase Audit

> [!IMPORTANT]
> **The CTR-GCN model uses `torch.einsum` in `CTRGC.forward()` which will fail ONNX/CoreML conversion.**
>
> Line 174 of [ctrgcn.py](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/ctrgcn/ctrgcn.py#L174):
> ```python
> x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
> ```
> This must be rewritten to equivalent `matmul` + `permute` operations before export. An existing [export.py](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/backend/analyzer/export.py) utility already handles ONNX export with TorchScript fallback.

> [!WARNING]
> **No pre-trained weight files exist in the repository.** The model initializes with seeded random weights (`seed=1337`). For the MVP, we will bundle the deterministically-initialized weights. When real fine-tuned weights become available, they can be swapped in without code changes.

> [!NOTE]
> **The model uses 33 MediaPipe joints (not NTU-25)** with a custom spatial graph. V=33, M=1. The iOS app must match this exactly.

## Open Questions

> [!IMPORTANT]
> **Skeleton Framework Selection**: The plan proposes **Apple Vision Framework** (`VNDetectHumanBodyPose3DRequest`) as the primary choice for battery efficiency and NPU optimization. However, Vision only provides **17 body joints** (vs MediaPipe's 33), which means:
> - The CTR-GCN model was trained/configured for 33 joints
> - 16 joints (face details, hand tips, heels, foot indices) would be zero-filled
>
> **Options:**
> 1. Use Apple Vision (17 joints) + zero-fill the 16 missing joints → simpler, better battery, but may degrade model accuracy
> 2. Use MediaPipe iOS SDK (33 joints) → exact parity with web app, but requires bridging and may not leverage NPU
> 3. Retrain/reconfigure the CTR-GCN model for 17-joint Apple Vision skeleton → cleanest long-term but significant work
>
> **Recommendation**: Start with **Option 2 (MediaPipe iOS)** for accuracy parity in the MVP, since the model was specifically built for the 33-joint graph. Migrate to Apple Vision in Phase 2 with a retrained model.

## Project Structure

```
DPMonitor/
├── ios/                                    # [NEW] iOS app root
│   ├── DPMonitor.xcodeproj/
│   ├── DPMonitor/
│   │   ├── App/
│   │   │   ├── DPMonitorApp.swift          # SwiftUI app entry point
│   │   │   └── ContentView.swift           # Root navigation
│   │   ├── Models/
│   │   │   └── CTRGCN.mlpackage            # Bundled Core ML model
│   │   ├── Core/
│   │   │   ├── PoseExtractor.swift         # AVFoundation + MediaPipe bridge
│   │   │   ├── PoseSmoother.swift          # EMA filter (Branch B)
│   │   │   ├── PoseNormalizer.swift         # Hip-center + spine-scale
│   │   │   ├── OcclusionHandler.swift       # Visibility carry-forward
│   │   │   ├── FrameBuffer.swift           # 64-frame sliding window
│   │   │   ├── ActionClassifier.swift       # Core ML inference wrapper
│   │   │   ├── RepCounter.swift            # Hip y-excursion heuristic
│   │   │   ├── Kinematics.swift            # Joint angles, ROM, tremor
│   │   │   └── SessionAnalyzer.swift        # Orchestrator (= Python analyzer)
│   │   ├── Views/
│   │   │   ├── CameraView.swift            # AVCaptureVideoPreviewLayer
│   │   │   ├── SkeletonOverlayView.swift   # Metal/CALayer AR overlay
│   │   │   ├── HUDView.swift               # Rep counter overlay
│   │   │   ├── SessionView.swift           # Live monitor screen
│   │   │   ├── ResultsView.swift           # Post-session summary
│   │   │   └── HistoryView.swift           # Session history list
│   │   ├── Storage/
│   │   │   ├── SessionStore.swift          # CoreData manager
│   │   │   └── DPMonitor.xcdatamodeld/     # CoreData schema
│   │   └── Resources/
│   │       └── Assets.xcassets
│   └── DPMonitorTests/
│       ├── PoseNormalizerTests.swift        # Parity tests vs Python
│       └── KinematicsTests.swift
├── scripts/
│   └── export_coreml.py                    # [NEW] PyTorch → ONNX → Core ML
├── backend/                                # Existing (unchanged)
├── frontend/                               # Existing (unchanged)
└── ctrgcn/                                 # Existing (modified for export)
```

---

## Proposed Changes

---

### Task Group 1: Core ML Model Conversion

#### Task 1.1: Eliminate `einsum` for ONNX Compatibility

##### [MODIFY] [ctrgcn/ctrgcn.py](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/ctrgcn/ctrgcn.py)

Replace the `einsum` in `CTRGC.forward()` (line 174) with equivalent matmul:

```diff
  def forward(self, x, A=None, alpha=1):
      x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
      x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
      x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
-     x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
+     # Equivalent to einsum('ncuv,nctv->nctu', x1, x3):
+     # x1 is (N,C,U,V), x3 is (N,C,T,V) → contract over V → (N,C,T,U)
+     # matmul: (N,C,U,V) @ (N,C,V,T) → (N,C,U,T) → permute → (N,C,T,U)
+     x1 = torch.matmul(x1, x3.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
      return x1
```

**Validation**: Run existing backend tests to verify numerical equivalence.

---

#### Task 1.2: Create Core ML Export Script

##### [NEW] scripts/export_coreml.py

Python script that:
1. Instantiates the CTR-GCN model with the MediaPipe 33-joint graph (same config as `CTRGCNAnalyzer._build_model()`)
2. Loads weights if available, otherwise uses seeded initialization
3. Exports to ONNX (opset 16) with fixed input shape `(1, 3, 64, 33, 1)`
4. Converts ONNX → Core ML `.mlpackage` using `coremltools`
5. Applies FP16 quantization for model size reduction
6. Validates output parity: PyTorch vs Core ML inference on identical dummy input

```python
# Key conversion steps:
import coremltools as ct

mlmodel = ct.convert(
    onnx_path,
    inputs=[ct.TensorType(name="input", shape=(1, 3, 64, 33, 1))],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
)
mlmodel.save("CTRGCN.mlpackage")
```

---

#### Task 1.3: NPU Profiling

After conversion, open `CTRGCN.mlpackage` in Xcode and use **Core ML Performance Report** to verify all operations target the Neural Engine. Flag any CPU/GPU fallback ops.

> [!NOTE]
> This is a manual Xcode Instruments step. The export script will print a layer-by-layer compute unit summary from `coremltools` as a preliminary check.

---

### Task Group 2: iOS Video Pipeline

#### Task 2.1: AVFoundation Camera Capture

##### [NEW] ios/DPMonitor/Core/PoseExtractor.swift

- Configure `AVCaptureSession` with `.photo` preset for high resolution
- Use `AVCaptureVideoDataOutput` with `kCVPixelFormatType_32BGRA` pixel format
- Delegate frames via `captureOutput(_:didOutput:from:)` at 30 FPS
- **Privacy**: Process `CMSampleBuffer` in-memory only. Never write to disk.
- **ARC Memory Management**: Rely on Swift ARC for buffer lifecycle. Do NOT call `CMSampleBufferInvalidate()` — ensure no strong references or escaping closures capture the buffer, allowing ARC to release it automatically when `captureOutput` returns.

```swift
class PoseExtractor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let processingQueue = DispatchQueue(label: "pose.processing")
    
    /// Branch A callback: raw screen-relative landmarks for UI overlay.
    var onRawPoseDetected: (([String: SIMD4<Float>]) -> Void)?
    /// Branch B callback: world-space landmarks (meters) for GCN model.
    var onWorldPoseDetected: (([String: SIMD4<Float>]) -> Void)?
    
    func captureOutput(_ output: AVCaptureVideoDataOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // Extract pose from pixelBuffer via MediaPipe
        // ARC releases sampleBuffer when this scope exits — no manual invalidation
    }
}
```

---

#### Task 2.2: Skeleton Tracking (MediaPipe iOS SDK)

##### [NEW] ios/DPMonitor/Core/PoseExtractor.swift (continued)

Integrate **MediaPipe Tasks Vision iOS SDK** (`MediaPipeTasksVision`) for 33-joint extraction:

- Create `PoseLandmarker` with `modelComplexity: .full` (equivalent to `modelComplexity: 2`)
- Set `minPoseDetectionConfidence: 0.7`, `minTrackingConfidence: 0.7`
- **Two landmark outputs for two branches**:
  - **Branch A (UI overlay)**: Extract `NormalizedLandmark` array (x, y in [0,1] screen-relative + visibility). These are mapped to screen coordinates for the AR skeleton overlay.
  - **Branch B (GCN model)**: Extract **`poseWorldLandmarks`** (x, y, z in **meters**, physical world coordinates). The Python CTR-GCN model was trained on absolute spatial coordinates — using normalized screen coordinates would corrupt the Z-axis depth scaling and produce invalid model outputs.
- Convert both to `[String: SIMD4<Float>]` dicts matching `LANDMARK_NAMES` ordering

> [!IMPORTANT]
> **Data alignment**: `poseWorldLandmarks` provides coordinates in meters centered at the hip, which is already partially centered. The `PoseNormalizer` must still apply full hip-midpoint centering and spine-length scaling to match the Python pipeline's output exactly.

**Dependency**: Add `MediaPipeTasksVision` via Swift Package Manager.

---

### Task Group 3: Swift Data Preprocessing

#### Task 3.1: EMA Smoother (Branch B)

##### [NEW] ios/DPMonitor/Core/PoseSmoother.swift

Port of [poseSmoothing.js](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/frontend/src/lib/poseSmoothing.js):

```swift
struct PoseSmoother {
    let alpha: Float = 0.6
    let occlusionThreshold: Float = 0.5
    let occludedAlpha: Float = 0.15
    private var prev: [String: SIMD4<Float>]? = nil
    
    mutating func smooth(_ raw: [String: SIMD4<Float>]) -> [String: SIMD4<Float>] {
        guard let prev else { self.prev = raw; return raw }
        var out = [String: SIMD4<Float>]()
        for (name, rawJoint) in raw {
            let a: Float = rawJoint.w < occlusionThreshold ? occludedAlpha : alpha
            if let prevJoint = prev[name] {
                out[name] = mix(prevJoint, rawJoint, t: a)  // simd.mix
            } else {
                out[name] = rawJoint
            }
        }
        self.prev = out
        return out
    }
    
    mutating func reset() { prev = nil }
}
```

Uses `simd.mix()` for hardware-accelerated vector interpolation.

---

#### Task 3.2: Spatial Normalization

##### [NEW] ios/DPMonitor/Core/PoseNormalizer.swift

Port of [normalization.py](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/backend/analyzer/normalization.py):

```swift
import Accelerate

struct PoseNormalizer {
    static let leftHipIdx = 23   // MediaPipe index
    static let rightHipIdx = 24
    static let leftShoulderIdx = 11
    static let rightShoulderIdx = 12
    
    /// Normalize a (33, 3) pose: center to hip midpoint, scale by spine length.
    /// Uses vDSP for vectorized subtraction and division.
    static func normalize(_ frame: UnsafeMutablePointer<Float>,   // 33×3 flat array
                          count: Int = 33 * 3) {
        // 1. Compute hip midpoint
        // 2. vDSP_vsub to center all joints
        // 3. Compute shoulder midpoint on centered pose
        // 4. cblas_snrm2 for spine length
        // 5. vDSP_vsdiv to scale
    }
}
```

**Validation requirement**: Swift output must match Python NumPy output with error < 1e-4.

---

#### Task 3.3: Occlusion Carry-Forward

##### [NEW] ios/DPMonitor/Core/OcclusionHandler.swift

Port of `apply_occlusion_carryforward()` from [normalization.py](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/backend/analyzer/normalization.py):

```swift
struct OcclusionHandler {
    private var prevNormed: [Float]? = nil  // (33×3) flat
    
    mutating func apply(_ normed: inout [Float],
                        visibility: [Float],
                        threshold: Float = 0.5) {
        guard let prev = prevNormed else {
            prevNormed = normed
            return
        }
        for v in 0..<33 {
            if visibility[v] < threshold {
                normed[v*3]     = prev[v*3]
                normed[v*3 + 1] = prev[v*3 + 1]
                normed[v*3 + 2] = prev[v*3 + 2]
            }
        }
        prevNormed = normed
    }
    
    mutating func reset() { prevNormed = nil }
}
```

---

#### Task 3.4: Frame Buffer & Tensor Formatting

##### [NEW] ios/DPMonitor/Core/FrameBuffer.swift

```swift
class FrameBuffer {
    let windowSize: Int = 64
    private var buffer: [[Float]] = []  // each element is (33×3) flat
    
    func append(_ frame: [Float]) {
        buffer.append(frame)
        if buffer.count > windowSize { buffer.removeFirst() }
    }
    
    var isFull: Bool { buffer.count == windowSize }
    
    /// Build MLMultiArray of shape (1, 3, 64, 33, 1) from the buffer.
    /// Permutation: (T, V, C) → (C, T, V)
    ///
    /// Uses direct pointer memory binding instead of NSNumber subscript
    /// to avoid Objective-C bridging overhead (6,336 allocations per call).
    func toMLMultiArray() throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 3, 64, 33, 1]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Bind the raw data pointer as contiguous Float memory.
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: 6336)
        
        // Element strides: [6336, 2112, 33, 1, 1]
        for t in 0..<64 {
            let frame = buffer[t]
            for v in 0..<33 {
                for c in 0..<3 {
                    let srcIdx = v * 3 + c
                    let dstIdx = c * 2112 + t * 33 + v
                    pointer[dstIdx] = frame[srcIdx]  // Direct write, no NSNumber
                }
            }
        }
        return array
    }
}
```

---

### Task Group 4: Core ML Integration

#### Task 4.1: Action Classifier

##### [NEW] ios/DPMonitor/Core/ActionClassifier.swift

```swift
import CoreML

class ActionClassifier {
    private let model: CTRGCN  // Auto-generated from .mlpackage
    private let inferenceQueue = DispatchQueue(label: "coreml.inference", qos: .userInitiated)
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Prefer Neural Engine
        self.model = try CTRGCN(configuration: config)
    }
    
    /// Asynchronous inference — runs on a dedicated background queue to avoid
    /// blocking the camera delegate queue or main thread (which drives 120Hz AR).
    func classify(_ input: MLMultiArray,
                  completion: @escaping (Result<(qualityScore: Float, isCompensatory: Bool), Error>) -> Void) {
        inferenceQueue.async { [model] in
            do {
                let prediction = try model.prediction(input: input)
                let logits = prediction.logits  // MLMultiArray shape (1, num_class)
                // Softmax → quality score
                let good = exp(logits[0].floatValue)
                let bad = exp(logits[1].floatValue)
                let sum = good + bad
                let goodProb = good / sum
                DispatchQueue.main.async {
                    completion(.success((goodProb, bad > good)))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
}
```

---

#### Task 4.2: Session Orchestrator

##### [NEW] ios/DPMonitor/Core/SessionAnalyzer.swift

Port of [ctrgcn_analyzer.py](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/backend/analyzer/ctrgcn_analyzer.py). Orchestrates the full per-frame pipeline:

```
CMSampleBuffer (camera delegate queue)
    → PoseExtractor (33 landmarks — two outputs)
    │
    ├─ Branch A (raw/normalized screen coords) → main thread → SkeletonOverlayView (120Hz)
    │
    └─ Branch B (world landmarks in meters)
        → PoseSmoother → PoseNormalizer → OcclusionHandler → FrameBuffer
        → Every 5 frames (when buffer full):
            → inferenceQueue.async { ActionClassifier.classify() }
            → main thread ← quality + compensation result
        → RepCounter (hip y-excursion heuristic, synchronous)
        → Kinematics (ROM, tremor, synchronous)
        → main thread ← Update UI (HUD, gauges)
```

**Thread architecture**:
- `processingQueue` (serial): camera delegate → pose extraction → smoother → normalizer → buffer append
- `inferenceQueue` (serial): Core ML model.prediction() — never blocks the camera or main thread
- `DispatchQueue.main`: all UI updates (skeleton overlay, HUD, gauges)

Maintains:
- `qualityRunning` EMA: `0.7 * qualityRunning + 0.3 * inferredQuality`
- `framesSinceInference` counter with stride = 5
- `isInferenceInFlight` flag to prevent overlapping inference calls
- Session log for post-session synthesis

---

### Task Group 5: UI/UX

#### Task 5.1: Camera Preview + AR Overlay

##### [NEW] ios/DPMonitor/Views/CameraView.swift

`UIViewRepresentable` wrapping `AVCaptureVideoPreviewLayer`. Full-screen camera feed.

##### [NEW] ios/DPMonitor/Views/SkeletonOverlayView.swift

`CALayer`-based skeleton rendering at display refresh rate (120Hz on iPhone 16 Pro Max):
- Draw joints as filled circles, bones as line segments
- Use `CADisplayLink` for frame timing
- Color coding: green (good), amber (caution), red (compensatory)
- **Occlusion**: Skip joints/bones with `visibility < 0.5`
- Branch A raw coordinates mapped to screen: `x * viewWidth`, `y * viewHeight`

##### [NEW] ios/DPMonitor/Views/HUDView.swift

SwiftUI overlay showing rep count, quality score gauge, and real-time feedback messages.

##### [NEW] ios/DPMonitor/Views/SessionView.swift

The main live monitor screen combining:
- Camera preview (full screen background)
- Skeleton overlay
- HUD overlay
- Start/Stop session controls
- Exercise type selector

---

#### Task 5.2: Results & History

##### [NEW] ios/DPMonitor/Views/ResultsView.swift

Post-session summary screen (port of `synthesize()` from [synthesis.py](file:///Users/houxiqing/Documents/USYD/ProfJinman/DPMonitor/backend/analyzer/synthesis.py)):
- Rep count, overall quality score, ROM chart
- Per-rep breakdown with stability trend
- Fatigue index

##### [NEW] ios/DPMonitor/Views/HistoryView.swift

List of past sessions stored in CoreData. Each row shows date, exercise type, rep count, quality score.

---

### Task Group 6: Local Storage

#### Task 6.1: CoreData Schema

##### [NEW] ios/DPMonitor/Storage/DPMonitor.xcdatamodeld

Entities:
- **Session**: `id` (UUID), `date` (Date), `exerciseType` (String), `repCount` (Int16), `qualityScore` (Float), `stabilityScore` (Float), `compensationEvents` (Int16), `durationSeconds` (Float)
- **SessionDetail**: `sessionId` (UUID), `summaryJSON` (String) — stores the full synthesis output for the results screen

##### [NEW] ios/DPMonitor/Storage/SessionStore.swift

CoreData manager with:
- `saveSession(_ summary: SessionSummary)`
- `fetchHistory() -> [Session]`
- `deleteSession(_ id: UUID)`

---

## Summary of All New Files

| Group | File | Purpose |
|---|---|---|
| **Export** | `scripts/export_coreml.py` | PyTorch → ONNX → Core ML conversion |
| **Model Fix** | `ctrgcn/ctrgcn.py` (modify) | Replace `einsum` with `matmul` |
| **App Entry** | `ios/.../DPMonitorApp.swift` | SwiftUI app entry |
| **App Entry** | `ios/.../ContentView.swift` | Root tab navigation |
| **Core** | `ios/.../PoseExtractor.swift` | AVFoundation + MediaPipe bridge |
| **Core** | `ios/.../PoseSmoother.swift` | EMA filter (Branch B) |
| **Core** | `ios/.../PoseNormalizer.swift` | Hip-center + spine-scale (Accelerate) |
| **Core** | `ios/.../OcclusionHandler.swift` | Visibility carry-forward |
| **Core** | `ios/.../FrameBuffer.swift` | 64-frame sliding window + MLMultiArray |
| **Core** | `ios/.../ActionClassifier.swift` | Core ML inference wrapper |
| **Core** | `ios/.../RepCounter.swift` | Hip y-excursion rep heuristic |
| **Core** | `ios/.../Kinematics.swift` | Joint angles, ROM, tremor (Accelerate) |
| **Core** | `ios/.../SessionAnalyzer.swift` | Full pipeline orchestrator |
| **Views** | `ios/.../CameraView.swift` | AVCaptureVideoPreviewLayer |
| **Views** | `ios/.../SkeletonOverlayView.swift` | CALayer AR skeleton overlay |
| **Views** | `ios/.../HUDView.swift` | Rep counter + quality gauge |
| **Views** | `ios/.../SessionView.swift` | Live monitor screen |
| **Views** | `ios/.../ResultsView.swift` | Post-session summary |
| **Views** | `ios/.../HistoryView.swift` | Session history list |
| **Storage** | `ios/.../SessionStore.swift` | CoreData manager |
| **Storage** | `ios/.../DPMonitor.xcdatamodeld` | CoreData schema |
| **Tests** | `ios/.../PoseNormalizerTests.swift` | Python parity validation |
| **Tests** | `ios/.../KinematicsTests.swift` | Angle/ROM/tremor parity |
| **Model** | `ios/.../CTRGCN.mlpackage` | Bundled Core ML model |

---

## Verification Plan

### Automated Tests

**Python (existing)**:
```bash
cd backend && .venv/bin/python manage.py test
```
Must pass after `einsum` replacement in `ctrgcn.py`.

**Swift (new)**:
- `PoseNormalizerTests`: Feed known input arrays, compare output to Python NumPy reference with tolerance < 1e-4
- `KinematicsTests`: Verify `joint_angle`, `range_of_motion`, `tremor_metrics` against Python reference values

### Core ML Parity Test
```bash
python scripts/export_coreml.py --validate
```
Runs identical dummy input through both PyTorch and Core ML models, asserts output difference < 1e-3.

### Manual Verification

**1. Airplane Mode Test**:
- Enable Airplane Mode → Install app → Open → Capture movement → Run full session → View results
- All features must work with zero network connectivity

**2. Thermal Test**:
- Run continuously for 15 minutes on iPhone 16 Pro Max
- Monitor thermal state via `ProcessInfo.processInfo.thermalState`
- Must not reach `.critical` or trigger screen dimming

**3. Accuracy Parity**:
- Record a reference exercise video → Process on both web app and iOS app
- Compare quality scores, rep counts, and ROM values between platforms
