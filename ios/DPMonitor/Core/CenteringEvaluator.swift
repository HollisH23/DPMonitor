//
//  CenteringEvaluator.swift
//  DPMonitor
//
//  Pre-session framing assistant. Swift port of
//  `Final/centering_logic.py :: evaluate_centering`, via the shared
//  reference at `backend/analyzer/centering.py`.
//
//  Why framing gets its own check
//  ------------------------------
//  Every downstream number degrades silently when framing is bad. A
//  patient at the edge of frame has landmarks clamped at the image
//  boundary, which reads to the GCN as a held pose. Clipped feet mean the
//  knee and ankle are extrapolated, so the ROM a physio reads is fiction.
//  Standing too far away shrinks the skeleton until landmark noise is a
//  meaningful fraction of the spine length — and normalisation then
//  amplifies exactly that noise. None of this throws. It just produces
//  confident, wrong numbers.
//
//  Coordinate space
//  ----------------
//  Screen-space (Branch A) landmarks: x/y normalised to [0, 1], y growing
//  downward. Deliberately NOT the world landmarks — those are hip-centred
//  and carry no information about where in the frame the patient stands.
//
//  Handedness
//  ----------
//  Messages are frame-relative, exactly as in the reference: "Patient too
//  far LEFT" means their body is toward x = 0, not that they should move
//  left. That is mirroring-agnostic, which is why there is no `mirrored`
//  flag: the overlay's arrow is drawn in screen space next to the
//  patient's own on-screen image, so it reads correctly whether or not
//  the preview is mirrored.
//
//  Thresholds and message strings are pinned against the Python by
//  `CenteringEvaluatorTests`, which loads the generated fixtures.
//
//  Precision caveat
//  ----------------
//  Landmarks arrive as `Float`; the Python reference computes in float64.
//  For a threshold like 0.30 the two representations differ by ~1e-8, so
//  a patient standing at *exactly* the boundary can land on opposite
//  branches on the two platforms — (0.21 + 0.39) / 2 is 0.30000000000000004
//  in float64 but 0.29999999 in Float. This is inherent, not a defect: no
//  amount of care in this file makes a float32 pipeline agree with a
//  float64 one at a discontinuity.
//
//  The fixtures therefore straddle every threshold at ±0.002 rather than
//  sitting on it. Do not "tighten" them back to exact boundary values —
//  that asserts a property floating point cannot provide. Exact-threshold
//  semantics (inclusive vs exclusive) are pinned in Python only, by
//  `CenteringTests`, where float64 exactness is meaningful.
//

import Foundation
import simd

// MARK: - Status

/// Stable token for the primary finding. Mirrors the Python `status_code`.
///
/// `moveLeft` / `moveRight` name the *correction* the patient should make
/// in frame space, while the `message` describes their current position.
/// So a patient standing at x = 0.15 gets `.moveRight` alongside
/// "Patient too far LEFT" — the arrow points the way out, the text says
/// what is wrong.
enum CenteringStatus: String {
    case centered
    case moveLeft = "move_left"
    case moveRight = "move_right"
    case tooClose = "too_close"
    case tooFar = "too_far"
    case headClipped = "head_clipped"
    case feetClipped = "feet_clipped"
    /// Not in the desktop reference, which indexes the landmark list
    /// directly and lets its caller handle the no-pose case. Folded in
    /// here so the iOS call site needs only one code path.
    case notDetected = "not_detected"
}

/// How loudly the overlay should present the finding. Replaces the
/// reference's BGR tuple, which is meaningless on iOS.
enum CenteringSeverity: String {
    case ok
    case warning
    case critical
}

// MARK: - Result

struct CenteringResult: Equatable {

    /// Human-readable primary message (the reference's `status`).
    var message: String
    var status: CenteringStatus
    var isCentered: Bool
    var severity: CenteringSeverity
    /// Every finding, in reference order. For a centred patient this is
    /// the two informational lines instead.
    var details: [String]

    var hipCenterX: Float?
    var shoulderCenterX: Float?
    var torsoHeightRatio: Float?

    /// The "no pose" state, matching the desktop caller's wording.
    init() {
        message = "Patient has left the field of view"
        status = .notDetected
        isCentered = false
        severity = .critical
        details = ["Pose not recognised."]
        hipCenterX = nil
        shoulderCenterX = nil
        torsoHeightRatio = nil
    }

    init(message: String,
         status: CenteringStatus,
         isCentered: Bool,
         severity: CenteringSeverity,
         details: [String],
         hipCenterX: Float?,
         shoulderCenterX: Float?,
         torsoHeightRatio: Float?) {
        self.message = message
        self.status = status
        self.isCentered = isCentered
        self.severity = severity
        self.details = details
        self.hipCenterX = hipCenterX
        self.shoulderCenterX = shoulderCenterX
        self.torsoHeightRatio = torsoHeightRatio
    }
}

// MARK: - Evaluator

enum CenteringEvaluator {

    // Thresholds, verbatim from the reference.
    static let hipXMin: Float = 0.30
    static let hipXMax: Float = 0.70
    static let shoulderXMin: Float = 0.25
    static let shoulderXMax: Float = 0.75
    static let headClipY: Float = 0.03
    /// The reference gates the head check on nose visibility < 0.3. This
    /// is NOT the 0.5 occlusion threshold used elsewhere in the pipeline;
    /// head framing is a coarser judgement than joint carry-forward.
    static let noseMinVisibility: Float = 0.3
    static let kneeClipY: Float = 0.97
    static let torsoRatioMin: Float = 0.12
    static let torsoRatioMax: Float = 0.70

    /// Landmarks the reference indexes unconditionally. Any one missing
    /// means we cannot evaluate.
    static let requiredLandmarks = [
        "nose",
        "left_hip", "right_hip",
        "left_shoulder", "right_shoulder",
        "left_knee", "right_knee",
    ]

    /// Evaluate framing from screen-space landmarks.
    ///
    /// - Parameter points: `[name: SIMD4(x, y, z, visibility)]`, the
    ///   Branch A dictionary `PoseExtractor` already emits.
    static func evaluate(_ points: [String: SIMD4<Float>]) -> CenteringResult {
        guard !points.isEmpty else { return CenteringResult() }
        for name in requiredLandmarks where points[name] == nil {
            return CenteringResult()
        }

        @inline(__always) func x(_ n: String) -> Float { points[n]!.x }
        @inline(__always) func y(_ n: String) -> Float { points[n]!.y }
        @inline(__always) func vis(_ n: String) -> Float { points[n]!.w }

        // (message, severity, status) in the reference's evaluation order.
        // Order matters: `issues[0]` becomes the headline.
        var issues: [(String, CenteringSeverity, CenteringStatus)] = []

        // --- Horizontal centering (hip midpoint) ----------------------
        let hipCX = (x("left_hip") + x("right_hip")) / 2.0
        if hipCX < hipXMin {
            issues.append(("Patient too far LEFT", .warning, .moveRight))
        } else if hipCX > hipXMax {
            issues.append(("Patient too far RIGHT", .warning, .moveLeft))
        }

        // --- Shoulder centering ---------------------------------------
        let shoulderCX = (x("left_shoulder") + x("right_shoulder")) / 2.0
        if shoulderCX < shoulderXMin {
            issues.append(("Shoulders shifted LEFT", .warning, .moveRight))
        } else if shoulderCX > shoulderXMax {
            issues.append(("Shoulders shifted RIGHT", .warning, .moveLeft))
        }

        // --- Head clipping ---------------------------------------------
        if y("nose") < headClipY || vis("nose") < noseMinVisibility {
            issues.append(("Patient HEAD may be cut off", .critical, .headClipped))
        }

        // --- Feet / knees clipping -------------------------------------
        if max(y("left_knee"), y("right_knee")) > kneeClipY {
            issues.append(("Patient FEET may be cut off", .critical, .feetClipped))
        }

        // --- Too close / too far ---------------------------------------
        // NOTE: nose-to-hip, NOT shoulder-to-hip. Measuring from the
        // shoulder would roughly halve the ratio and silently invalidate
        // both thresholds below.
        let torso = abs((y("left_hip") + y("right_hip")) / 2.0 - y("nose"))
        if torso < torsoRatioMin {
            issues.append(("Patient is TOO FAR from camera", .warning, .tooFar))
        } else if torso > torsoRatioMax {
            issues.append(("Patient is TOO CLOSE to camera", .warning, .tooClose))
        }

        let hip = round4(hipCX)
        let shoulder = round4(shoulderCX)
        let ratio = round4(torso)

        guard let primary = issues.first else {
            return CenteringResult(
                message: "Patient is CENTERED",
                status: .centered,
                isCentered: true,
                severity: .ok,
                details: [
                    "Hip center: \(percent(hipCX)) (ideal ~50%)",
                    "Shoulder center: \(percent(shoulderCX))",
                ],
                hipCenterX: hip,
                shoulderCenterX: shoulder,
                torsoHeightRatio: ratio)
        }

        var details = issues.map(\.0)
        details.append("Hip center: \(percent(hipCX))")

        return CenteringResult(
            message: primary.0,
            status: primary.2,
            isCentered: false,
            severity: primary.1,
            details: details,
            hipCenterX: hip,
            shoulderCenterX: shoulder,
            torsoHeightRatio: ratio)
    }

    // MARK: Formatting

    /// Reproduces Python's `f"{value:.0%}"`.
    ///
    /// Python rounds the scaled value half-to-even; Swift's `rounded()`
    /// rounds half-away-from-zero. They differ only when `value * 100`
    /// lands exactly on .5, which the fixture inputs deliberately avoid.
    static func percent(_ value: Float) -> String {
        "\(Int((Double(value) * 100).rounded()))%"
    }

    /// Matches Python's `round(x, 4)` closely enough for the 1e-4 parity
    /// tolerance the fixtures assert.
    @inline(__always)
    static func round4(_ value: Float) -> Float {
        (value * 10_000).rounded() / 10_000
    }
}
