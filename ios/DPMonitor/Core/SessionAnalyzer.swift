//
//  SessionAnalyzer.swift
//  DPMonitor
//
//  Pipeline orchestrator. Port of backend/analyzer/ctrgcn_analyzer.py.
//
//  Per-frame flow
//  --------------
//    CMSampleBuffer  (camera delegate queue)
//      -> PoseExtractor (33 landmarks, two branches)
//         |
//         +- Branch A: screen-relative
//         |     -> RepCounter (hip-y hysteresis)
//         |     -> main thread -> SkeletonOverlayView (display-linked)
//         |
//         +- Branch B: world landmarks (metres)
//               -> PoseSmoother -> PoseNormalizer -> OcclusionHandler
//               -> FrameBuffer (64-frame window)
//               -> every `inferenceStride` frames, once full:
//                    inferenceQueue.async { ActionClassifier.classify() }
//               -> Kinematics side-band (ROM, tremor)
//               -> main thread -> HUD
//
//  Thread architecture
//  -------------------
//    pipelineQueue  (serial) — owns `PipelineCore` exclusively
//    inferenceQueue (serial) — owned by ActionClassifier; never blocks the
//                              camera or the main thread
//    main           — `SessionAnalyzer` itself: every @Published mutation
//
//  The split into two types is deliberate. `SessionAnalyzer` is
//  `@MainActor` because SwiftUI requires it; `PipelineCore` is not, because
//  it lives entirely on `pipelineQueue`. Keeping the mutable numeric state
//  inside a non-isolated type makes the ownership boundary explicit instead
//  of relying on every call site remembering to hop queues.
//
//  `isInferenceInFlight` prevents overlapping forward passes. Without it a
//  slow pass under thermal throttling would queue work faster than it
//  drains and the UI would fall further behind every second. It also makes
//  FrameBuffer's reusable scratch tensor safe: only one prediction can be
//  reading it at a time.
//

import Combine
import CoreML
import Foundation
import simd
import os

private let sessionLogger = Logger(subsystem: "au.edu.usyd.dpmonitor", category: "session")

// MARK: - Exercise types

enum ExerciseType: String, CaseIterable, Identifiable {
    case squat
    case lunge
    case shoulderRaise = "shoulder_raise"
    case kneeExtension = "knee_extension"
    case chestExpansion = "chest_expansion"
    case custom

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .squat:          return "Squat"
        case .lunge:          return "Lunge"
        case .shoulderRaise:  return "Shoulder Raise"
        case .kneeExtension:  return "Knee Extension"
        case .chestExpansion: return "Chest Expansion"
        case .custom:         return "Custom"
        }
    }
}

// MARK: - Live snapshot

/// Everything the HUD needs, in one value type so the UI takes a single
/// dependency instead of nine `@Published` properties.
struct LiveMetrics {
    var repCount: Int = 0
    var qualityScore: Double = 1.0
    var isCompensatory: Bool = false
    var feedback: [String] = []
    /// 0–1 warm-up progress toward a full 64-frame window.
    var windowFill: Double = 0
    var inferenceCalls: Int = 0
    var similarityScore: Double?
    var elapsedSeconds: Double = 0
    var jointSummaries: [Kinematics.JointSummary] = []
    /// False when the bundled model has seeded-random weights, in which case
    /// `qualityScore` and `isCompensatory` are not meaningful and the UI
    /// must not present them as clinical findings.
    var qualityIsCalibrated: Bool = true
}

/// One frame's worth of results, handed from the pipeline to the UI.
private struct FrameOutcome {
    var quality: Double
    var windowFill: Double
    var inferenceCalls: Int
    var similarity: Double?
    var jointSummaries: [Kinematics.JointSummary]
    var feedback: [String]
    var heuristicCompensation: Bool
}

// MARK: - PipelineCore

/// All mutable numeric state. Confined to `SessionAnalyzer.pipelineQueue`;
/// nothing outside that queue may touch an instance of this type.
private final class PipelineCore {

    let windowSize: Int
    var inferenceStride: Int

    var smoother = PoseSmoother()
    var occlusion = OcclusionHandler()
    let frameBuffer: FrameBuffer
    var repCounter = RepCounter()

    /// Un-normalised world frames, for the kinematics side-band.
    var rawWindow: [[Float]] = []

    var qualityRunning: Double = 1.0
    var qualitySamples: [Double] = []
    var compensationEvents = 0
    var framesSinceInference = 0
    var inferenceCalls = 0
    var isInferenceInFlight = false
    var latestSimilarity: Double?
    var sessionLogEntries: [SessionLogEntry] = []
    var frameIndex = 0
    var startedAt: Date?

    /// True only between `start()` and `stop()`.
    ///
    /// The camera and pose extraction run from the moment the screen
    /// appears, so the centering assistant can guide the patient before
    /// anything is recorded. This flag is what separates "frames are
    /// flowing" from "frames count towards a session" — without it the
    /// rep counter would tick up while the patient is still walking into
    /// position, and the 64-frame window would fill with them arriving.
    var isRecording = false

    init(windowSize: Int, inferenceStride: Int) {
        self.windowSize = windowSize
        self.inferenceStride = inferenceStride
        self.frameBuffer = FrameBuffer(windowSize: windowSize)
    }

    /// Mirrors `CTRGCNAnalyzer.reset`.
    func reset() {
        smoother.reset()
        occlusion.reset()
        frameBuffer.reset()
        repCounter.reset()
        rawWindow.removeAll(keepingCapacity: true)
        qualityRunning = 1.0
        qualitySamples.removeAll(keepingCapacity: true)
        compensationEvents = 0
        framesSinceInference = 0
        inferenceCalls = 0
        isInferenceInFlight = false
        latestSimilarity = nil
        sessionLogEntries.removeAll(keepingCapacity: true)
        frameIndex = 0
    }

    // MARK: Branch B ingestion

    /// Runs smoothing, normalisation, carry-forward, buffering and the
    /// per-frame angle snapshot. Returns the values the UI should show.
    func ingestWorldPose(_ points: [String: SIMD4<Float>],
                         timestampMs: Double) -> FrameOutcome {
        frameIndex += 1

        // 1) EMA smoothing on the dictionary, exactly as the web app does.
        let smoothed = smoother.smooth(points)

        // 2) Flatten to the ordered (V, C) layout + visibility.
        let frame = PoseFrame(dictionary: smoothed)

        // Keep the un-normalised window for the kinematics side-band.
        rawWindow.append(frame.coords)
        if rawWindow.count > windowSize {
            rawWindow.removeFirst(rawWindow.count - windowSize)
        }

        // 3) Normalise, then carry occluded joints forward. Order matters:
        //    carry-forward must operate in the normalised frame, otherwise a
        //    carried value would be re-centred against a different hip
        //    midpoint on the next frame and drift.
        var normed = frame.coords
        PoseNormalizer.normalize(&normed)
        occlusion.apply(&normed, visibility: frame.visibility)

        // 4) Slide the window.
        frameBuffer.append(normed)
        framesSinceInference += 1

        // 5) Per-frame angle snapshot for the session log.
        let angles = Kinematics.frameAngles(frame.coords)
        sessionLogEntries.append(SessionLogEntry(frameIndex: frameIndex,
                                                 timestampMs: timestampMs,
                                                 similarity: latestSimilarity,
                                                 angles: angles))

        // 6) Cheap geometric form hints, independent of the model.
        var feedback: [String] = []
        var compensatory = false
        for triplet in PoseLandmarks.angleTriplets {
            guard let angle = angles[triplet.name] else { continue }
            if triplet.name.contains("knee") && angle < 70 {
                if !feedback.contains("Knees collapsing inward") {
                    feedback.append("Knees collapsing inward")
                }
                compensatory = true
            } else if triplet.name.contains("hip") && angle < 150 {
                // The Python checks a "back" angle the web client supplies;
                // the closest quantity we compute on-device is the hip angle.
                if !feedback.contains("Keep your chest up") {
                    feedback.append("Keep your chest up")
                }
                compensatory = true
            }
        }

        qualitySamples.append(qualityRunning)

        return FrameOutcome(
            quality: qualityRunning,
            windowFill: min(1.0, Double(frameBuffer.count) / Double(windowSize)),
            inferenceCalls: inferenceCalls,
            similarity: latestSimilarity,
            jointSummaries: frameBuffer.isFull ? Kinematics.windowSummary(rawWindow) : [],
            feedback: feedback,
            heuristicCompensation: compensatory
        )
    }

    /// True when this frame should trigger a forward pass.
    func shouldRunInference(modelAvailable: Bool) -> Bool {
        guard modelAvailable, frameBuffer.isFull, !isInferenceInFlight else { return false }
        guard framesSinceInference >= inferenceStride else { return false }
        framesSinceInference = 0
        return true
    }

    /// Fold a completed forward pass into the running state.
    ///
    /// `calibrated` gates the quality signal rather than the inference: we
    /// still run the forward pass (it exercises the whole Core ML path, and
    /// the feature embedding is still usable for similarity), but an
    /// untrained model's saturated softmax must not drive the score or the
    /// compensation counter. Letting it through would pin the gauge at 0%
    /// and raise a form-drift alert on literally every window.
    func absorb(_ result: ClassificationResult, calibrated: Bool) {
        inferenceCalls += 1
        latestSimilarity = result.similarityScore
        guard calibrated else { return }
        // EMA toward the model estimate, suppressing per-pass jitter.
        qualityRunning = 0.7 * qualityRunning + 0.3 * Double(result.qualityScore)
        if result.isCompensatory { compensationEvents += 1 }
    }

    // MARK: Summary

    func makeSummary(exercise: ExerciseType,
                     duration: TimeInterval,
                     calibrated: Bool) -> SessionSummary {
        let quality = qualitySamples.isEmpty
            ? 0.0
            : qualitySamples.reduce(0, +) / Double(qualitySamples.count)

        return SessionSummary(
            id: UUID(),
            date: startedAt ?? Date(),
            exerciseType: exercise,
            repCount: repCounter.count,
            qualityScore: quality,
            stabilityScore: repCounter.stabilityScore,
            compensationEvents: compensationEvents,
            durationSeconds: duration,
            framesAnalyzed: frameIndex,
            inferenceCalls: inferenceCalls,
            windowSize: windowSize,
            inferenceStride: inferenceStride,
            finalWindowKinematics: Kinematics.windowSummary(rawWindow).map {
                KinematicsRecord(joint: $0.joint,
                                 minDeg: $0.rom.minDeg,
                                 maxDeg: $0.rom.maxDeg,
                                 rangeDeg: $0.rom.rangeDeg,
                                 velocityRMS: $0.tremor.velocityRMS,
                                 accelerationRMS: $0.tremor.accelerationRMS)
            },
            qualityIsCalibrated: calibrated,
            synthesis: Synthesis.synthesize(sessionLogEntries, qualitySamples: qualitySamples)
        )
    }
}

// MARK: - SessionAnalyzer

@MainActor
final class SessionAnalyzer: ObservableObject {

    // MARK: Published state

    @Published private(set) var metrics = LiveMetrics()
    /// Branch A landmarks for the overlay, screen-normalised.
    @Published private(set) var overlayPose: [String: SIMD4<Float>] = [:]
    @Published private(set) var isRunning = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var modelIsAvailable = false
    /// False when the bundled model was exported from seeded-random weights.
    @Published private(set) var modelIsCalibrated = false

    /// Live framing assessment, evaluated from the Branch A screen-space
    /// landmarks. Published on every pose so the pre-session overlay can
    /// track the patient in real time.
    @Published private(set) var centeringResult = CenteringResult()

    /// Whether the centering overlay should keep running once a session
    /// starts. Off by default: during a session the guide lines compete
    /// with the skeleton and HUD. Evaluation itself is cheap either way.
    @Published var showCenteringDuringSession = false

    // MARK: Configuration

    let windowSize: Int
    /// Frames between forward passes. Widened automatically under thermal
    /// pressure — see `applyThermalState`.
    private(set) var inferenceStride: Int
    private let baseInferenceStride: Int

    var exerciseType: ExerciseType = .custom

    // MARK: Collaborators

    let extractor: PoseExtractor
    private let classifier = ActionClassifier()

    private let pipelineQueue = DispatchQueue(
        label: "au.edu.usyd.dpmonitor.pipeline", qos: .userInitiated)
    private let core: PipelineCore

    // MARK: Init

    init(windowSize: Int = 64, inferenceStride: Int = 5) {
        let stride = max(1, inferenceStride)
        self.windowSize = windowSize
        self.inferenceStride = stride
        self.baseInferenceStride = stride
        self.core = PipelineCore(windowSize: windowSize, inferenceStride: stride)
        self.extractor = PoseExtractor()

        modelIsAvailable = classifier.modelIsAvailable
        modelIsCalibrated = classifier.isCalibrated
        metrics.qualityIsCalibrated = classifier.isCalibrated
        wireExtractor()
    }

    private func wireExtractor() {
        let queue = pipelineQueue
        let core = self.core

        // Branch A — screen space. Arrives on the extractor's processing
        // queue; hop to `pipelineQueue` so rep state stays single-owner.
        extractor.onRawPoseDetected = { [weak self] points in
            queue.async {
                // Centering runs whether or not a session is recording —
                // guiding the patient into frame is the whole point of
                // doing it before they press start.
                let centering = CenteringEvaluator.evaluate(points)

                var repCompleted = false
                if core.isRecording, let hipY = RepCounter.hipY(points) {
                    repCompleted = core.repCounter.update(hipY: hipY)
                }
                let count = core.repCounter.count

                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.overlayPose = points
                    // CenteringResult is Equatable, so an unchanged
                    // assessment costs no SwiftUI invalidation.
                    if self.centeringResult != centering {
                        self.centeringResult = centering
                    }
                    guard core.isRecording else { return }
                    self.metrics.repCount = count
                    if repCompleted { self.pushFeedback("Rep counted") }
                }
            }
        }

        // Branch B — world space, metres.
        extractor.onWorldPoseDetected = { [weak self] points, timestampMs in
            queue.async {
                // Nothing accumulates until the session actually starts.
                guard core.isRecording else { return }
                let outcome = core.ingestWorldPose(points, timestampMs: timestampMs)
                if core.shouldRunInference(modelAvailable: self?.classifierIsAvailable ?? false) {
                    self?.runInference()
                }
                Task { @MainActor [weak self] in
                    self?.apply(outcome)
                }
            }
        }

        extractor.onError = { [weak self] error in
            Task { @MainActor [weak self] in
                self?.errorMessage = error.localizedDescription
            }
        }
    }

    /// Readable from any thread: `ActionClassifier.modelIsAvailable` only
    /// inspects an immutable `let`.
    nonisolated private var classifierIsAvailable: Bool { classifier.modelIsAvailable }

    // MARK: Lifecycle

    /// Configure the camera and begin streaming poses.
    ///
    /// Deliberately starts capture immediately rather than waiting for
    /// `start()`: the centering assistant has to see the patient in order
    /// to guide them into position, and a black preview behind a "stand
    /// in the middle" prompt would be useless.
    func prepare() async {
        do {
            try await extractor.prepare()
            errorMessage = nil
            extractor.start()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Begin recording. Capture is already running by this point.
    func start() {
        guard !isRunning else { return }
        metrics = LiveMetrics()
        metrics.qualityIsCalibrated = modelIsCalibrated
        let startedAt = Date()
        sessionStart = startedAt
        pipelineQueue.async { [core] in
            core.reset()
            core.startedAt = startedAt
            core.isRecording = true
        }
        isRunning = true
        extractor.start()   // no-op if already running
    }

    /// Stop recording but leave the camera live, so the patient can see
    /// themselves and re-centre before the next set.
    func stop() {
        guard isRunning else { return }
        pipelineQueue.async { [core] in core.isRecording = false }
        isRunning = false
    }

    /// Tear down capture entirely. Call when the screen goes away.
    func stopCamera() {
        pipelineQueue.async { [core] in core.isRecording = false }
        isRunning = false
        extractor.stop()
    }

    /// Widen the inference stride once the device reports thermal pressure.
    ///
    /// Halving the forward-pass rate is a far better failure mode than
    /// letting iOS throttle the whole app: the skeleton overlay and rep
    /// count stay at full rate, only the quality score updates more slowly.
    func applyThermalState(_ state: ProcessInfo.ThermalState) {
        let stride: Int
        switch state {
        case .nominal, .fair: stride = baseInferenceStride
        case .serious:        stride = baseInferenceStride * 2
        case .critical:       stride = baseInferenceStride * 4
        @unknown default:     stride = baseInferenceStride
        }
        guard stride != inferenceStride else { return }
        inferenceStride = stride
        sessionLogger.notice("thermal state changed; inference stride now \(stride)")
        pipelineQueue.async { [core] in core.inferenceStride = stride }
    }

    // MARK: Frame results

    private func apply(_ outcome: FrameOutcome) {
        metrics.qualityScore = outcome.quality
        metrics.windowFill = outcome.windowFill
        metrics.inferenceCalls = outcome.inferenceCalls
        metrics.similarityScore = outcome.similarity
        if !outcome.jointSummaries.isEmpty {
            metrics.jointSummaries = outcome.jointSummaries
        }
        if let started = sessionStart {
            metrics.elapsedSeconds = Date().timeIntervalSince(started)
        }
        if outcome.heuristicCompensation { metrics.isCompensatory = true }
        for message in outcome.feedback { pushFeedback(message) }
    }

    /// Main-actor mirror of `core.startedAt`, so the HUD's elapsed-time
    /// readout does not need a queue hop on every frame.
    private var sessionStart: Date?

    // MARK: Inference

    /// Called on `pipelineQueue` — `shouldRunInference` has already claimed
    /// the in-flight slot by resetting `framesSinceInference`.
    nonisolated private func runInference() {
        let core = self.core
        let input: MLMultiArray
        do {
            input = try core.frameBuffer.makeInputTensor()
        } catch {
            sessionLogger.error("tensor packing failed: \(error.localizedDescription)")
            return
        }

        core.isInferenceInFlight = true
        classifier.classify(input) { [weak self] result in
            // ActionClassifier completes on main; hop back to the pipeline
            // queue before touching any pipeline-confined state.
            guard let self else { return }
            self.pipelineQueue.async {
                core.isInferenceInFlight = false
                switch result {
                case .success(let r):
                    let calibrated = self.classifier.isCalibrated
                    core.absorb(r, calibrated: calibrated)
                    guard calibrated else { break }
                    let compensatory = r.isCompensatory
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.metrics.isCompensatory = compensatory
                        if compensatory {
                            self.pushFeedback("Form drift detected — slow down and reset")
                        }
                    }
                case .failure(let error):
                    sessionLogger.error("inference failed: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: Feedback ticker

    /// Keeps the three most recent distinct messages. Re-posting the same
    /// message refreshes it rather than stacking duplicates.
    private func pushFeedback(_ message: String) {
        var list = metrics.feedback
        list.removeAll { $0 == message }
        list.insert(message, at: 0)
        if list.count > 3 { list.removeLast(list.count - 3) }
        metrics.feedback = list
    }

    // MARK: Summary

    /// Build the end-of-session summary. Mirrors `generate_summary` plus
    /// `synthesize`. Async because the log lives on the pipeline queue.
    func makeSummary() async -> SessionSummary {
        let exercise = exerciseType
        let duration = metrics.elapsedSeconds
        let calibrated = modelIsCalibrated
        let core = self.core

        return await withCheckedContinuation { continuation in
            pipelineQueue.async {
                let elapsed = core.startedAt.map { Date().timeIntervalSince($0) } ?? duration
                continuation.resume(
                    returning: core.makeSummary(exercise: exercise,
                                                duration: elapsed,
                                                calibrated: calibrated))
            }
        }
    }
}
