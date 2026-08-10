//
//  PoseExtractor.swift
//  DPMonitor
//
//  AVFoundation capture + MediaPipe Tasks Vision pose landmarking.
//
//  Two branches, deliberately kept separate (mirrors the web app):
//
//    Branch A — `onRawPoseDetected`
//        Screen-normalised landmarks (x, y in [0, 1]) straight from
//        MediaPipe, unsmoothed. Drives the AR skeleton overlay, where
//        pixel-exact registration matters more than temporal stability.
//
//    Branch B — `onWorldPoseDetected`
//        World landmarks in METRES, hip-origin. Feeds the EMA smoother ->
//        normaliser -> CTR-GCN. The model was configured for absolute
//        spatial coordinates; screen-normalised x/y with an unrelated z
//        scale would corrupt the depth channel.
//
//  Privacy: `CMSampleBuffer`s are consumed in-memory and never written to
//  disk. Nothing here touches the filesystem or the network.
//

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import UIKit
import simd
import os

#if canImport(MediaPipeTasksVision)
import MediaPipeTasksVision
#endif

private let poseLog = Logger(subsystem: "au.edu.usyd.dpmonitor", category: "pose")

// MARK: - Configuration

struct PoseExtractorConfig {
    /// Bundled BlazePose bundle. `full` is the Tasks-API equivalent of the
    /// legacy Solutions `modelComplexity: 2` used by the web app.
    /// Swap to `pose_landmarker_lite` if the thermal budget demands it.
    var modelName: String = "pose_landmarker_full"
    var modelExtension: String = "task"

    var minPoseDetectionConfidence: Float = 0.7
    var minPosePresenceConfidence: Float = 0.7
    var minTrackingConfidence: Float = 0.7
    var numPoses: Int = 1

    /// Front camera reads more naturally for self-guided rehab; the session
    /// screen exposes a flip control.
    var cameraPosition: AVCaptureDevice.Position = .front

    /// 720p at 30 FPS is the sweet spot: BlazePose downsamples to 256×256
    /// internally, so 1080p buys nothing but heat.
    var sessionPreset: AVCaptureSession.Preset = .hd1280x720
    var targetFrameRate: Int32 = 30
}

// MARK: - Errors

enum PoseExtractorError: LocalizedError {
    case cameraUnavailable
    case cameraAccessDenied
    case configurationFailed(String)
    case modelMissing(String)
    case mediaPipeUnavailable

    var errorDescription: String? {
        switch self {
        case .cameraUnavailable:
            return "No camera is available on this device."
        case .cameraAccessDenied:
            return "Camera access was denied. Enable it in Settings › DPMonitor."
        case .configurationFailed(let detail):
            return "Could not configure the capture session: \(detail)"
        case .modelMissing(let name):
            return "The pose model '\(name)' is not in the app bundle. See ios/README.md › Bundled assets."
        case .mediaPipeUnavailable:
            return "MediaPipeTasksVision is not linked. Run `pod install` in ios/ and open DPMonitor.xcworkspace."
        }
    }
}

// MARK: - PoseExtractor

final class PoseExtractor: NSObject {

    // MARK: Callbacks

    /// Branch A: screen-relative landmarks (x, y in [0,1], z relative, w = visibility).
    ///
    /// Delivered on `processingQueue`, NOT on main. Two consumers need it:
    /// the overlay (which hops to main itself) and the rep counter, whose
    /// hip-excursion heuristic only works in screen space — world landmarks
    /// are hip-centred, so their hip y is ~0 every frame and carries no
    /// rep signal at all.
    var onRawPoseDetected: (([String: SIMD4<Float>]) -> Void)?

    /// Branch B: world-space landmarks in metres (w = visibility).
    /// Delivered on `processingQueue` — it feeds the numeric pipeline.
    var onWorldPoseDetected: (([String: SIMD4<Float>], Double) -> Void)?

    /// Surfaced so the session screen can show a real error instead of a
    /// frozen preview.
    var onError: ((Error) -> Void)?

    // MARK: Capture plumbing

    let session = AVCaptureSession()
    private let processingQueue = DispatchQueue(label: "au.edu.usyd.dpmonitor.pose.processing",
                                                qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()
    private var config: PoseExtractorConfig
    private var isConfigured = false

    /// Guards against MediaPipe rejecting a non-monotonic timestamp when the
    /// capture clock hiccups during a rotation or an interruption.
    private var lastTimestampMs: Int = -1

    #if canImport(MediaPipeTasksVision)
    private var landmarker: PoseLandmarker?
    #endif

    init(config: PoseExtractorConfig = PoseExtractorConfig()) {
        self.config = config
        super.init()
    }

    deinit {
        if session.isRunning { session.stopRunning() }
    }

    // MARK: Lifecycle

    /// Requests camera permission, builds the capture graph and loads the
    /// landmarker. Safe to call more than once.
    func prepare() async throws {
        try await requestCameraAccess()
        try configureSessionIfNeeded()
        try loadLandmarker()
    }

    func start() {
        guard isConfigured else {
            poseLog.error("start() called before prepare() succeeded")
            return
        }
        processingQueue.async { [session] in
            guard !session.isRunning else { return }
            session.startRunning()
        }
    }

    func stop() {
        processingQueue.async { [session] in
            guard session.isRunning else { return }
            session.stopRunning()
        }
        lastTimestampMs = -1
    }

    /// Flip between the front and rear camera mid-session.
    func switchCamera() {
        processingQueue.async { [weak self] in
            guard let self else { return }
            let next: AVCaptureDevice.Position =
                (self.config.cameraPosition == .front) ? .back : .front
            self.config.cameraPosition = next
            self.session.beginConfiguration()
            for input in self.session.inputs { self.session.removeInput(input) }
            do {
                try self.attachCameraInput()
            } catch {
                poseLog.error("camera switch failed: \(error.localizedDescription)")
                self.onError?(error)
            }
            self.session.commitConfiguration()
            self.applyOutputOrientation()
        }
    }

    // MARK: Permission

    private func requestCameraAccess() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if !granted { throw PoseExtractorError.cameraAccessDenied }
        case .denied, .restricted:
            throw PoseExtractorError.cameraAccessDenied
        @unknown default:
            throw PoseExtractorError.cameraAccessDenied
        }
    }

    // MARK: Capture graph

    private func configureSessionIfNeeded() throws {
        guard !isConfigured else { return }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // The plan called for `.photo`, but `.photo` delivers a still-image
        // aspect that AVCaptureVideoDataOutput has to letterbox, and it
        // pushes far more pixels than BlazePose consumes. 720p30 is the
        // documented sweet spot for MediaPipe on-device pose.
        if session.canSetSessionPreset(config.sessionPreset) {
            session.sessionPreset = config.sessionPreset
        } else {
            session.sessionPreset = .high
        }

        try attachCameraInput()

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        // Dropping is correct here: a late frame is worse than a missing one
        // because the EMA smoother and the 64-frame window both assume a
        // roughly uniform cadence.
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)

        guard session.canAddOutput(videoOutput) else {
            throw PoseExtractorError.configurationFailed("cannot add video data output")
        }
        session.addOutput(videoOutput)

        applyOutputOrientation()
        isConfigured = true
    }

    private func attachCameraInput() throws {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .builtInDualWideCamera, .builtInTrueDepthCamera],
            mediaType: .video,
            position: config.cameraPosition
        )
        guard let device = discovery.devices.first
                ?? AVCaptureDevice.default(for: .video) else {
            throw PoseExtractorError.cameraUnavailable
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw PoseExtractorError.configurationFailed("cannot add camera input")
        }
        session.addInput(input)

        // Pin the frame rate so the temporal window covers a predictable
        // wall-clock span (64 frames ≈ 2.13 s at 30 FPS).
        try? device.lockForConfiguration()
        let target = CMTime(value: 1, timescale: config.targetFrameRate)
        if device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
            $0.minFrameDuration <= target && target <= $0.maxFrameDuration
        }) {
            device.activeVideoMinFrameDuration = target
            device.activeVideoMaxFrameDuration = target
        }
        device.unlockForConfiguration()
    }

    private func applyOutputOrientation() {
        guard let connection = videoOutput.connection(with: .video) else { return }
        if connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90     // portrait
        }
        // Mirror the front camera so the overlay matches what the user sees.
        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = (config.cameraPosition == .front)
        }
    }

    // MARK: Landmarker

    private func loadLandmarker() throws {
        #if canImport(MediaPipeTasksVision)
        guard landmarker == nil else { return }
        guard let path = Bundle.main.path(forResource: config.modelName,
                                          ofType: config.modelExtension) else {
            throw PoseExtractorError.modelMissing(
                "\(config.modelName).\(config.modelExtension)")
        }

        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = path
        // `.all` lets MediaPipe place the graph on the GPU/ANE where it can.
        options.baseOptions.delegate = .GPU
        options.runningMode = .liveStream
        options.numPoses = config.numPoses
        options.minPoseDetectionConfidence = config.minPoseDetectionConfidence
        options.minPosePresenceConfidence = config.minPosePresenceConfidence
        options.minTrackingConfidence = config.minTrackingConfidence
        // Segmentation masks are pure cost for us — we only need landmarks.
        options.shouldOutputSegmentationMasks = false
        options.poseLandmarkerLiveStreamDelegate = self

        do {
            landmarker = try PoseLandmarker(options: options)
        } catch {
            // GPU delegate is unavailable on some simulators; retry on CPU
            // rather than failing the whole session.
            poseLog.warning("GPU delegate unavailable (\(error.localizedDescription)); falling back to CPU")
            options.baseOptions.delegate = .CPU
            landmarker = try PoseLandmarker(options: options)
        }
        #else
        throw PoseExtractorError.mediaPipeUnavailable
        #endif
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension PoseExtractor: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureVideoDataOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Presentation time is the authoritative clock: it is monotonic and
        // survives dropped frames, unlike a frame counter.
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        var timestampMs = Int(CMTimeGetSeconds(pts) * 1000.0)
        if timestampMs <= lastTimestampMs {
            // MediaPipe's live-stream mode rejects non-increasing timestamps.
            timestampMs = lastTimestampMs + 1
        }
        lastTimestampMs = timestampMs

        #if canImport(MediaPipeTasksVision)
        guard let landmarker else { return }
        do {
            // MPImage retains what it needs; the CMSampleBuffer itself is
            // released by ARC when this scope exits. We deliberately do NOT
            // call CMSampleBufferInvalidate() — no strong reference or
            // escaping closure captures the buffer, so manual invalidation
            // would only risk tearing down memory MPImage still owns.
            let image = try MPImage(sampleBuffer: sampleBuffer, orientation: .up)
            try landmarker.detectAsync(image: image, timestampInMilliseconds: timestampMs)
        } catch {
            poseLog.debug("detectAsync skipped frame: \(error.localizedDescription)")
        }
        #else
        _ = timestampMs
        #endif
    }

    func captureOutput(_ output: AVCaptureVideoDataOutput,
                       didDrop sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Expected under load; the sliding window tolerates gaps.
    }
}

// MARK: - PoseLandmarkerLiveStreamDelegate

#if canImport(MediaPipeTasksVision)
extension PoseExtractor: PoseLandmarkerLiveStreamDelegate {

    func poseLandmarker(_ poseLandmarker: PoseLandmarker,
                        didFinishDetection result: PoseLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        if let error {
            poseLog.debug("landmarker error: \(error.localizedDescription)")
            return
        }
        guard let result else { return }

        // Branch A — screen-relative, for the overlay and the rep counter.
        // Emitted first so the overlay never lags the metrics by a frame.
        if let screen = result.landmarks.first, !screen.isEmpty {
            onRawPoseDetected?(Self.dictionary(fromNormalized: screen))
        }

        // Branch B — world-space metres, for the model.
        if let world = result.worldLandmarks.first, !world.isEmpty {
            let dict = Self.dictionary(fromWorld: world)
            // Already on `processingQueue` (MediaPipe calls back on the queue
            // that submitted the frame), so hand it straight through.
            onWorldPoseDetected?(dict, Double(timestampInMilliseconds))
        }
    }

    private static func dictionary(
        fromNormalized landmarks: [NormalizedLandmark]
    ) -> [String: SIMD4<Float>] {
        var out = [String: SIMD4<Float>](minimumCapacity: PoseLandmarks.count)
        let n = min(landmarks.count, PoseLandmarks.count)
        for i in 0..<n {
            let l = landmarks[i]
            out[PoseLandmarks.names[i]] = SIMD4(l.x, l.y, l.z, confidence(of: l.visibility, l.presence))
        }
        return out
    }

    private static func dictionary(
        fromWorld landmarks: [Landmark]
    ) -> [String: SIMD4<Float>] {
        var out = [String: SIMD4<Float>](minimumCapacity: PoseLandmarks.count)
        let n = min(landmarks.count, PoseLandmarks.count)
        for i in 0..<n {
            let l = landmarks[i]
            out[PoseLandmarks.names[i]] = SIMD4(l.x, l.y, l.z, confidence(of: l.visibility, l.presence))
        }
        return out
    }

    /// MediaPipe reports `visibility` and `presence` separately and either
    /// may be nil depending on the model bundle. The web app consumes a
    /// single `visibility` scalar, so we collapse them the same way: take
    /// visibility when present, else presence, else assume visible (1.0) so
    /// a nil field can never be misread as "occluded".
    private static func confidence(of visibility: NSNumber?, _ presence: NSNumber?) -> Float {
        if let v = visibility { return v.floatValue }
        if let p = presence { return p.floatValue }
        return 1.0
    }
}
#endif
