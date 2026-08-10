//
//  ActionClassifier.swift
//  DPMonitor
//
//  Core ML wrapper around the exported CTR-GCN.
//
//  Model contract — must match scripts/export_coreml.py:
//      input   "input"     float32 (1, 3, 64, 33, 1)   N, C, T, V, M
//      output  "logits"    float32 (1, num_class)
//      output  "features"  float32 (1, 256)            pre-FC embedding
//
//  The model is loaded through the generic `MLModel` API rather than the
//  Xcode-generated `CTRGCN` wrapper class. That wrapper only exists once
//  CTRGCN.mlpackage is in the target, so depending on it would make the
//  whole project fail to compile before the export script has been run —
//  a bad trade for a handful of saved lines. `modelIsAvailable` lets the
//  UI say so plainly instead of crashing.
//
//  Threading: `classify` hops to a dedicated serial queue. The camera
//  delegate queue must never block (it would stall pose extraction) and
//  the main thread must never block (it drives the 120 Hz overlay).
//

import CoreML
import Foundation
import os

private let mlLog = Logger(subsystem: "au.edu.usyd.dpmonitor", category: "coreml")

// MARK: - Result

struct ClassificationResult {
    /// P(good form) after softmax, in [0, 1].
    let qualityScore: Float
    /// True when P(bad) exceeds P(good).
    let isCompensatory: Bool
    /// Raw logits, surfaced for diagnostics.
    let logits: [Float]
    /// Pre-FC embedding, used for reference-movement similarity.
    let features: [Float]
    /// 0–100 similarity against the loaded reference, or nil if none is set.
    let similarityScore: Double?
}

enum ActionClassifierError: LocalizedError {
    case modelNotBundled
    case unexpectedOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelNotBundled:
            return "CTRGCN.mlpackage is not in the app bundle. Run `python scripts/export_coreml.py` and regenerate the project."
        case .unexpectedOutput(let name):
            return "Core ML model did not produce the expected output '\(name)'."
        }
    }
}

// MARK: - Classifier

final class ActionClassifier {

    /// Index of the "good form" logit. Matches `_DEFAULT_GOOD_FORM_INDEX`.
    var goodFormIndex: Int = 0
    /// Index of the "compensatory form" logit.
    var badFormIndex: Int = 1

    /// Optional clinician reference embedding for similarity scoring.
    private var referenceFeature: [Float]?

    private let model: MLModel?
    private let inferenceQueue = DispatchQueue(
        label: "au.edu.usyd.dpmonitor.coreml.inference", qos: .userInitiated)

    var modelIsAvailable: Bool { model != nil }

    /// False when the bundled model was exported from seeded-random weights
    /// rather than a trained checkpoint.
    ///
    /// This matters far more than it sounds. With untrained weights the
    /// logits reach ~1e4 after ten GCN blocks, so the softmax saturates to
    /// exactly [0, 1] on every window — which means `isCompensatory` is
    /// permanently true, the skeleton is permanently red and the HUD loops
    /// "Form drift detected". That reads as a broken app rather than an
    /// uncalibrated one. When this is false the UI shows the score as
    /// unavailable instead of showing a confident, meaningless zero.
    ///
    /// Read from `CTRGCN.manifest.json`, written by scripts/export_coreml.py.
    let isCalibrated: Bool

    init() {
        isCalibrated = Self.readCalibration()

        let config = MLModelConfiguration()
        // `.all` lets Core ML pick the Neural Engine where the op set allows
        // it and fall back to GPU/CPU per layer. Verify actual placement with
        // Xcode's Core ML Performance Report — see README › NPU profiling.
        config.computeUnits = .all
        config.allowLowPrecisionAccumulationOnGPU = true

        // Xcode compiles a bundled .mlpackage to .mlmodelc at build time.
        if let url = Bundle.main.url(forResource: "CTRGCN", withExtension: "mlmodelc") {
            model = try? MLModel(contentsOf: url, configuration: config)
            if model == nil {
                mlLog.error("CTRGCN.mlmodelc present but failed to load")
            }
        } else {
            mlLog.error("CTRGCN.mlmodelc not found in bundle — run scripts/export_coreml.py")
            model = nil
        }
    }

    // MARK: Reference movement

    func setReferenceFeature(_ feature: [Float]?) {
        inferenceQueue.async { [weak self] in
            self?.referenceFeature = feature
        }
    }

    // MARK: Inference

    /// Run one forward pass. The completion fires on the main queue.
    func classify(_ input: MLMultiArray,
                  completion: @escaping (Result<ClassificationResult, Error>) -> Void) {
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            let outcome: Result<ClassificationResult, Error>
            do {
                outcome = .success(try self.runSync(input))
            } catch {
                outcome = .failure(error)
            }
            DispatchQueue.main.async { completion(outcome) }
        }
    }

    /// Synchronous variant — for tests and for callers already off the
    /// main thread that want to control their own queueing.
    func runSync(_ input: MLMultiArray) throws -> ClassificationResult {
        guard let model else { throw ActionClassifierError.modelNotBundled }

        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": input])
        let out = try model.prediction(from: provider)

        guard let logitsArray = out.featureValue(for: "logits")?.multiArrayValue else {
            throw ActionClassifierError.unexpectedOutput("logits")
        }
        let logits = Self.floats(from: logitsArray)
        let features = out.featureValue(for: "features")?.multiArrayValue
            .map(Self.floats(from:)) ?? []

        let probs = Self.softmax(logits)
        let good = goodFormIndex < probs.count ? probs[goodFormIndex] : 0
        let bad = badFormIndex < probs.count ? probs[badFormIndex] : 0

        return ClassificationResult(
            qualityScore: good,
            isCompensatory: bad > good,
            logits: logits,
            features: features,
            similarityScore: Self.similarityScore(current: features, target: referenceFeature)
        )
    }

    // MARK: Numerics

    /// Max-subtracted softmax. The shift is not cosmetic: FP16 logits from
    /// the Neural Engine can reach magnitudes where a naive `exp` overflows.
    static func softmax(_ logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else { return [] }
        let exps = logits.map { expf($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return [Float](repeating: 0, count: logits.count) }
        return exps.map { $0 / sum }
    }

    /// Port of `analyzer/similarity.py :: cosine_similarity` (float64).
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0.0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in a.indices {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        na = na.squareRoot()
        nb = nb.squareRoot()
        guard na >= 1e-12, nb >= 1e-12 else { return 0.0 }
        return dot / (na * nb)
    }

    /// Port of `analyzer/similarity.py :: similarity_score`.
    ///
    /// Negative similarity is clipped to zero: in this domain it means
    /// "completely different posture", not "opposite posture". `nil` when
    /// no clinician reference is loaded, so the UI can distinguish that
    /// from "matches poorly".
    static func similarityScore(current: [Float], target: [Float]?) -> Double? {
        guard let target else { return nil }
        let sim = max(0.0, cosineSimilarity(current, target))
        return (100.0 * sim * 100).rounded() / 100
    }

    /// Parse `CTRGCN.manifest.json`'s `weights` field.
    ///
    /// The export script writes the literal string `"seeded-random-init"`
    /// when no checkpoint was supplied, and the checkpoint path otherwise.
    /// A missing manifest is treated as uncalibrated: assuming the model is
    /// trained when we cannot tell is the failure mode that produces a
    /// confidently wrong clinical number.
    private static func readCalibration() -> Bool {
        guard let url = Bundle.main.url(forResource: "CTRGCN.manifest", withExtension: "json")
                ?? Bundle.main.url(forResource: "CTRGCN", withExtension: "manifest.json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weights = json["weights"] as? String else {
            mlLog.notice("no CTRGCN.manifest.json in bundle — treating model as uncalibrated")
            return false
        }
        let calibrated = weights != "seeded-random-init"
        mlLog.notice("model weights: \(weights, privacy: .public) — calibrated: \(calibrated)")
        return calibrated
    }

    private static func floats(from array: MLMultiArray) -> [Float] {
        let count = array.count
        var out = [Float](repeating: 0, count: count)
        array.withUnsafeBufferPointer(ofType: Float.self) { ptr in
            for i in 0..<min(count, ptr.count) { out[i] = ptr[i] }
        }
        return out
    }
}
