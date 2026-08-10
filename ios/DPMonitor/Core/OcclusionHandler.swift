//
//  OcclusionHandler.swift
//  DPMonitor
//
//  Port of `apply_occlusion_carryforward` from
//  backend/analyzer/normalization.py.
//
//  When a joint's visibility falls below the threshold, its detection is a
//  low-confidence guess. Feeding that into the 64-frame CTR-GCN window
//  injects spatial noise that looks, to the model, like real movement. We
//  carry the previous frame's NORMALISED value forward instead — held
//  position is a far better prior than a hallucinated one.
//
//  Ordering note: this runs AFTER normalisation, not before, so the
//  carried-forward value is already in the model's coordinate frame. If it
//  ran first, a carried-forward raw joint would be re-centred against a
//  different hip midpoint and drift.
//

import Foundation

struct OcclusionHandler {

    static let defaultThreshold: Float = 0.5

    /// Previous frame's normalised 33×3 coordinates.
    private var prevNormed: [Float]?

    let threshold: Float

    init(threshold: Float = OcclusionHandler.defaultThreshold) {
        self.threshold = threshold
        self.prevNormed = nil
    }

    /// Replace occluded joints in `normed` with the previous frame's values.
    ///
    /// - Parameters:
    ///   - normed: 33×3 row-major normalised coordinates, modified in place.
    ///   - visibility: 33 per-joint confidences in [0, 1].
    mutating func apply(_ normed: inout [Float], visibility: [Float]) {
        precondition(normed.count == PoseLandmarks.count * PoseLandmarks.channels)

        // Mismatched visibility length: Python bails out rather than
        // guessing. Same here.
        guard visibility.count == PoseLandmarks.count else {
            prevNormed = normed
            return
        }

        // First frame of a session — nothing to carry forward.
        guard let prev = prevNormed else {
            prevNormed = normed
            return
        }

        for v in 0..<PoseLandmarks.count where visibility[v] < threshold {
            let base = v * 3
            normed[base]     = prev[base]
            normed[base + 1] = prev[base + 1]
            normed[base + 2] = prev[base + 2]
        }

        prevNormed = normed
    }

    /// Non-mutating variant used by the parity tests, which need to inject
    /// an explicit `prev` rather than build one up frame by frame.
    static func applied(_ normed: [Float],
                        visibility: [Float],
                        prev: [Float]?,
                        threshold: Float = OcclusionHandler.defaultThreshold) -> [Float] {
        guard let prev, visibility.count == normed.count / 3 else { return normed }
        var out = normed
        for v in 0..<(normed.count / 3) where visibility[v] < threshold {
            let base = v * 3
            out[base]     = prev[base]
            out[base + 1] = prev[base + 1]
            out[base + 2] = prev[base + 2]
        }
        return out
    }

    mutating func reset() {
        prevNormed = nil
    }
}
