//
//  PoseSmoother.swift
//  DPMonitor
//
//  Exponential Moving Average temporal filter for the Branch B (model)
//  pipeline. Direct port of frontend/src/lib/poseSmoothing.js.
//
//  This is NOT applied to the overlay (Branch A), which renders raw
//  MediaPipe coordinates so the skeleton stays pixel-locked to the body.
//  The model cares about temporal stability; the overlay cares about
//  positional fidelity. Smoothing both would make the skeleton lag.
//
//  Occlusion behaviour: when a joint's visibility drops below the
//  threshold, alpha collapses from 0.6 to 0.15, so the smoothed position
//  leans hard on the previous frame instead of chasing a low-confidence
//  guess. Visibility itself is smoothed with the same alpha to stop the
//  overlay flickering at the threshold boundary.
//

import Foundation
import simd

struct PoseSmoother {

    /// Base smoothing factor. Higher = more responsive, lower = smoother.
    let alpha: Float
    /// Visibility below which a joint counts as occluded.
    let occlusionThreshold: Float
    /// Alpha used for occluded joints — low enough to effectively hold position.
    let occludedAlpha: Float

    private var prev: [String: SIMD4<Float>]?

    init(alpha: Float = 0.6,
         occlusionThreshold: Float = 0.5,
         occludedAlpha: Float = 0.15) {
        self.alpha = alpha
        self.occlusionThreshold = occlusionThreshold
        self.occludedAlpha = occludedAlpha
        self.prev = nil
    }

    /// Smooth one frame. Returns a new dictionary of the same shape.
    mutating func smooth(_ raw: [String: SIMD4<Float>]) -> [String: SIMD4<Float>] {
        // First frame: nothing to blend against, seed the state.
        guard let previous = prev else {
            prev = raw
            return raw
        }

        var out = [String: SIMD4<Float>](minimumCapacity: raw.count)
        for (name, rawJoint) in raw {
            guard let prevJoint = previous[name] else {
                // A joint that did not exist last frame — take it raw.
                out[name] = rawJoint
                continue
            }
            // `rawJoint.w` is visibility. Note this reads the *raw*
            // visibility, not the smoothed one, matching the JS reference.
            let a: Float = rawJoint.w < occlusionThreshold ? occludedAlpha : alpha
            // Written as a*raw + (1-a)*prev (rather than the algebraically
            // equivalent prev + (raw-prev)*a) so the float rounding matches
            // the Python/JS reference bit-for-bit in the parity fixtures.
            out[name] = SIMD4(repeating: a) * rawJoint
                      + SIMD4(repeating: 1 - a) * prevJoint
        }

        prev = out
        return out
    }

    /// Clear state. Call on session start, stop and recalibrate.
    mutating func reset() {
        prev = nil
    }
}
