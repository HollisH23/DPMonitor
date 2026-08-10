//
//  RepCounter.swift
//  DPMonitor
//
//  Deterministic hip-excursion rep heuristic. Port of
//  `CTRGCNAnalyzer._update_rep_counter` + `_hip_y`.
//
//  Rep counting is deliberately NOT derived from the model's logits.
//  CTR-GCN is an action-quality classifier, not a counter; coupling the
//  count to its output would conflate "how well" with "how many" and make
//  a bad-form rep silently uncountable. The classifier drives quality; this
//  geometry drives the count.
//
//  The signal is the hip midpoint's y, smoothed over a 5-frame box window,
//  with a direction-tracking hysteresis: a rep is committed when the signal
//  climbs `minExcursion` back out of a trough it previously fell into.
//

import Foundation

struct RepCounter {

    /// Minimum normalised y excursion that separates a real rep from
    /// camera shake and postural sway. 0.06 in screen-normalised units.
    static let minExcursion: Float = 0.06

    /// Box-filter length applied to hip y before the hysteresis runs.
    static let smoothingWindow = 5

    private(set) var count: Int = 0
    private(set) var smoothedHistory: [Float] = []

    private var hipWindow: [Float] = []
    private var lastExtreme: Float?
    /// +1 rising, -1 falling, 0 undetermined.
    private var direction: Int = 0

    let minExcursion: Float

    init(minExcursion: Float = RepCounter.minExcursion) {
        self.minExcursion = minExcursion
    }

    /// Hip midpoint y from a raw (un-normalised) landmark dictionary.
    ///
    /// Uses raw coordinates on purpose: normalisation puts the hip at the
    /// origin every frame, which would flatten the very signal we count.
    static func hipY(_ points: [String: SIMD4<Float>]) -> Float? {
        guard let lh = points["left_hip"], let rh = points["right_hip"] else { return nil }
        return 0.5 * (lh.y + rh.y)
    }

    /// Feed one frame's hip y. Returns `true` on the frame a rep completes.
    @discardableResult
    mutating func update(hipY: Float) -> Bool {
        hipWindow.append(hipY)
        if hipWindow.count > Self.smoothingWindow {
            hipWindow.removeFirst(hipWindow.count - Self.smoothingWindow)
        }
        let smoothed = hipWindow.reduce(0, +) / Float(hipWindow.count)
        return applyHysteresis(smoothed)
    }

    private mutating func applyHysteresis(_ smoothed: Float) -> Bool {
        smoothedHistory.append(smoothed)

        guard let extreme = lastExtreme else {
            lastExtreme = smoothed
            return false
        }

        let delta = smoothed - extreme

        if direction >= 0 && delta > 0 {
            // Rising: track the peak.
            direction = 1
            lastExtreme = max(extreme, smoothed)
        } else if direction == 1 && delta < -minExcursion {
            // Fell far enough off the peak — we are now descending.
            direction = -1
            lastExtreme = smoothed
        } else if direction == -1 && delta < 0 {
            // Still descending: track the trough.
            lastExtreme = min(extreme, smoothed)
        } else if direction == -1 && delta > minExcursion {
            // Climbed back out of the trough: that is one completed rep.
            count += 1
            direction = 1
            lastExtreme = smoothed
            return true
        }
        return false
    }

    /// Stability from the dispersion of frame-to-frame hip movement.
    /// Port of the `generate_summary` stability calculation.
    var stabilityScore: Double {
        guard smoothedHistory.count >= 3 else { return 0.0 }
        var diffs = [Double]()
        diffs.reserveCapacity(smoothedHistory.count - 1)
        for i in 1..<smoothedHistory.count {
            diffs.append(Double(smoothedHistory[i] - smoothedHistory[i - 1]))
        }
        let mean = diffs.reduce(0, +) / Double(diffs.count)
        let variance = diffs.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(diffs.count)
        let spread = variance.squareRoot()
        return max(0.0, 1.0 - min(1.0, spread * 50.0))
    }

    mutating func reset() {
        count = 0
        smoothedHistory.removeAll(keepingCapacity: true)
        hipWindow.removeAll(keepingCapacity: true)
        lastExtreme = nil
        direction = 0
    }
}
