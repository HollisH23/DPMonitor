//
//  Kinematics.swift
//  DPMonitor
//
//  Geometric counterpart to the learned model. Exact port of
//  backend/analyzer/kinematics.py.
//
//  CTR-GCN answers "does this look like good form?"; clinicians also want
//  numbers that map onto recovery milestones. These are computed from the
//  same sliding window:
//
//    * jointAngle        — angle at a vertex, in degrees
//    * rangeOfMotion     — peak-to-trough swing (the clinical ROM number)
//    * tremorMetrics     — RMS of velocity and acceleration; smooth motion
//                          tends to 0, jitter grows
//
//  Everything runs in `Double` because the Python reference is float64 and
//  the parity tests hold to 1e-4. NaN is a first-class value here: a
//  missing landmark must propagate as NaN, never silently become 0, or it
//  would drag a patient's reported ROM toward zero.
//

import Accelerate
import Foundation
import simd

enum Kinematics {

    // MARK: - Joint angle

    /// Angle at `vertex` in degrees, between the rays vertex→a and vertex→b.
    ///
    /// Returns `.nan` when either segment has (near) zero length — the
    /// "missing landmark" case. Callers use `isNaN` to skip the frame.
    static func jointAngle(a: SIMD3<Double>,
                           vertex: SIMD3<Double>,
                           b: SIMD3<Double>) -> Double {
        let v1 = a - vertex
        let v2 = b - vertex
        let n1 = simd_length(v1)
        let n2 = simd_length(v2)
        guard n1 >= 1e-9, n2 >= 1e-9 else { return .nan }
        let cosTheta = min(max(simd_dot(v1, v2) / (n1 * n2), -1.0), 1.0)
        return acos(cosTheta) * 180.0 / Double.pi
    }

    /// Convenience overload for the `Float` storage used by the pipeline.
    static func jointAngle(a: SIMD3<Float>,
                           vertex: SIMD3<Float>,
                           b: SIMD3<Float>) -> Double {
        jointAngle(a: SIMD3<Double>(a), vertex: SIMD3<Double>(vertex), b: SIMD3<Double>(b))
    }

    /// Read the angle for one triplet out of a flat 33×3 frame.
    static func jointAngle(frame: [Float],
                           triplet: (name: String, a: Int, vertex: Int, b: Int)) -> Double {
        @inline(__always)
        func joint(_ v: Int) -> SIMD3<Double> {
            SIMD3(Double(frame[v * 3]), Double(frame[v * 3 + 1]), Double(frame[v * 3 + 2]))
        }
        return jointAngle(a: joint(triplet.a), vertex: joint(triplet.vertex), b: joint(triplet.b))
    }

    /// Every clinical angle for one frame. NaN entries are dropped, matching
    /// `CTRGCNAnalyzer._latest_frame_angles`.
    static func frameAngles(_ frame: [Float]) -> [String: Double] {
        var out = [String: Double](minimumCapacity: PoseLandmarks.angleTriplets.count)
        for triplet in PoseLandmarks.angleTriplets {
            let angle = jointAngle(frame: frame, triplet: triplet)
            if !angle.isNaN { out[triplet.name] = angle }
        }
        return out
    }

    /// Per-frame angle series over a `(T, V, C)` window.
    /// Frames with a missing landmark yield NaN so `rangeOfMotion` can skip them.
    static func jointAngleSeries(window: [[Float]],
                                 triplet: (name: String, a: Int, vertex: Int, b: Int)) -> [Double] {
        window.map { jointAngle(frame: $0, triplet: triplet) }
    }

    // MARK: - Range of motion

    struct ROM {
        var minDeg: Double
        var maxDeg: Double
        var rangeDeg: Double

        static let undefined = ROM(minDeg: .nan, maxDeg: .nan, rangeDeg: .nan)
        var isDefined: Bool { !minDeg.isNaN && !maxDeg.isNaN }
    }

    /// Min / max / range of an angle series, ignoring NaNs.
    static func rangeOfMotion(_ series: [Double]) -> ROM {
        guard !series.isEmpty else { return .undefined }
        var mn = Double.infinity
        var mx = -Double.infinity
        var sawValue = false
        for v in series where !v.isNaN {
            sawValue = true
            if v < mn { mn = v }
            if v > mx { mx = v }
        }
        guard sawValue else { return .undefined }
        return ROM(minDeg: mn, maxDeg: mx, rangeDeg: mx - mn)
    }

    // MARK: - Tremor

    struct Tremor {
        var velocityRMS: Double
        var accelerationRMS: Double

        static let zero = Tremor(velocityRMS: 0, accelerationRMS: 0)
    }

    /// RMS of the first and second differences of a 1-D signal.
    ///
    /// NaN runs are linearly interpolated first so the derivatives stay
    /// defined across a brief dropout. A series shorter than 3 samples, or
    /// one that is entirely NaN, reports zeros rather than NaN — that keeps
    /// the fatigue variance well-defined for very short reps.
    static func tremorMetrics(_ signal: [Double]) -> Tremor {
        guard signal.count >= 3 else { return .zero }

        var sig = signal
        if sig.contains(where: { $0.isNaN }) {
            guard interpolateNaNsInPlace(&sig) else { return .zero }
        }

        var sumV = 0.0
        var sumA = 0.0
        let nV = sig.count - 1
        let nA = sig.count - 2

        var prevV = 0.0
        for i in 0..<nV {
            let v = sig[i + 1] - sig[i]
            sumV += v * v
            if i > 0 {
                let a = v - prevV
                sumA += a * a
            }
            prevV = v
        }

        return Tremor(velocityRMS: (sumV / Double(nV)).squareRoot(),
                      accelerationRMS: nA > 0 ? (sumA / Double(nA)).squareRoot() : 0)
    }

    /// Replicates `np.interp` over NaN positions, including its edge
    /// behaviour: samples before the first valid point take that point's
    /// value, samples after the last take the last point's value.
    ///
    /// - Returns: `false` when every sample is NaN (nothing to interpolate from).
    @discardableResult
    static func interpolateNaNsInPlace(_ sig: inout [Double]) -> Bool {
        let validIndices = sig.indices.filter { !sig[$0].isNaN }
        guard let first = validIndices.first, let last = validIndices.last else {
            return false
        }

        for i in sig.indices where sig[i].isNaN {
            if i <= first {
                sig[i] = sig[first]
            } else if i >= last {
                sig[i] = sig[last]
            } else {
                // Bracket i between the nearest valid samples on each side.
                // `validIndices` is sorted, so a linear scan from the front
                // is fine for the 64-sample windows we deal with.
                var lo = first
                var hi = last
                for v in validIndices {
                    if v < i { lo = v } else { hi = v; break }
                }
                let span = Double(hi - lo)
                let t = span > 0 ? Double(i - lo) / span : 0
                sig[i] = sig[lo] + (sig[hi] - sig[lo]) * t
            }
        }
        return true
    }

    // MARK: - Window summary

    struct JointSummary {
        var joint: String
        var rom: ROM
        var tremor: Tremor
    }

    /// ROM + tremor for every tracked joint over a `(T, V, C)` window.
    ///
    /// Computed on the UN-normalised buffer. Normalisation is a similarity
    /// transform so it would not change an angle, but reporting on the raw
    /// coordinates keeps the numbers obviously in real degrees.
    static func windowSummary(_ window: [[Float]]) -> [JointSummary] {
        guard window.count >= 3 else { return [] }
        return PoseLandmarks.angleTriplets.map { triplet in
            let series = jointAngleSeries(window: window, triplet: triplet)
            return JointSummary(joint: triplet.name,
                                rom: rangeOfMotion(series),
                                tremor: tremorMetrics(series))
        }
    }
}
