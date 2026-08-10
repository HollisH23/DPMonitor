//
//  Synthesis.swift
//  DPMonitor
//
//  Post-session synthesis. Port of backend/analyzer/synthesis.py.
//
//  Turns the per-frame session log into the four headline numbers plus the
//  two chart series the results screen renders:
//
//    * overall accuracy   — mean similarity if a reference was loaded,
//                           else mean quality scaled to 0–100
//    * rep count by angle — troughs in the most-active joint's angle series
//    * per-rep peak ROM   — max − min within each rep slice
//    * fatigue index      — variance of per-rep tremor; late reps that
//                           tremble more than early ones push this up
//
//  Deliberately kept free of UI and Core Data types so it can be unit
//  tested against the Python fixtures directly.
//

import Foundation

// MARK: - Session log

/// One analysed frame. Mirrors the dicts appended to
/// `CTRGCNAnalyzer._session_log`.
struct SessionLogEntry {
    var frameIndex: Int
    var timestampMs: Double
    /// 0–100 cosine similarity, or nil when no reference is loaded.
    var similarity: Double?
    var angles: [String: Double]
}

// MARK: - Output

struct RepROM: Identifiable {
    var rep: Int
    var minDeg: Double
    var maxDeg: Double
    var rangeDeg: Double
    var id: Int { rep }
}

struct ROMCurvePoint: Identifiable {
    var timestampMs: Double
    var angle: Double
    var isTrough: Bool
    var id: Double { timestampMs }
}

struct StabilityPoint: Identifiable {
    var rep: Int
    var stability: Double
    var id: Int { rep }
}

struct SessionSynthesis {
    var overallAccuracy: Double?
    var repCountByAngle: Int = 0
    var perRepROM: [RepROM] = []
    var fatigueIndex: Double?
    var primaryJoint: String?
    var romCurve: [ROMCurvePoint] = []
    var stabilityTrend: [StabilityPoint] = []

    static let empty = SessionSynthesis(overallAccuracy: nil)
}

// MARK: - Synthesis

enum Synthesis {

    /// Minimum drop below the running peak for a sample to count as a
    /// trough. 8° ignores camera jitter but still catches shallow assisted
    /// reps such as supported knee extension.
    static let troughHysteresisDeg: Double = 8.0

    /// The UI canvas renders a few hundred points comfortably; longer
    /// sessions are strided down rather than shipped whole.
    static let maxCurvePoints = 400

    /// Joint that moved most across the session — the one whose rep cycle
    /// the user actually cares about (knee for squats, elbow for raises).
    ///
    /// Iterates `PoseLandmarks.angleTriplets` in its fixed order so ties
    /// resolve deterministically, unlike Python's dict iteration.
    static func pickPrimaryJoint(_ log: [SessionLogEntry]) -> String? {
        guard !log.isEmpty else { return nil }

        var valuesByJoint: [String: (min: Double, max: Double, n: Int)] = [:]
        for entry in log {
            for (joint, deg) in entry.angles where !deg.isNaN {
                if var acc = valuesByJoint[joint] {
                    acc.min = Swift.min(acc.min, deg)
                    acc.max = Swift.max(acc.max, deg)
                    acc.n += 1
                    valuesByJoint[joint] = acc
                } else {
                    valuesByJoint[joint] = (deg, deg, 1)
                }
            }
        }
        guard !valuesByJoint.isEmpty else { return nil }

        var best: String?
        var bestRange = -1.0
        for triplet in PoseLandmarks.angleTriplets {
            guard let acc = valuesByJoint[triplet.name], acc.n >= 3 else { continue }
            let range = acc.max - acc.min
            if range > bestRange {
                bestRange = range
                best = triplet.name
            }
        }
        // Any joint not in the canonical triplet list (shouldn't happen, but
        // the log is just strings) still gets a fair shot.
        for (joint, acc) in valuesByJoint where acc.n >= 3 {
            guard PoseLandmarks.angleTriplets.first(where: { $0.name == joint }) == nil else { continue }
            let range = acc.max - acc.min
            if range > bestRange { bestRange = range; best = joint }
        }
        return best
    }

    /// Indices of trough samples — one per completed rep cycle.
    ///
    /// Hysteresis peak detector adapted to inverted signals: rehab reps
    /// usually *decrease* a flexion angle and then come back up, so we arm
    /// on a sufficient drop below the running peak and commit the trough
    /// once the signal recovers by the same margin.
    static func detectRepTroughs(_ series: [Double],
                                 hysteresisDeg: Double = troughHysteresisDeg) -> [Int] {
        guard series.count >= 3 else { return [] }

        var sig = series
        if sig.contains(where: { $0.isNaN }) {
            guard Kinematics.interpolateNaNsInPlace(&sig) else { return [] }
        }

        var troughs: [Int] = []
        var runningPeak = sig[0]
        var armed = false
        var candidateIdx = -1
        var candidateVal = Double.infinity

        for (i, v) in sig.enumerated() {
            if v > runningPeak {
                runningPeak = v
                if !armed {
                    candidateIdx = -1
                    candidateVal = .infinity
                }
            }
            if !armed && (runningPeak - v) >= hysteresisDeg {
                armed = true
                candidateIdx = i
                candidateVal = v
            } else if armed {
                if v < candidateVal {
                    candidateIdx = i
                    candidateVal = v
                } else if (v - candidateVal) >= hysteresisDeg {
                    troughs.append(candidateIdx)
                    armed = false
                    runningPeak = v
                    candidateIdx = -1
                    candidateVal = .infinity
                }
            }
        }
        return troughs
    }

    /// Split a series into `[start, end)` windows, one per detected rep.
    static func repSlices(sampleCount: Int, troughs: [Int]) -> [(Int, Int)] {
        guard !troughs.isEmpty else { return [] }
        let bounds = [0] + troughs + [sampleCount]
        var slices: [(Int, Int)] = []
        for i in 0..<(bounds.count - 1) {
            let a = bounds[i], b = bounds[i + 1]
            if b - a >= 3 { slices.append((a, b)) }   // ignore tail slivers
        }
        return slices
    }

    /// Compute the results-screen payload from a session log.
    static func synthesize(_ log: [SessionLogEntry],
                           qualitySamples: [Double] = []) -> SessionSynthesis {
        var out = SessionSynthesis.empty
        guard !log.isEmpty else { return out }

        // ---- Overall accuracy -----------------------------------------
        let sims = log.compactMap(\.similarity)
        if !sims.isEmpty {
            out.overallAccuracy = sims.reduce(0, +) / Double(sims.count)
        } else if !qualitySamples.isEmpty {
            let mean = qualitySamples.reduce(0, +) / Double(qualitySamples.count)
            out.overallAccuracy = mean * 100.0
        }

        // ---- Primary joint --------------------------------------------
        guard let primary = pickPrimaryJoint(log) else { return out }
        out.primaryJoint = primary

        let times = log.map(\.timestampMs)
        let angles = log.map { $0.angles[primary] ?? Double.nan }

        // ---- Rep detection + per-rep ROM -------------------------------
        let troughs = detectRepTroughs(angles)
        out.repCountByAngle = troughs.count
        let troughSet = Set(troughs)

        var perRepROM: [RepROM] = []
        var perRepTremor: [Double] = []
        var stability: [StabilityPoint] = []

        for (r, slice) in repSlices(sampleCount: angles.count, troughs: troughs).enumerated() {
            let segment = Array(angles[slice.0..<slice.1])
            let rom = Kinematics.rangeOfMotion(segment)
            perRepROM.append(RepROM(rep: r + 1,
                                    minDeg: rom.minDeg,
                                    maxDeg: rom.maxDeg,
                                    rangeDeg: rom.rangeDeg))
            let accel = Kinematics.tremorMetrics(segment).accelerationRMS
            perRepTremor.append(accel)
            // Stability ∈ [0, 1]; saturates at 0 for wildly jittery reps.
            let s = max(0.0, 1.0 - min(1.0, accel / 5.0))
            stability.append(StabilityPoint(rep: r + 1, stability: (s * 10000).rounded() / 10000))
        }

        out.perRepROM = perRepROM
        out.stabilityTrend = stability

        // ---- Fatigue --------------------------------------------------
        if perRepTremor.count >= 2 {
            let mean = perRepTremor.reduce(0, +) / Double(perRepTremor.count)
            out.fatigueIndex = perRepTremor
                .reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(perRepTremor.count)
        }

        // ---- ROM curve -------------------------------------------------
        let stride = max(1, Int(ceil(Double(angles.count) / Double(maxCurvePoints))))
        var curve: [ROMCurvePoint] = []
        var i = 0
        while i < angles.count {
            let v = angles[i]
            if !v.isNaN {
                curve.append(ROMCurvePoint(timestampMs: times[i],
                                           angle: v,
                                           isTrough: troughSet.contains(i)))
            }
            i += stride
        }
        out.romCurve = curve

        return out
    }
}
