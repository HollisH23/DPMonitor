//
//  SessionSummary.swift
//  DPMonitor
//
//  Whole-session result. Mirrors `AnalyzerSummary` in
//  backend/analyzer/base.py plus the `summary_box` synthesis payload.
//
//  Kept as a plain `Codable` value type, separate from the Core Data
//  entity: the headline fields are stored as indexed columns (so the
//  history list can sort without deserialising anything) while the rich
//  detail is archived as JSON in a companion entity.
//

import Foundation

struct KinematicsRecord: Codable, Hashable, Identifiable {
    var joint: String
    var minDeg: Double
    var maxDeg: Double
    var rangeDeg: Double
    var velocityRMS: Double
    var accelerationRMS: Double

    var id: String { joint }

    /// Human-readable joint name, e.g. "left_knee" -> "Left Knee".
    var displayName: String {
        joint.split(separator: "_").map(\.capitalized).joined(separator: " ")
    }
}

struct SessionSummary: Identifiable {
    var id: UUID
    var date: Date
    var exerciseType: ExerciseType
    var repCount: Int
    /// Mean running quality across the session, 0–1.
    var qualityScore: Double
    /// 0–1, from the dispersion of frame-to-frame hip movement.
    var stabilityScore: Double
    var compensationEvents: Int
    var durationSeconds: Double

    var framesAnalyzed: Int
    var inferenceCalls: Int
    var windowSize: Int
    var inferenceStride: Int
    var finalWindowKinematics: [KinematicsRecord]

    /// False when the session ran against seeded-random weights, so
    /// `qualityScore` and `compensationEvents` are not clinical findings.
    /// Persisted, because a history list that silently mixes calibrated and
    /// uncalibrated sessions is worse than one that shows neither.
    var qualityIsCalibrated: Bool = true

    var synthesis: SessionSynthesis

    static let empty = SessionSummary(
        id: UUID(),
        date: Date(),
        exerciseType: .custom,
        repCount: 0,
        qualityScore: 0,
        stabilityScore: 0,
        compensationEvents: 0,
        durationSeconds: 0,
        framesAnalyzed: 0,
        inferenceCalls: 0,
        windowSize: 64,
        inferenceStride: 5,
        finalWindowKinematics: [],
        qualityIsCalibrated: true,
        synthesis: .empty
    )
}

// MARK: - Persistable detail payload

/// The JSON blob archived alongside the indexed columns.
///
/// `SessionSynthesis` carries `Identifiable` view models rather than
/// `Codable` ones, so this mirror type does the encoding. Keeping them
/// separate means a UI-layer change can never silently break the
/// on-disk format of past sessions.
struct SessionDetailPayload: Codable {
    struct RepROMRecord: Codable {
        var rep: Int
        var minDeg: Double
        var maxDeg: Double
        var rangeDeg: Double
    }
    struct CurvePoint: Codable {
        var timestampMs: Double
        var angle: Double
        var isTrough: Bool
    }
    struct StabilityRecord: Codable {
        var rep: Int
        var stability: Double
    }

    var schemaVersion: Int = 1
    var overallAccuracy: Double?
    var repCountByAngle: Int
    var primaryJoint: String?
    var fatigueIndex: Double?
    var perRepROM: [RepROMRecord]
    var romCurve: [CurvePoint]
    var stabilityTrend: [StabilityRecord]
    var finalWindowKinematics: [KinematicsRecord]
    var framesAnalyzed: Int
    var inferenceCalls: Int
    var windowSize: Int
    var inferenceStride: Int
    /// Defaults to true so sessions archived before this field existed
    /// decode without error.
    var qualityIsCalibrated: Bool = true

    init(summary: SessionSummary) {
        let s = summary.synthesis
        overallAccuracy = s.overallAccuracy
        repCountByAngle = s.repCountByAngle
        primaryJoint = s.primaryJoint
        fatigueIndex = s.fatigueIndex
        perRepROM = s.perRepROM.map {
            RepROMRecord(rep: $0.rep, minDeg: $0.minDeg,
                         maxDeg: $0.maxDeg, rangeDeg: $0.rangeDeg)
        }
        romCurve = s.romCurve.map {
            CurvePoint(timestampMs: $0.timestampMs, angle: $0.angle, isTrough: $0.isTrough)
        }
        stabilityTrend = s.stabilityTrend.map {
            StabilityRecord(rep: $0.rep, stability: $0.stability)
        }
        finalWindowKinematics = summary.finalWindowKinematics
        framesAnalyzed = summary.framesAnalyzed
        inferenceCalls = summary.inferenceCalls
        windowSize = summary.windowSize
        inferenceStride = summary.inferenceStride
        qualityIsCalibrated = summary.qualityIsCalibrated
    }

    /// Rebuild the view-facing synthesis from an archived payload.
    var synthesis: SessionSynthesis {
        SessionSynthesis(
            overallAccuracy: overallAccuracy,
            repCountByAngle: repCountByAngle,
            perRepROM: perRepROM.map {
                RepROM(rep: $0.rep, minDeg: $0.minDeg,
                       maxDeg: $0.maxDeg, rangeDeg: $0.rangeDeg)
            },
            fatigueIndex: fatigueIndex,
            primaryJoint: primaryJoint,
            romCurve: romCurve.map {
                ROMCurvePoint(timestampMs: $0.timestampMs, angle: $0.angle, isTrough: $0.isTrough)
            },
            stabilityTrend: stabilityTrend.map {
                StabilityPoint(rep: $0.rep, stability: $0.stability)
            }
        )
    }
}

// MARK: - Formatting helpers

extension SessionSummary {
    var qualityPercent: Int { Int((qualityScore * 100).rounded()) }
    var stabilityPercent: Int { Int((stabilityScore * 100).rounded()) }

    var durationText: String {
        let total = Int(durationSeconds.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
