//
//  KinematicsTests.swift
//  DPMonitorTests
//
//  Parity for the clinical numerics: joint angles, ROM, tremor — plus the
//  synthesis layer and the rep-counting heuristic built on top of them.
//
//  These are the numbers a physio actually reads, so the NaN paths get as
//  much attention as the happy path: a dropped landmark must surface as
//  "no data", never as a plausible-looking zero.
//

import XCTest
import simd
@testable import DPMonitor

final class KinematicsTests: XCTestCase {

    private var fixtures: ReferenceFixtures!

    override func setUpWithError() throws {
        fixtures = try ReferenceFixtures.load()
    }

    // MARK: Joint angle

    func testJointAngleMatchesPython() throws {
        let cases = try fixtures.cases("joint_angle")
        XCTAssertFalse(cases.isEmpty)

        for c in cases {
            let name = c.string("name") ?? "?"
            guard let a = c.doubles("a"), let v = c.doubles("vertex"), let b = c.doubles("b") else {
                XCTFail("malformed joint_angle case \(name)")
                continue
            }
            let actual = Kinematics.jointAngle(a: SIMD3(a[0], a[1], a[2]),
                                               vertex: SIMD3(v[0], v[1], v[2]),
                                               b: SIMD3(b[0], b[1], b[2]))
            assertClose(actual, c.double("expected_deg"), "joint_angle[\(name)]")

            // Where the fixture carries an analytic value, check that too —
            // it guards against Swift and Python sharing a mistake.
            if let analytic = c["analytic_deg"] as? NSNumber {
                assertClose(actual, analytic.doubleValue, tolerance: 1e-6,
                            "joint_angle[\(name)] analytic")
            }
        }
    }

    func testJointAngleReturnsNaNForZeroLengthSegment() {
        let origin = SIMD3<Double>(0, 0, 0)
        XCTAssertTrue(Kinematics.jointAngle(a: origin, vertex: origin,
                                            b: SIMD3(1, 0, 0)).isNaN)
        XCTAssertTrue(Kinematics.jointAngle(a: SIMD3(1, 0, 0), vertex: origin,
                                            b: origin).isNaN)
    }

    // MARK: Range of motion

    func testRangeOfMotionMatchesPython() throws {
        let cases = try fixtures.cases("range_of_motion")
        XCTAssertFalse(cases.isEmpty)

        for c in cases {
            let name = c.string("name") ?? "?"
            guard let series = c.doubles("series"),
                  let expected = c["expected"] as? [String: Any] else {
                XCTFail("malformed range_of_motion case \(name)")
                continue
            }
            let rom = Kinematics.rangeOfMotion(series)
            assertClose(rom.minDeg, expected.double("min_deg"), "rom[\(name)].min")
            assertClose(rom.maxDeg, expected.double("max_deg"), "rom[\(name)].max")
            assertClose(rom.rangeDeg, expected.double("range_deg"), "rom[\(name)].range")
        }
    }

    func testRangeOfMotionIsUndefinedForEmptyAndAllNaN() {
        XCTAssertFalse(Kinematics.rangeOfMotion([]).isDefined)
        XCTAssertFalse(Kinematics.rangeOfMotion([.nan, .nan, .nan]).isDefined)
        XCTAssertTrue(Kinematics.rangeOfMotion([.nan, 10, .nan, 40]).isDefined)
        assertClose(Kinematics.rangeOfMotion([.nan, 10, .nan, 40]).rangeDeg, 30,
                    tolerance: 1e-9, "NaN-skipping range")
    }

    // MARK: Tremor

    func testTremorMetricsMatchPython() throws {
        let cases = try fixtures.cases("tremor")
        XCTAssertFalse(cases.isEmpty)

        for c in cases {
            let name = c.string("name") ?? "?"
            guard let signal = c.doubles("signal"),
                  let expected = c["expected"] as? [String: Any] else {
                XCTFail("malformed tremor case \(name)")
                continue
            }
            let tremor = Kinematics.tremorMetrics(signal)
            assertClose(tremor.velocityRMS, expected.double("velocity_rms"),
                        "tremor[\(name)].velocity")
            assertClose(tremor.accelerationRMS, expected.double("acceleration_rms"),
                        "tremor[\(name)].acceleration")
        }
    }

    /// `np.interp` clamps outside its sample range; a NaN before the first
    /// valid sample must take that sample's value, not extrapolate.
    func testNaNInterpolationMatchesNumpyEdgeBehaviour() {
        var sig: [Double] = [.nan, .nan, 10, .nan, 20, .nan, .nan]
        XCTAssertTrue(Kinematics.interpolateNaNsInPlace(&sig))
        assertClose(sig[0], 10, tolerance: 1e-9, "leading NaN clamps to first valid")
        assertClose(sig[1], 10, tolerance: 1e-9, "leading NaN clamps to first valid")
        assertClose(sig[3], 15, tolerance: 1e-9, "interior NaN interpolates linearly")
        assertClose(sig[5], 20, tolerance: 1e-9, "trailing NaN clamps to last valid")
        assertClose(sig[6], 20, tolerance: 1e-9, "trailing NaN clamps to last valid")

        var allNaN: [Double] = [.nan, .nan]
        XCTAssertFalse(Kinematics.interpolateNaNsInPlace(&allNaN))
    }

    func testTremorIsZeroForShortSignals() {
        XCTAssertEqual(Kinematics.tremorMetrics([1, 2]).velocityRMS, 0)
        XCTAssertEqual(Kinematics.tremorMetrics([]).accelerationRMS, 0)
    }

    /// A smooth densely-sampled sinusoid has near-zero discrete derivatives;
    /// added jitter must measurably raise both metrics.
    func testTremorSeparatesSmoothFromJittery() {
        let smooth = (0..<128).map { 30.0 * sin(Double($0) * 2 * .pi / 128) }
        var jittery = smooth
        for i in jittery.indices { jittery[i] += (i % 2 == 0 ? 3.0 : -3.0) }

        let a = Kinematics.tremorMetrics(smooth)
        let b = Kinematics.tremorMetrics(jittery)
        XCTAssertLessThan(a.accelerationRMS, b.accelerationRMS)
        XCTAssertLessThan(a.velocityRMS, b.velocityRMS)
    }

    // MARK: Window summary

    func testWindowSummaryCoversEveryTriplet() {
        // 40 frames of a synthetic squat: knees flex and extend.
        var window: [[Float]] = []
        for t in 0..<40 {
            var frame = [Float](repeating: 0, count: 99)
            let phase = Float(t) * .pi / 20
            func set(_ v: Int, _ x: Float, _ y: Float, _ z: Float) {
                frame[v * 3] = x; frame[v * 3 + 1] = y; frame[v * 3 + 2] = z
            }
            set(PoseLandmarks.leftShoulder, 0, 0, 0)
            set(PoseLandmarks.rightShoulder, 0.3, 0, 0)
            set(PoseLandmarks.leftHip, 0, 1, 0)
            set(PoseLandmarks.rightHip, 0.3, 1, 0)
            set(PoseLandmarks.leftKnee, sin(phase) * 0.4, 1.6, 0)
            set(PoseLandmarks.rightKnee, 0.3 + sin(phase) * 0.4, 1.6, 0)
            set(PoseLandmarks.leftAnkle, 0, 2.2, 0)
            set(PoseLandmarks.rightAnkle, 0.3, 2.2, 0)
            set(PoseLandmarks.leftElbow, -0.2, 0.5, 0)
            set(PoseLandmarks.rightElbow, 0.5, 0.5, 0)
            set(PoseLandmarks.leftWrist, -0.3, 1.0, 0)
            set(PoseLandmarks.rightWrist, 0.6, 1.0, 0)
            window.append(frame)
        }

        let summary = Kinematics.windowSummary(window)
        XCTAssertEqual(summary.count, PoseLandmarks.angleTriplets.count)
        guard let leftKnee = summary.first(where: { $0.joint == "left_knee" }) else {
            return XCTFail("left_knee missing from window summary")
        }
        XCTAssertTrue(leftKnee.rom.isDefined)
        XCTAssertGreaterThan(leftKnee.rom.rangeDeg, 1.0, "the knee visibly moves")

        XCTAssertTrue(Kinematics.windowSummary([]).isEmpty)
    }

    // MARK: Synthesis

    func testDetectRepTroughsFindsOneTroughPerCycle() {
        // Four clean cycles of a 90°-amplitude flexion signal.
        let cycles = 4
        let perCycle = 40
        let series = (0..<(cycles * perCycle)).map { i -> Double in
            160.0 - 45.0 * (1.0 - cos(Double(i) * 2 * .pi / Double(perCycle)))
        }
        let troughs = Synthesis.detectRepTroughs(series)
        XCTAssertEqual(troughs.count, cycles, "expected one trough per cycle, got \(troughs)")
    }

    func testDetectRepTroughsIgnoresSubHysteresisNoise() {
        // ±3° wobble, well under the 8° hysteresis.
        let series = (0..<200).map { 120.0 + 3.0 * sin(Double($0) * 0.4) }
        XCTAssertTrue(Synthesis.detectRepTroughs(series).isEmpty)
    }

    func testSynthesizeProducesPerRepBreakdown() {
        var log: [SessionLogEntry] = []
        let perCycle = 40
        for i in 0..<(3 * perCycle) {
            let angle = 160.0 - 45.0 * (1.0 - cos(Double(i) * 2 * .pi / Double(perCycle)))
            log.append(SessionLogEntry(
                frameIndex: i,
                timestampMs: Double(i) * 33.3,
                similarity: nil,
                // The elbow barely moves, so the knee must win "primary".
                angles: ["left_knee": angle, "left_elbow": 90.0 + 0.5 * sin(Double(i))]
            ))
        }

        let result = Synthesis.synthesize(log, qualitySamples: [0.8, 0.9, 0.85])
        XCTAssertEqual(result.primaryJoint, "left_knee")
        XCTAssertEqual(result.repCountByAngle, 3)
        XCTAssertFalse(result.perRepROM.isEmpty)
        XCTAssertFalse(result.romCurve.isEmpty)
        XCTAssertEqual(result.stabilityTrend.count, result.perRepROM.count)
        XCTAssertNotNil(result.fatigueIndex)
        // No similarity anywhere in the log, so accuracy falls back to
        // scaled mean quality: mean([0.8, 0.9, 0.85]) * 100 = 85.
        assertClose(result.overallAccuracy ?? .nan, 85.0, tolerance: 1e-6, "fallback accuracy")
    }

    func testSynthesizePrefersSimilarityOverQualityFallback() {
        let log = (0..<10).map { i in
            SessionLogEntry(frameIndex: i, timestampMs: Double(i) * 33.3,
                            similarity: 91.0, angles: ["left_knee": 120.0])
        }
        let result = Synthesis.synthesize(log, qualitySamples: [0.1])
        assertClose(result.overallAccuracy ?? .nan, 91.0, tolerance: 1e-6,
                    "similarity wins when a reference is loaded")
    }

    func testSynthesizeOnEmptyLog() {
        let result = Synthesis.synthesize([], qualitySamples: [])
        XCTAssertNil(result.primaryJoint)
        XCTAssertNil(result.overallAccuracy)
        XCTAssertEqual(result.repCountByAngle, 0)
    }

    // MARK: Rep counter

    func testRepCounterCountsHipCycles() {
        var counter = RepCounter()
        let cycles = 5
        let perCycle = 30
        for i in 0..<(cycles * perCycle) {
            // Hip y in screen-normalised units: 0.5 ± 0.15 — comfortably
            // above the 0.06 excursion threshold.
            let y = 0.5 + 0.15 * Float(sin(Double(i) * 2 * .pi / Double(perCycle)))
            counter.update(hipY: y)
        }
        // The hysteresis commits a rep on the way back OUT of a trough, so
        // the final partial cycle is not yet counted.
        XCTAssertGreaterThanOrEqual(counter.count, cycles - 1)
        XCTAssertLessThanOrEqual(counter.count, cycles)
    }

    func testRepCounterIgnoresSway() {
        var counter = RepCounter()
        for i in 0..<300 {
            counter.update(hipY: 0.5 + 0.01 * Float(sin(Double(i) * 0.3)))
        }
        XCTAssertEqual(counter.count, 0, "postural sway must not count as reps")
    }

    func testRepCounterHipYRequiresBothHips() {
        XCTAssertNil(RepCounter.hipY(["left_hip": SIMD4(0, 0.5, 0, 1)]))
        let y = RepCounter.hipY([
            "left_hip": SIMD4(0, 0.4, 0, 1),
            "right_hip": SIMD4(0, 0.6, 0, 1),
        ])
        XCTAssertNotNil(y)
        assertClose(Double(y ?? .nan), 0.5, tolerance: 1e-6, "hip midpoint y")
    }

    func testRepCounterResetClearsState() {
        var counter = RepCounter()
        for i in 0..<60 {
            counter.update(hipY: 0.5 + 0.15 * Float(sin(Double(i) * 0.2)))
        }
        counter.reset()
        XCTAssertEqual(counter.count, 0)
        XCTAssertEqual(counter.stabilityScore, 0)
    }

    // MARK: Similarity

    func testCosineSimilarityMatchesPython() {
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [2, 4, 6, 8]
        assertClose(ActionClassifier.cosineSimilarity(a, b), 1.0, tolerance: 1e-9,
                    "parallel vectors")
        assertClose(ActionClassifier.cosineSimilarity(a, [0, 0, 0, 0]), 0.0,
                    tolerance: 1e-12, "zero vector guard")

        // Negative similarity is clipped to 0 — see similarity.py.
        let opposite: [Float] = [-1, -2, -3, -4]
        assertClose(ActionClassifier.similarityScore(current: a, target: opposite) ?? .nan,
                    0.0, tolerance: 1e-9, "negative similarity clips to zero")
        XCTAssertNil(ActionClassifier.similarityScore(current: a, target: nil))
        assertClose(ActionClassifier.similarityScore(current: a, target: b) ?? .nan,
                    100.0, tolerance: 1e-6, "identical direction scores 100")
    }

    func testSoftmaxIsStableForLargeLogits() {
        let probs = ActionClassifier.softmax([1000, 1000])
        assertClose(Double(probs[0]), 0.5, tolerance: 1e-6, "no overflow")
        assertClose(Double(probs.reduce(0, +)), 1.0, tolerance: 1e-6, "sums to one")
    }
}
