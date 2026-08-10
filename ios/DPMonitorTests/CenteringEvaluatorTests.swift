//
//  CenteringEvaluatorTests.swift
//  DPMonitorTests
//
//  Pins CenteringEvaluator against golden values generated from
//  backend/analyzer/centering.py, which is itself verified byte-for-byte
//  against the desktop reference Final/centering_logic.py.
//
//  Message strings are compared EXACTLY. They are part of the contract:
//  the desktop app, the JSONL recordings and this app must all describe
//  the same posture with the same words, or a clinician comparing a
//  session log to a screenshot sees a discrepancy that isn't real.
//

import XCTest
import simd
@testable import DPMonitor

final class CenteringEvaluatorTests: XCTestCase {

    private var fixtures: ReferenceFixtures!

    override func setUpWithError() throws {
        fixtures = try ReferenceFixtures.load()
    }

    // MARK: Thresholds

    /// The constants must match Python exactly. A drifted threshold still
    /// produces plausible-looking guidance, so nothing else would notice.
    func testThresholdsMatchPython() throws {
        let t = try fixtures.object("centering_thresholds")
        assertClose(Double(CenteringEvaluator.hipXMin), t.double("hip_x_min"),
                    tolerance: 1e-9, "hip_x_min")
        assertClose(Double(CenteringEvaluator.hipXMax), t.double("hip_x_max"),
                    tolerance: 1e-9, "hip_x_max")
        assertClose(Double(CenteringEvaluator.shoulderXMin), t.double("shoulder_x_min"),
                    tolerance: 1e-9, "shoulder_x_min")
        assertClose(Double(CenteringEvaluator.shoulderXMax), t.double("shoulder_x_max"),
                    tolerance: 1e-9, "shoulder_x_max")
        assertClose(Double(CenteringEvaluator.headClipY), t.double("head_clip_y"),
                    tolerance: 1e-9, "head_clip_y")
        assertClose(Double(CenteringEvaluator.noseMinVisibility),
                    t.double("nose_min_visibility"), tolerance: 1e-9,
                    "nose_min_visibility")
        assertClose(Double(CenteringEvaluator.kneeClipY), t.double("knee_clip_y"),
                    tolerance: 1e-9, "knee_clip_y")
        assertClose(Double(CenteringEvaluator.torsoRatioMin), t.double("torso_ratio_min"),
                    tolerance: 1e-9, "torso_ratio_min")
        assertClose(Double(CenteringEvaluator.torsoRatioMax), t.double("torso_ratio_max"),
                    tolerance: 1e-9, "torso_ratio_max")
    }

    /// The nose gate is 0.3, NOT the 0.5 occlusion threshold used
    /// elsewhere. Easy to "tidy up" into a shared constant and break.
    func testNoseVisibilityGateIsDistinctFromOcclusionThreshold() {
        XCTAssertNotEqual(CenteringEvaluator.noseMinVisibility,
                          OcclusionHandler.defaultThreshold,
                          "nose gate is 0.3 by design; occlusion carry-forward is 0.5")
    }

    // MARK: Full parity

    func testCenteringMatchesPython() throws {
        let cases = try fixtures.cases("centering")
        XCTAssertFalse(cases.isEmpty)

        for c in cases {
            let name = c.string("name") ?? "?"
            guard let raw = c["points"] as? [String: Any],
                  let expected = c["expected"] as? [String: Any] else {
                XCTFail("malformed centering case \(name)")
                continue
            }

            let points = landmarks(from: raw)
            let actual = CenteringEvaluator.evaluate(points)

            XCTAssertEqual(actual.message, expected.string("status") ?? "",
                           "centering[\(name)].message")
            XCTAssertEqual(actual.status.rawValue, expected.string("status_code") ?? "",
                           "centering[\(name)].status")
            XCTAssertEqual(actual.severity.rawValue, expected.string("severity") ?? "",
                           "centering[\(name)].severity")
            XCTAssertEqual(actual.isCentered, (expected["is_centered"] as? Bool) ?? false,
                           "centering[\(name)].isCentered")
            XCTAssertEqual(actual.details, (expected["details"] as? [String]) ?? [],
                           "centering[\(name)].details")

            assertOptionalClose(actual.hipCenterX, expected["hip_center_x"],
                                "centering[\(name)].hipCenterX")
            assertOptionalClose(actual.shoulderCenterX, expected["shoulder_center_x"],
                                "centering[\(name)].shoulderCenterX")
            assertOptionalClose(actual.torsoHeightRatio, expected["torso_height_ratio"],
                                "centering[\(name)].torsoHeightRatio")
        }
    }

    // MARK: Behavioural invariants

    /// `isCentered` must agree with the message. A green banner over
    /// "Patient too far LEFT" would be worse than no banner at all.
    func testCenteredImpliesCenteredMessageAndSeverity() throws {
        for c in try fixtures.cases("centering") {
            guard let raw = c["points"] as? [String: Any] else { continue }
            let r = CenteringEvaluator.evaluate(landmarks(from: raw))
            if r.isCentered {
                XCTAssertEqual(r.message, "Patient is CENTERED", c.string("name") ?? "")
                XCTAssertEqual(r.status, .centered)
                XCTAssertEqual(r.severity, .ok)
            } else {
                XCTAssertNotEqual(r.status, .centered, c.string("name") ?? "")
                XCTAssertNotEqual(r.severity, .ok, c.string("name") ?? "")
            }
        }
    }

    /// Torso height is nose-to-hip. Measuring shoulder-to-hip roughly
    /// halves it and silently invalidates both distance thresholds.
    func testTorsoRatioIsNoseToHipNotShoulderToHip() {
        let pose = pose(hipX: 0.5, noseY: 0.10, hipY: 0.60, shoulderY: 0.32)
        let r = CenteringEvaluator.evaluate(pose)
        assertClose(Double(r.torsoHeightRatio ?? .nan), 0.50, tolerance: 1e-5,
                    "nose(0.10) -> hip(0.60) = 0.50")
        // Shoulder-to-hip would have been 0.28.
        XCTAssertNotEqual(r.torsoHeightRatio ?? 0, 0.28, accuracy: 1e-5)
    }

    func testMissingRequiredLandmarkYieldsNotDetected() {
        for missing in CenteringEvaluator.requiredLandmarks {
            var p = pose(hipX: 0.5)
            p.removeValue(forKey: missing)
            let r = CenteringEvaluator.evaluate(p)
            XCTAssertEqual(r.status, .notDetected, "missing \(missing)")
            XCTAssertFalse(r.isCentered)
            XCTAssertEqual(r.message, "Patient has left the field of view")
            XCTAssertEqual(r.details, ["Pose not recognised."])
            XCTAssertNil(r.hipCenterX)
        }
        XCTAssertEqual(CenteringEvaluator.evaluate([:]).status, .notDetected)
    }

    /// The arrow points the way out, not the way in.
    func testCorrectionDirectionOpposesDisplacement() {
        XCTAssertEqual(CenteringEvaluator.evaluate(pose(hipX: 0.10)).status, .moveRight)
        XCTAssertEqual(CenteringEvaluator.evaluate(pose(hipX: 0.90)).status, .moveLeft)
    }

    /// Hip position is evaluated first, so it wins the headline even when
    /// later checks also fail.
    func testHipIssueOutranksLaterChecks() {
        let p = pose(hipX: 0.05, noseY: 0.50, hipY: 0.58, kneeY: 0.99)
        let r = CenteringEvaluator.evaluate(p)
        XCTAssertEqual(r.message, "Patient too far LEFT")
        XCTAssertGreaterThan(r.details.count, 2, "all findings should be listed")
        XCTAssertTrue(r.details.contains("Patient FEET may be cut off"))
        XCTAssertTrue(r.details.contains("Patient is TOO FAR from camera"))
        // Last line is always the hip readout.
        XCTAssertTrue(r.details.last?.hasPrefix("Hip center:") ?? false)
    }

    func testPercentFormattingMatchesPython() {
        XCTAssertEqual(CenteringEvaluator.percent(0.5), "50%")
        XCTAssertEqual(CenteringEvaluator.percent(0.18), "18%")
        XCTAssertEqual(CenteringEvaluator.percent(0.0), "0%")
        XCTAssertEqual(CenteringEvaluator.percent(1.0), "100%")
    }

    // MARK: Helpers

    private func landmarks(from raw: [String: Any]) -> [String: SIMD4<Float>] {
        var out = [String: SIMD4<Float>](minimumCapacity: raw.count)
        for (name, value) in raw {
            guard let v = value as? [Any] else { continue }
            let f = v.map { (($0 as? NSNumber)?.floatValue) ?? 0 }
            guard f.count >= 4 else { continue }
            out[name] = SIMD4(f[0], f[1], f[2], f[3])
        }
        return out
    }

    private func pose(hipX: Float,
                      noseY: Float = 0.10,
                      hipY: Float = 0.60,
                      shoulderY: Float = 0.32,
                      kneeY: Float = 0.80,
                      noseVis: Float = 0.9) -> [String: SIMD4<Float>] {
        let w: Float = 0.09
        return [
            "nose":           SIMD4(hipX, noseY, 0, noseVis),
            "left_shoulder":  SIMD4(hipX - w, shoulderY, 0, 0.9),
            "right_shoulder": SIMD4(hipX + w, shoulderY, 0, 0.9),
            "left_hip":       SIMD4(hipX - w, hipY, 0, 0.9),
            "right_hip":      SIMD4(hipX + w, hipY, 0, 0.9),
            "left_knee":      SIMD4(hipX - w, kneeY, 0, 0.9),
            "right_knee":     SIMD4(hipX + w, kneeY, 0, 0.9),
        ]
    }

    private func assertOptionalClose(_ actual: Float?, _ expected: Any?,
                                     _ label: String,
                                     file: StaticString = #filePath,
                                     line: UInt = #line) {
        if expected == nil || expected is NSNull {
            XCTAssertNil(actual, "\(label): expected nil", file: file, line: line)
            return
        }
        guard let actual else {
            XCTFail("\(label): expected a value, got nil", file: file, line: line)
            return
        }
        let e = (expected as? NSNumber)?.doubleValue ?? .nan
        assertClose(Double(actual), e, tolerance: kParityTolerance, label,
                    file: file, line: line)
    }
}
