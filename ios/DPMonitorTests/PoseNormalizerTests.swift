//
//  PoseNormalizerTests.swift
//  DPMonitorTests
//
//  Parity against the Python reference for everything that touches the
//  model's input tensor: normalisation, occlusion carry-forward, EMA
//  smoothing, the landmark ordering itself and the (T,V,C) -> (1,C,T,V,1)
//  repack.
//
//  These are the tests that matter most. A drift here does not crash — it
//  quietly feeds the CTR-GCN a distribution it was never configured for,
//  and the quality score becomes meaningless while still looking plausible.
//

import CoreML
import XCTest
import simd
@testable import DPMonitor

final class PoseNormalizerTests: XCTestCase {

    private var fixtures: ReferenceFixtures!

    override func setUpWithError() throws {
        fixtures = try ReferenceFixtures.load()
    }

    // MARK: Topology

    /// The 33-name ordering is the contract between the Swift app, the
    /// Python analyser and the exported adjacency matrix.
    func testLandmarkOrderingMatchesPython() throws {
        XCTAssertEqual(PoseLandmarks.names, fixtures.landmarkNames)
        XCTAssertEqual(PoseLandmarks.count, 33)
        XCTAssertEqual(PoseLandmarks.names[PoseLandmarks.leftHip], "left_hip")
        XCTAssertEqual(PoseLandmarks.names[PoseLandmarks.rightHip], "right_hip")
        XCTAssertEqual(PoseLandmarks.names[PoseLandmarks.leftShoulder], "left_shoulder")
        XCTAssertEqual(PoseLandmarks.names[PoseLandmarks.rightShoulder], "right_shoulder")
    }

    func testAngleTripletsMatchPython() throws {
        let expected = try fixtures.object("joint_angle_triplets")
        XCTAssertEqual(PoseLandmarks.angleTriplets.count, expected.count)
        for triplet in PoseLandmarks.angleTriplets {
            guard let names = expected[triplet.name] as? [String] else {
                XCTFail("missing triplet \(triplet.name)")
                continue
            }
            XCTAssertEqual(PoseLandmarks.names[triplet.a], names[0], triplet.name)
            XCTAssertEqual(PoseLandmarks.names[triplet.vertex], names[1], triplet.name)
            XCTAssertEqual(PoseLandmarks.names[triplet.b], names[2], triplet.name)
        }
    }

    // MARK: Normalisation

    func testNormalizeMatchesPython() throws {
        let cases = try fixtures.cases("normalize")
        XCTAssertFalse(cases.isEmpty)

        for c in cases {
            let name = c.string("name") ?? "?"
            guard let input = c.matrixFlattened("input"),
                  let expected = c.matrixFlattened("expected") else {
                XCTFail("malformed normalize case \(name)")
                continue
            }
            var actual = input
            PoseNormalizer.normalize(&actual)
            assertClose(actual, expected, "normalize[\(name)]")
        }
    }

    /// After a successful normalisation the hip midpoint sits at the origin
    /// and the spine has unit length. Checking the invariant as well as the
    /// values catches the case where Swift and Python are equally wrong.
    func testNormalizeInvariants() throws {
        for c in try fixtures.cases("normalize")
        where (c.string("name") ?? "").hasPrefix("plausible") {
            guard let input = c.matrixFlattened("input") else { continue }
            var out = input
            PoseNormalizer.normalize(&out)

            let hipMid = midpoint(out, PoseLandmarks.leftHip, PoseLandmarks.rightHip)
            assertClose(Double(simd_length(hipMid)), 0.0, tolerance: 1e-5, "hip at origin")

            let shoulderMid = midpoint(out, PoseLandmarks.leftShoulder, PoseLandmarks.rightShoulder)
            assertClose(Double(simd_length(shoulderMid)), 1.0, tolerance: 1e-5, "unit spine")
        }
    }

    private func midpoint(_ coords: [Float], _ a: Int, _ b: Int) -> SIMD3<Float> {
        let pa = SIMD3(coords[a * 3], coords[a * 3 + 1], coords[a * 3 + 2])
        let pb = SIMD3(coords[b * 3], coords[b * 3 + 1], coords[b * 3 + 2])
        return 0.5 * (pa + pb)
    }

    // MARK: Occlusion carry-forward

    func testOcclusionCarryForwardMatchesPython() throws {
        let cases = try fixtures.cases("occlusion")
        XCTAssertFalse(cases.isEmpty)

        for c in cases {
            let name = c.string("name") ?? "?"
            guard let normed = c.matrixFlattened("normed"),
                  let visibility = c.floats("visibility"),
                  let expected = c.matrixFlattened("expected") else {
                XCTFail("malformed occlusion case \(name)")
                continue
            }
            let prev = c.matrixFlattened("prev")   // nil for the first-frame case
            let threshold = Float(c.double("threshold"))

            let actual = OcclusionHandler.applied(normed,
                                                  visibility: visibility,
                                                  prev: prev,
                                                  threshold: threshold)
            assertClose(actual, expected, "occlusion[\(name)]")
        }
    }

    /// The stateful path must agree with the stateless one: frame 1 seeds,
    /// frame 2 carries forward from frame 1.
    func testOcclusionHandlerStatefulSequence() throws {
        var handler = OcclusionHandler(threshold: 0.5)

        var first = [Float](repeating: 1.0, count: 99)
        let allVisible = [Float](repeating: 0.9, count: 33)
        handler.apply(&first, visibility: allVisible)
        XCTAssertEqual(first, [Float](repeating: 1.0, count: 99),
                       "first frame must pass through untouched")

        var second = [Float](repeating: 5.0, count: 99)
        var visibility = [Float](repeating: 0.9, count: 33)
        visibility[7] = 0.1      // occlude joint 7 only
        handler.apply(&second, visibility: visibility)

        XCTAssertEqual(second[7 * 3], 1.0, "occluded joint carries forward")
        XCTAssertEqual(second[7 * 3 + 1], 1.0)
        XCTAssertEqual(second[7 * 3 + 2], 1.0)
        XCTAssertEqual(second[6 * 3], 5.0, "visible joint keeps its new value")

        handler.reset()
        var third = [Float](repeating: 9.0, count: 99)
        handler.apply(&third, visibility: [Float](repeating: 0.0, count: 33))
        XCTAssertEqual(third, [Float](repeating: 9.0, count: 99),
                       "reset must clear the carry-forward state")
    }

    // MARK: EMA smoothing

    func testEMASmootherMatchesJSReference() throws {
        let cases = try fixtures.cases("ema")
        let names = fixtures.landmarkNames

        for c in cases {
            let label = c.string("name") ?? "?"
            guard let frames = c["frames"] as? [[[Any]]],
                  let expectedFrames = c["expected"] as? [[[Any]]] else {
                XCTFail("malformed ema case \(label)")
                continue
            }
            XCTAssertEqual(frames.count, expectedFrames.count)

            var smoother = PoseSmoother(
                alpha: Float(c.double("alpha")),
                occlusionThreshold: Float(c.double("occlusion_threshold")),
                occludedAlpha: Float(c.double("occluded_alpha"))
            )

            for (t, rawRows) in frames.enumerated() {
                let raw = landmarks(from: rawRows, names: names)
                let expected = landmarks(from: expectedFrames[t], names: names)
                let actual = smoother.smooth(raw)

                for name in names {
                    guard let a = actual[name], let e = expected[name] else {
                        XCTFail("ema[\(label)] frame \(t): missing \(name)")
                        continue
                    }
                    assertClose([a.x, a.y, a.z, a.w], [e.x, e.y, e.z, e.w],
                                "ema[\(label)] frame \(t) \(name)")
                }
            }
        }
    }

    /// The occluded branch must actually engage — otherwise the previous
    /// test would pass with a single hard-coded alpha.
    func testEMAUsesReducedAlphaWhenOccluded() {
        var smoother = PoseSmoother(alpha: 0.6, occlusionThreshold: 0.5, occludedAlpha: 0.15)
        let name = "left_wrist"

        _ = smoother.smooth([name: SIMD4<Float>(0, 0, 0, 1.0)])
        let occluded = smoother.smooth([name: SIMD4<Float>(10, 0, 0, 0.1)])
        // 0.15 * 10 + 0.85 * 0 = 1.5
        assertClose(Double(occluded[name]!.x), 1.5, tolerance: 1e-5, "occluded alpha")

        var visibleSmoother = PoseSmoother(alpha: 0.6, occlusionThreshold: 0.5, occludedAlpha: 0.15)
        _ = visibleSmoother.smooth([name: SIMD4<Float>(0, 0, 0, 1.0)])
        let visible = visibleSmoother.smooth([name: SIMD4<Float>(10, 0, 0, 0.9)])
        // 0.6 * 10 + 0.4 * 0 = 6.0
        assertClose(Double(visible[name]!.x), 6.0, tolerance: 1e-5, "base alpha")
    }

    private func landmarks(from rows: [[Any]], names: [String]) -> [String: SIMD4<Float>] {
        var out = [String: SIMD4<Float>](minimumCapacity: rows.count)
        for (i, row) in rows.enumerated() where i < names.count {
            let v = row.map { (($0 as? NSNumber)?.floatValue) ?? 0 }
            guard v.count >= 4 else { continue }
            out[names[i]] = SIMD4(v[0], v[1], v[2], v[3])
        }
        return out
    }

    // MARK: PoseFrame stacking

    /// Port check for `CTRGCNAnalyzer._stack_frame`: names absent from the
    /// dictionary must land as zeros with zero visibility, not be skipped
    /// in a way that shifts every later joint by one slot.
    func testPoseFrameStackingHandlesMissingJoints() {
        var dict: [String: SIMD4<Float>] = [:]
        dict["nose"] = SIMD4(1, 2, 3, 0.9)
        dict["right_foot_index"] = SIMD4(7, 8, 9, 0.4)

        let frame = PoseFrame(dictionary: dict)
        XCTAssertEqual(frame.coords.count, 99)
        XCTAssertEqual(frame.coords[0], 1)
        XCTAssertEqual(frame.coords[1], 2)
        XCTAssertEqual(frame.coords[2], 3)
        XCTAssertEqual(frame.visibility[0], 0.9)

        XCTAssertEqual(frame.coords[32 * 3], 7)
        XCTAssertEqual(frame.visibility[32], 0.4)

        // Everything in between is the zero sentinel.
        XCTAssertEqual(frame.coords[15 * 3], 0)
        XCTAssertEqual(frame.visibility[15], 0)
    }

    // MARK: Tensor packing

    /// Pins `dstIdx = c*T*V + t*V + v` against probe indices computed by
    /// NumPy from the same ramp input.
    func testFrameBufferTensorLayoutMatchesNumpy() throws {
        let layout = try fixtures.object("tensor_layout")
        guard let sourceFlat = layout.floats("source_flat_TVC"),
              let probeIndices = layout.doubles("probe_indices")?.map({ Int($0) }),
              let probeValues = layout.doubles("probe_values") else {
            throw FixtureError.malformed("tensor_layout")
        }

        let T = 64, V = 33, C = 3
        XCTAssertEqual(sourceFlat.count, T * V * C)

        let buffer = FrameBuffer(windowSize: T)
        for t in 0..<T {
            let start = t * V * C
            buffer.append(Array(sourceFlat[start..<(start + V * C)]))
        }
        XCTAssertTrue(buffer.isFull)

        let tensor = try buffer.makeInputTensor()
        XCTAssertEqual(tensor.shape.map(\.intValue), [1, C, T, V, 1])
        XCTAssertEqual(tensor.count, T * V * C)

        var flat = [Float](repeating: 0, count: tensor.count)
        tensor.withUnsafeBufferPointer(ofType: Float.self) { ptr, _ in
            for i in 0..<min(flat.count, ptr.count) { flat[i] = ptr[i] }
        }

        for (probe, expected) in zip(probeIndices, probeValues) {
            assertClose(Double(flat[probe]), expected, tolerance: 1e-6,
                        "tensor_layout[\(probe)]")
        }
    }

    func testFrameBufferSlidesAndRejectsPartialWindow() {
        let buffer = FrameBuffer(windowSize: 4)
        let frame = [Float](repeating: 0, count: 99)

        for _ in 0..<3 { buffer.append(frame) }
        XCTAssertFalse(buffer.isFull)
        XCTAssertThrowsError(try buffer.makeInputTensor())

        for _ in 0..<10 { buffer.append(frame) }
        XCTAssertTrue(buffer.isFull)
        XCTAssertEqual(buffer.count, 4, "window must not grow past its size")

        buffer.reset()
        XCTAssertEqual(buffer.count, 0)
    }
}
