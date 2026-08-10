//
//  ReferenceFixtures.swift
//  DPMonitorTests
//
//  Loads Fixtures/reference_fixtures.json — golden values produced by the
//  *actual* Python reference implementations via
//  `scripts/gen_reference_fixtures.py`.
//
//  Hand-written expectations would drift silently the first time someone
//  edits normalization.py. Generated ones fail loudly instead: regenerate
//  the fixture, watch the test go red, and decide whether the change was
//  intended.
//
//  NaN crosses the JSON boundary as `null`, because JSON has no NaN. The
//  decoding helpers below turn `null` back into `.nan`.
//

import Foundation
import XCTest

/// Tolerance from the fixture header. float32 Swift vs float32/float64
/// NumPy leaves ~1e-7 of slack; 1e-4 is comfortably above the noise and
/// still tight enough to catch a genuine algorithmic divergence.
let kParityTolerance: Double = 1e-4

enum FixtureError: Error, CustomStringConvertible {
    case notFound
    case malformed(String)

    var description: String {
        switch self {
        case .notFound:
            return """
            reference_fixtures.json is missing from the test bundle. \
            Run `python scripts/gen_reference_fixtures.py` and regenerate \
            the project with `xcodegen generate`.
            """
        case .malformed(let key):
            return "reference_fixtures.json is missing or malformed at '\(key)'"
        }
    }
}

struct ReferenceFixtures {

    let root: [String: Any]

    static func load() throws -> ReferenceFixtures {
        let bundle = Bundle(for: BundleToken.self)
        guard let url = bundle.url(forResource: "reference_fixtures", withExtension: "json")
                ?? bundle.url(forResource: "reference_fixtures",
                              withExtension: "json",
                              subdirectory: "Fixtures") else {
            throw FixtureError.notFound
        }
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw FixtureError.malformed("<root>")
        }
        return ReferenceFixtures(root: root)
    }

    func cases(_ key: String) throws -> [[String: Any]] {
        guard let list = root[key] as? [[String: Any]] else {
            throw FixtureError.malformed(key)
        }
        return list
    }

    func object(_ key: String) throws -> [String: Any] {
        guard let obj = root[key] as? [String: Any] else {
            throw FixtureError.malformed(key)
        }
        return obj
    }

    var landmarkNames: [String] { root["landmark_names"] as? [String] ?? [] }

    private final class BundleToken {}
}

// MARK: - Decoding helpers

extension Dictionary where Key == String, Value == Any {

    /// A JSON number, with `null` decoded as NaN.
    func double(_ key: String) -> Double {
        guard let raw = self[key] else { return .nan }
        if raw is NSNull { return .nan }
        return (raw as? NSNumber)?.doubleValue ?? .nan
    }

    func int(_ key: String) -> Int? {
        (self[key] as? NSNumber)?.intValue
    }

    func string(_ key: String) -> String? {
        self[key] as? String
    }

    /// A flat `[Double]`, with `null` entries decoded as NaN.
    func doubles(_ key: String) -> [Double]? {
        guard let raw = self[key] as? [Any] else { return nil }
        return raw.map { $0 is NSNull ? Double.nan : ((($0 as? NSNumber)?.doubleValue) ?? .nan) }
    }

    func floats(_ key: String) -> [Float]? {
        doubles(key).map { $0.map(Float.init) }
    }

    /// A `(V, C)` matrix flattened row-major, matching `PoseFrame.coords`.
    func matrixFlattened(_ key: String) -> [Float]? {
        guard let rows = self[key] as? [[Any]] else { return nil }
        var out = [Float]()
        out.reserveCapacity(rows.count * 3)
        for row in rows {
            for v in row {
                out.append(v is NSNull ? Float.nan : (((v as? NSNumber)?.floatValue) ?? .nan))
            }
        }
        return out
    }

    /// A `(V, 4)` matrix as a landmark dictionary, for the EMA fixtures.
    func matrixAsLandmarks(_ key: String, names: [String]) -> [String: SIMD4<Float>]? {
        guard let rows = self[key] as? [[Any]] else { return nil }
        var out = [String: SIMD4<Float>](minimumCapacity: rows.count)
        for (i, row) in rows.enumerated() where i < names.count {
            let v = row.map { (($0 as? NSNumber)?.floatValue) ?? 0 }
            guard v.count >= 4 else { continue }
            out[names[i]] = SIMD4(v[0], v[1], v[2], v[3])
        }
        return out
    }
}

// MARK: - Assertions

/// Compare two float arrays, treating NaN as equal to NaN.
///
/// NaN equality matters: a missing landmark must stay missing on both
/// sides. `XCTAssertEqual` on NaN would fail, which would hide the real
/// signal behind noise.
func assertClose(_ actual: [Float],
                 _ expected: [Float],
                 tolerance: Double = kParityTolerance,
                 _ label: String,
                 file: StaticString = #filePath,
                 line: UInt = #line) {
    XCTAssertEqual(actual.count, expected.count, "\(label): length mismatch",
                   file: file, line: line)
    guard actual.count == expected.count else { return }

    var worst: (index: Int, delta: Double) = (-1, 0)
    for i in actual.indices {
        let a = Double(actual[i]), e = Double(expected[i])
        if a.isNaN && e.isNaN { continue }
        let delta = abs(a - e)
        if delta.isNaN || delta > worst.delta { worst = (i, delta.isNaN ? .infinity : delta) }
    }
    if worst.index >= 0 && worst.delta > tolerance {
        XCTFail("""
                \(label): max |Δ| = \(worst.delta) at index \(worst.index) \
                (swift \(actual[worst.index]) vs python \(expected[worst.index])), \
                tolerance \(tolerance)
                """, file: file, line: line)
    }
}

func assertClose(_ actual: Double,
                 _ expected: Double,
                 tolerance: Double = kParityTolerance,
                 _ label: String,
                 file: StaticString = #filePath,
                 line: UInt = #line) {
    if actual.isNaN && expected.isNaN { return }
    if actual.isNaN != expected.isNaN {
        XCTFail("\(label): NaN mismatch (swift \(actual) vs python \(expected))",
                file: file, line: line)
        return
    }
    let delta = abs(actual - expected)
    if delta > tolerance {
        XCTFail("\(label): |Δ| = \(delta) (swift \(actual) vs python \(expected)), tolerance \(tolerance)",
                file: file, line: line)
    }
}
