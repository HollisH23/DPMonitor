//
//  PoseNormalizer.swift
//  DPMonitor
//
//  Spatial normalisation. Exact port of
//  backend/analyzer/normalization.py :: normalize_pose.
//
//  Two nuisance factors change with camera placement but say nothing about
//  the movement itself:
//
//    * translation — where in the frame the patient stands
//    * scale       — how far they are from the camera
//
//  Both are stripped per frame:
//    1. subtract the hip midpoint, so the pelvis sits at the origin
//    2. divide by the hip→shoulder (spine) length, so the torso is ~1 unit
//
//  Degenerate cases return early rather than throwing, exactly as the
//  Python does: a live session must keep streaming even when landmarks
//  drop out. The branch ORDER matters and is asserted by the parity tests —
//  in particular the shoulder presence check runs on the ALREADY-CENTRED
//  pose, so a shoulder sitting exactly on the hip midpoint is treated as
//  "missing" rather than as a short spine.
//

import Accelerate
import Foundation
import simd

enum PoseNormalizer {

    static let leftHipIdx = PoseLandmarks.leftHip           // 23
    static let rightHipIdx = PoseLandmarks.rightHip         // 24
    static let leftShoulderIdx = PoseLandmarks.leftShoulder // 11
    static let rightShoulderIdx = PoseLandmarks.rightShoulder // 12

    /// Below this spine length the rescale would amplify numerical noise
    /// into huge coordinates. Matches `_MIN_SPINE_LENGTH` in Python.
    static let minSpineLength: Float = 1e-3

    private static let V = PoseLandmarks.count       // 33
    private static let C = PoseLandmarks.channels    // 3

    /// Normalise a 33×3 row-major pose in place.
    ///
    /// - Parameter coords: 99 floats, joint-major: joint `v` channel `c`
    ///   lives at `v * 3 + c`.
    static func normalize(_ coords: inout [Float]) {
        precondition(coords.count == V * C, "expected \(V * C) floats, got \(coords.count)")

        coords.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }

            // --- 1. Hip midpoint -------------------------------------
            guard let hipMid = midpoint(base, leftHipIdx, rightHipIdx) else {
                // Both hips missing: nothing trustworthy to centre on.
                return
            }

            // --- 2. Centre ------------------------------------------
            // Strided add of -hipMid[c] across all 33 joints, one channel
            // at a time. vDSP handles the stride-3 walk natively, so this
            // is three vector ops rather than 99 scalar subtractions.
            var negX = -hipMid.x, negY = -hipMid.y, negZ = -hipMid.z
            vDSP_vsadd(base + 0, C, &negX, base + 0, C, vDSP_Length(V))
            vDSP_vsadd(base + 1, C, &negY, base + 1, C, vDSP_Length(V))
            vDSP_vsadd(base + 2, C, &negZ, base + 2, C, vDSP_Length(V))

            // --- 3. Shoulder midpoint, measured on the CENTRED pose ---
            guard let shoulderMid = midpoint(base, leftShoulderIdx, rightShoulderIdx) else {
                // Shoulders missing: keep the centred pose, skip the rescale.
                return
            }

            // --- 4. Spine length -------------------------------------
            // The centred shoulder midpoint IS the hip→shoulder vector,
            // so its magnitude is the spine length. `simd_length` compiles
            // to a single vector instruction — no need to reach for BLAS
            // for a 3-element norm.
            let spine = simd_length(shoulderMid)
            guard spine >= minSpineLength else {
                // Degenerate: shoulders ≈ hips. Centre only.
                return
            }

            // --- 5. Rescale ------------------------------------------
            var divisor = spine
            vDSP_vsdiv(base, 1, &divisor, base, 1, vDSP_Length(V * C))
        }
    }

    /// Convenience wrapper returning a new array.
    static func normalized(_ coords: [Float]) -> [Float] {
        var copy = coords
        normalize(&copy)
        return copy
    }

    /// Normalise a whole `(T, V, C)` window — port of `normalize_window`.
    static func normalizeWindow(_ frames: [[Float]]) -> [[Float]] {
        frames.map { normalized($0) }
    }

    // MARK: - Midpoint

    /// Midpoint of two landmarks, or `nil` when both are absent.
    ///
    /// Port of `normalization._midpoint`. A landmark counts as "missing"
    /// when its x and y are both exactly zero — the sentinel the upstream
    /// stacking code writes for an absent point. When only one side is
    /// missing we fall back to the present one; when both are missing the
    /// caller must short-circuit.
    @inline(__always)
    private static func midpoint(_ base: UnsafeMutablePointer<Float>,
                                 _ idxA: Int,
                                 _ idxB: Int) -> SIMD3<Float>? {
        let a = SIMD3(base[idxA * C], base[idxA * C + 1], base[idxA * C + 2])
        let b = SIMD3(base[idxB * C], base[idxB * C + 1], base[idxB * C + 2])

        let aPresent = !(a.x == 0.0 && a.y == 0.0)
        let bPresent = !(b.x == 0.0 && b.y == 0.0)

        if aPresent && bPresent { return 0.5 * (a + b) }
        if aPresent { return a }
        if bPresent { return b }
        return nil
    }
}
