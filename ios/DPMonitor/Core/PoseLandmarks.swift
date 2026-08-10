//
//  PoseLandmarks.swift
//  DPMonitor
//
//  Single source of truth for the 33-joint MediaPipe Pose topology.
//
//  This MUST stay byte-identical to the Python side:
//    backend/analyzer/mediapipe_graph.py  ->  LANDMARK_NAMES, _EDGE_PAIRS_BY_NAME
//
//  The CTR-GCN model's spatial adjacency matrix is baked from that edge set
//  at export time, so a mismatch here does not throw — it silently feeds the
//  model joints in the wrong slots. `PoseLandmarkTopologyTests` pins the
//  ordering against the generated fixture file.
//

import Foundation
import simd

enum PoseLandmarks {

    /// MediaPipe Pose canonical 33-landmark order.
    static let names: [String] = [
        "nose",                                                   // 0
        "left_eye_inner", "left_eye", "left_eye_outer",           // 1, 2, 3
        "right_eye_inner", "right_eye", "right_eye_outer",        // 4, 5, 6
        "left_ear", "right_ear",                                  // 7, 8
        "mouth_left", "mouth_right",                              // 9, 10
        "left_shoulder", "right_shoulder",                        // 11, 12
        "left_elbow", "right_elbow",                              // 13, 14
        "left_wrist", "right_wrist",                              // 15, 16
        "left_pinky", "right_pinky",                              // 17, 18
        "left_index", "right_index",                              // 19, 20
        "left_thumb", "right_thumb",                              // 21, 22
        "left_hip", "right_hip",                                  // 23, 24
        "left_knee", "right_knee",                                // 25, 26
        "left_ankle", "right_ankle",                              // 27, 28
        "left_heel", "right_heel",                                // 29, 30
        "left_foot_index", "right_foot_index",                    // 31, 32
    ]

    /// V in the CTR-GCN tensor. 33 — *not* NTU-25.
    static let count: Int = 33

    /// C in the CTR-GCN tensor: (x, y, z).
    static let channels: Int = 3

    /// Name -> ordinal, built once.
    static let indexByName: [String: Int] = {
        var map = [String: Int](minimumCapacity: count)
        for (i, n) in names.enumerated() { map[n] = i }
        return map
    }()

    // Hot-path indices used by PoseNormalizer and RepCounter. Hard-coded
    // rather than looked up so the normaliser stays allocation-free.
    static let leftShoulder = 11
    static let rightShoulder = 12
    static let leftHip = 23
    static let rightHip = 24
    static let leftKnee = 25
    static let rightKnee = 26
    static let leftAnkle = 27
    static let rightAnkle = 28
    static let leftElbow = 13
    static let rightElbow = 14
    static let leftWrist = 15
    static let rightWrist = 16

    /// Skeleton edges, mirroring `_EDGE_PAIRS_BY_NAME` in `mediapipe_graph.py`.
    /// Used for the AR overlay; the model's adjacency comes from the export.
    static let bones: [(Int, Int)] = [
        // Face arc
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (0, 9), (0, 10),
        // Shoulders & arms
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        // Hand tips
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
        // Torso
        (11, 23), (12, 24), (23, 24),
        // Head anchored to torso
        (7, 11), (8, 12),
        // Legs
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    ]

    /// Clinical joint-angle definitions — mirrors `JOINT_ANGLE_TRIPLETS`
    /// in `backend/analyzer/kinematics.py`. Each entry is (a, vertex, b);
    /// the reported angle is the one at `vertex`.
    ///
    /// Kept as an ordered array (not a dictionary) so iteration order is
    /// deterministic across runs — the synthesis layer's "primary joint"
    /// tie-break would otherwise be unstable.
    static let angleTriplets: [(name: String, a: Int, vertex: Int, b: Int)] = [
        ("left_knee",   leftHip,      leftKnee,      leftAnkle),
        ("right_knee",  rightHip,     rightKnee,     rightAnkle),
        ("left_elbow",  leftShoulder, leftElbow,     leftWrist),
        ("right_elbow", rightShoulder, rightElbow,   rightWrist),
        ("left_hip",    leftShoulder, leftHip,       leftKnee),
        ("right_hip",   rightShoulder, rightHip,     rightKnee),
    ]
}

// MARK: - PoseFrame

/// One frame of pose data in the flat layout the pipeline uses downstream.
///
/// `coords` is 33 × 3 row-major — joint `v`'s channel `c` lives at
/// `v * 3 + c`. This is exactly the `(V, C)` numpy array that
/// `CTRGCNAnalyzer._stack_frame` builds, so the Swift and Python pipelines
/// index identically.
struct PoseFrame {
    /// 99 floats: (x, y, z) per joint, row-major by joint.
    var coords: [Float]
    /// 33 floats in [0, 1].
    var visibility: [Float]

    init() {
        coords = [Float](repeating: 0, count: PoseLandmarks.count * PoseLandmarks.channels)
        visibility = [Float](repeating: 0, count: PoseLandmarks.count)
    }

    init(coords: [Float], visibility: [Float]) {
        precondition(coords.count == PoseLandmarks.count * PoseLandmarks.channels)
        precondition(visibility.count == PoseLandmarks.count)
        self.coords = coords
        self.visibility = visibility
    }

    /// Flatten a landmark dictionary into the ordered layout.
    ///
    /// Port of `CTRGCNAnalyzer._stack_frame`: a name that is absent from the
    /// dictionary is left as `(0, 0, 0)` with visibility `0`, which is the
    /// same "missing landmark" sentinel the normaliser tests for.
    init(dictionary: [String: SIMD4<Float>]) {
        self.init()
        for (i, name) in PoseLandmarks.names.enumerated() {
            guard let p = dictionary[name] else { continue }
            coords[i * 3 + 0] = p.x
            coords[i * 3 + 1] = p.y
            coords[i * 3 + 2] = p.z
            visibility[i] = p.w
        }
    }

    @inline(__always)
    subscript(joint v: Int) -> SIMD3<Float> {
        get { SIMD3(coords[v * 3], coords[v * 3 + 1], coords[v * 3 + 2]) }
        set {
            coords[v * 3] = newValue.x
            coords[v * 3 + 1] = newValue.y
            coords[v * 3 + 2] = newValue.z
        }
    }

    /// Rebuild the name-keyed representation (used by the overlay layer).
    var dictionary: [String: SIMD4<Float>] {
        var out = [String: SIMD4<Float>](minimumCapacity: PoseLandmarks.count)
        for (i, name) in PoseLandmarks.names.enumerated() {
            out[name] = SIMD4(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2], visibility[i])
        }
        return out
    }
}
