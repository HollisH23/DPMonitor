//
//  FrameBuffer.swift
//  DPMonitor
//
//  64-frame sliding window plus the (T, V, C) -> (1, C, T, V, 1) repack the
//  Core ML model expects.
//
//  Layout, spelled out because an off-by-one here is silent:
//
//      source  frame[v * 3 + c]                (V, C) row-major, V = 33
//      dest    ptr[c * (T*V) + t * V + v]      (1, C, T, V, 1), T = 64
//
//  which is the same permutation `CTRGCNAnalyzer._make_tensor` performs
//  with `.permute(2, 0, 1)` followed by `unsqueeze(0)` / `unsqueeze(-1)`.
//  `scripts/gen_reference_fixtures.py` emits probe indices that pin this.
//

import CoreML
import Foundation

final class FrameBuffer {

    let windowSize: Int
    private let V = PoseLandmarks.count       // 33
    private let C = PoseLandmarks.channels    // 3

    /// Ring of normalised frames, each `V * C` floats.
    private var buffer: [[Float]] = []

    /// Reusable backing array. Allocating a fresh `MLMultiArray` every fifth
    /// frame would churn 25 KB at 6 Hz for no reason; Core ML copies the
    /// contents during `prediction`, so one buffer is safe as long as we
    /// only fill it from the inference queue.
    private var scratch: MLMultiArray?

    init(windowSize: Int = 64) {
        self.windowSize = windowSize
        buffer.reserveCapacity(windowSize)
    }

    var count: Int { buffer.count }
    var isFull: Bool { buffer.count >= windowSize }

    func append(_ frame: [Float]) {
        precondition(frame.count == V * C, "frame must be \(V * C) floats")
        buffer.append(frame)
        if buffer.count > windowSize {
            buffer.removeFirst(buffer.count - windowSize)
        }
    }

    func reset() {
        buffer.removeAll(keepingCapacity: true)
    }

    /// Snapshot of the current window as `(T, V, C)`, for the kinematics
    /// side-band which wants plain arrays rather than an MLMultiArray.
    var window: [[Float]] { buffer }

    // MARK: - Tensor packing

    /// Build the `(1, 3, 64, 33, 1)` float32 input tensor.
    ///
    /// Writes through a raw `Float` pointer rather than the `NSNumber`
    /// subscript: the subscript path boxes every element, which would mean
    /// 6,336 `NSNumber` allocations per inference — roughly 38,000 per
    /// second at a stride of 5. The pointer path is a flat memory write.
    func makeInputTensor() throws -> MLMultiArray {
        guard isFull else {
            throw FrameBufferError.windowNotFull(have: buffer.count, need: windowSize)
        }

        let shape: [NSNumber] = [1, NSNumber(value: C), NSNumber(value: windowSize),
                                 NSNumber(value: V), 1]
        let array: MLMultiArray
        if let existing = scratch, existing.shape == shape {
            array = existing
        } else {
            array = try MLMultiArray(shape: shape, dataType: .float32)
            scratch = array
        }

        let planeStride = windowSize * V          // 2112
        let elementCount = C * planeStride        // 6336

        array.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, strides in
            // Trust the reported strides rather than assuming contiguity;
            // Core ML is free to hand back a padded allocation.
            let sC = strides[1], sT = strides[2], sV = strides[3]
            precondition(ptr.count >= elementCount, "MLMultiArray backing too small")

            for t in 0..<windowSize {
                let frame = buffer[t]
                let tOffset = t * sT
                frame.withUnsafeBufferPointer { src in
                    for v in 0..<V {
                        let srcBase = v * C
                        let dstBase = tOffset + v * sV
                        ptr[dstBase + 0 * sC] = src[srcBase + 0]   // x
                        ptr[dstBase + 1 * sC] = src[srcBase + 1]   // y
                        ptr[dstBase + 2 * sC] = src[srcBase + 2]   // z
                    }
                }
            }
        }

        return array
    }
}

enum FrameBufferError: LocalizedError {
    case windowNotFull(have: Int, need: Int)

    var errorDescription: String? {
        switch self {
        case .windowNotFull(let have, let need):
            return "Sliding window not full yet (\(have)/\(need) frames)."
        }
    }
}
