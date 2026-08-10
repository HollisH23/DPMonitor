//
//  SkeletonOverlayView.swift
//  DPMonitor
//
//  AR skeleton overlay drawn on a CALayer, driven by CADisplayLink so it
//  runs at the display's native refresh rate (120 Hz on iPhone 16 Pro Max)
//  rather than at the 30 FPS pose cadence.
//
//  Why not SwiftUI shapes: at 120 Hz a SwiftUI redraw of ~70 primitives
//  costs a full diff of the view tree every frame. A single CAShapeLayer
//  whose path we swap is one layout-free commit.
//
//  Occlusion: joints and bones below the visibility threshold are simply
//  not drawn. Fading them instead reads as "the tracker is unsure", which
//  is exactly the message a patient should get.
//

import QuartzCore
import SwiftUI
import UIKit
import simd

struct SkeletonOverlayView: UIViewRepresentable {

    /// Branch A landmarks, screen-normalised (x, y in [0, 1], w = visibility).
    var pose: [String: SIMD4<Float>]
    /// Drives the joint/bone colour.
    var quality: SkeletonQuality
    var visibilityThreshold: Float = 0.5

    func makeUIView(context: Context) -> SkeletonUIView {
        let view = SkeletonUIView()
        view.backgroundColor = .clear
        view.isUserInteractionEnabled = false
        view.visibilityThreshold = visibilityThreshold
        return view
    }

    func updateUIView(_ uiView: SkeletonUIView, context: Context) {
        uiView.visibilityThreshold = visibilityThreshold
        uiView.quality = quality
        uiView.update(pose: pose)
    }

    static func dismantleUIView(_ uiView: SkeletonUIView, coordinator: ()) {
        uiView.stopDisplayLink()
    }
}

// MARK: - Quality colouring

enum SkeletonQuality {
    case good
    case caution
    case compensatory
    /// No trustworthy quality signal — draw neutral rather than alarming.
    case uncalibrated

    var color: UIColor {
        switch self {
        case .good:         return UIColor.systemGreen
        case .caution:      return UIColor.systemOrange
        case .compensatory: return UIColor.systemRed
        case .uncalibrated: return UIColor.white.withAlphaComponent(0.85)
        }
    }

    /// Map the running quality score and the compensation flag onto a colour.
    ///
    /// `isCalibrated` short-circuits everything: an untrained model reports
    /// every window as compensatory, which would paint the skeleton
    /// permanently red and read as a genuine clinical warning.
    static func from(quality: Double,
                     isCompensatory: Bool,
                     isCalibrated: Bool = true) -> SkeletonQuality {
        guard isCalibrated else { return .uncalibrated }
        if isCompensatory { return .compensatory }
        return quality >= 0.7 ? .good : .caution
    }
}

// MARK: - UIView

final class SkeletonUIView: UIView {

    var visibilityThreshold: Float = 0.5
    var quality: SkeletonQuality = .good {
        didSet { needsRedraw = true }
    }

    private let boneLayer = CAShapeLayer()
    private let jointLayer = CAShapeLayer()
    private var displayLink: CADisplayLink?

    /// Latest pose, flattened to view-independent normalised points.
    private var points: [SIMD3<Float>?] = Array(repeating: nil, count: PoseLandmarks.count)
    private var needsRedraw = false

    override init(frame: CGRect) {
        super.init(frame: frame)
        configureLayers()
        startDisplayLink()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configureLayers()
        startDisplayLink()
    }

    deinit {
        displayLink?.invalidate()
    }

    private func configureLayers() {
        boneLayer.fillColor = nil
        boneLayer.lineWidth = 4
        boneLayer.lineCap = .round
        boneLayer.lineJoin = .round
        boneLayer.strokeColor = quality.color.cgColor
        boneLayer.shadowColor = UIColor.black.cgColor
        boneLayer.shadowOpacity = 0.35
        boneLayer.shadowRadius = 3
        boneLayer.shadowOffset = .zero

        jointLayer.strokeColor = nil
        jointLayer.fillColor = UIColor.white.cgColor

        layer.addSublayer(boneLayer)
        layer.addSublayer(jointLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        boneLayer.frame = bounds
        jointLayer.frame = bounds
        needsRedraw = true
    }

    // MARK: Display link

    private func startDisplayLink() {
        let link = CADisplayLink(target: self, selector: #selector(tick))
        // Let the system run us at the panel's native rate; ProMotion will
        // give 120 Hz when thermals allow and back off on its own otherwise.
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func tick() {
        guard needsRedraw else { return }
        needsRedraw = false
        redraw()
    }

    // MARK: Data

    func update(pose: [String: SIMD4<Float>]) {
        guard !pose.isEmpty else {
            if points.contains(where: { $0 != nil }) {
                points = Array(repeating: nil, count: PoseLandmarks.count)
                needsRedraw = true
            }
            return
        }
        for (i, name) in PoseLandmarks.names.enumerated() {
            guard let p = pose[name], p.w >= visibilityThreshold else {
                points[i] = nil
                continue
            }
            points[i] = SIMD3(p.x, p.y, p.z)
        }
        needsRedraw = true
    }

    // MARK: Rendering

    private func redraw() {
        let size = bounds.size
        guard size.width > 0, size.height > 0 else { return }

        let bones = UIBezierPath()
        for (a, b) in PoseLandmarks.bones {
            guard let pa = points[a], let pb = points[b] else { continue }
            bones.move(to: screenPoint(pa, in: size))
            bones.addLine(to: screenPoint(pb, in: size))
        }

        let joints = UIBezierPath()
        let radius: CGFloat = 5
        for p in points {
            guard let p else { continue }
            let c = screenPoint(p, in: size)
            joints.append(UIBezierPath(
                arcCenter: c, radius: radius,
                startAngle: 0, endAngle: .pi * 2, clockwise: true))
        }

        // Path swaps inside a disabled implicit-animation transaction —
        // otherwise Core Animation interpolates every vertex and the
        // skeleton visibly smears at 120 Hz.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        boneLayer.strokeColor = quality.color.cgColor
        boneLayer.path = bones.cgPath
        jointLayer.path = joints.cgPath
        CATransaction.commit()
    }

    /// Map MediaPipe's normalised coordinates onto the view.
    ///
    /// The preview layer uses `.resizeAspectFill`, so the camera image is
    /// cropped, not letterboxed — a plain `x * width, y * height` mapping
    /// is correct as long as the preview and the overlay share bounds.
    @inline(__always)
    private func screenPoint(_ p: SIMD3<Float>, in size: CGSize) -> CGPoint {
        CGPoint(x: CGFloat(p.x) * size.width,
                y: CGFloat(p.y) * size.height)
    }
}
