//
//  CenteringOverlayView.swift
//  DPMonitor
//
//  Pre-session framing guide. SwiftUI counterpart to `draw_overlay` in
//  Final/centering_logic.py — same guide lines, same 30%/70% safe zone,
//  same semi-transparent status banner.
//
//  Sits between the skeleton overlay and the HUD in SessionView's ZStack:
//  above the skeleton so the guides stay legible against a busy pose,
//  below the HUD so the rep counter is never occluded.
//
//  The directional arrow is drawn in SCREEN space, pointing toward the
//  frame centre. That is correct whether or not the preview is mirrored,
//  because the patient sees their own image next to the arrow and simply
//  follows it.
//

import SwiftUI

struct CenteringOverlayView: View {

    let result: CenteringResult
    /// Dims the guide lines once recording starts, so they stop competing
    /// with the skeleton while remaining available for a quick glance.
    var isCompact: Bool = false

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .top) {
                guides(in: geo.size)
                if !isCompact { directionalCue(in: geo.size) }
                banner
            }
        }
        .allowsHitTesting(false)
        .animation(.easeOut(duration: 0.2), value: result)
    }

    // MARK: Guide lines

    private func guides(in size: CGSize) -> some View {
        let lineOpacity = isCompact ? 0.25 : 1.0
        return ZStack {
            // Safe-zone boundaries at 30% and 70%, drawn as one dashed path
            // so there is no per-line layout to get wrong.
            Path { path in
                for fraction in [CenteringEvaluator.hipXMin, CenteringEvaluator.hipXMax] {
                    let x = CGFloat(fraction) * size.width
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                }
            }
            .stroke(Color.gray.opacity(0.9 * lineOpacity),
                    style: StrokeStyle(lineWidth: 1, dash: [6, 6]))

            // Vertical centre line.
            Path { path in
                path.move(to: CGPoint(x: size.width / 2, y: 0))
                path.addLine(to: CGPoint(x: size.width / 2, y: size.height))
            }
            .stroke(Color.yellow.opacity(0.85 * lineOpacity), lineWidth: 1)
        }
    }

    // MARK: Directional cue

    /// A large translucent chevron pointing the way back toward centre.
    @ViewBuilder
    private func directionalCue(in size: CGSize) -> some View {
        switch result.status {
        case .moveLeft, .moveRight:
            let pointsLeft = (result.status == .moveLeft)
            Image(systemName: pointsLeft ? "chevron.compact.left" : "chevron.compact.right")
                .font(.system(size: 120, weight: .thin))
                .foregroundStyle(tint.opacity(0.75))
                .position(x: pointsLeft ? size.width * 0.14 : size.width * 0.86,
                          y: size.height * 0.5)
                .transition(.opacity)
        default:
            EmptyView()
        }
    }

    // MARK: Status banner

    private var banner: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: symbolName)
                    .font(.headline)
                Text(result.message)
                    .font(.headline)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(tint)

            if !isCompact {
                ForEach(result.details, id: \.self) { line in
                    Text(line)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.8))
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.black.opacity(0.65))
        .padding(.top, 8)
    }

    private var tint: Color {
        switch result.severity {
        case .ok:       return .green
        case .warning:  return .orange
        case .critical: return .red
        }
    }

    private var symbolName: String {
        switch result.status {
        case .centered:    return "checkmark.circle.fill"
        case .moveLeft:    return "arrow.left.circle.fill"
        case .moveRight:   return "arrow.right.circle.fill"
        case .tooClose:    return "arrow.down.backward.and.arrow.up.forward.circle.fill"
        case .tooFar:      return "arrow.up.forward.and.arrow.down.backward.circle.fill"
        case .headClipped: return "exclamationmark.triangle.fill"
        case .feetClipped: return "exclamationmark.triangle.fill"
        case .notDetected: return "person.slash.fill"
        }
    }
}

// MARK: - Previews

#Preview("Centered") {
    ZStack {
        Color.gray
        CenteringOverlayView(result: CenteringEvaluator.evaluate(previewPose(hipX: 0.5)))
    }
    .ignoresSafeArea()
}

#Preview("Too far left") {
    ZStack {
        Color.gray
        CenteringOverlayView(result: CenteringEvaluator.evaluate(previewPose(hipX: 0.15)))
    }
    .ignoresSafeArea()
}

/// Minimal well-formed pose for the previews above.
private func previewPose(hipX: Float) -> [String: SIMD4<Float>] {
    let w: Float = 0.09
    return [
        "nose":           SIMD4(hipX, 0.10, 0, 0.9),
        "left_shoulder":  SIMD4(hipX - w, 0.32, 0, 0.9),
        "right_shoulder": SIMD4(hipX + w, 0.32, 0, 0.9),
        "left_hip":       SIMD4(hipX - w, 0.60, 0, 0.9),
        "right_hip":      SIMD4(hipX + w, 0.60, 0, 0.9),
        "left_knee":      SIMD4(hipX - w, 0.80, 0, 0.9),
        "right_knee":     SIMD4(hipX + w, 0.80, 0, 0.9),
    ]
}
