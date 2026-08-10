//
//  HUDView.swift
//  DPMonitor
//
//  Live overlay: rep count, quality gauge, warm-up progress and the
//  feedback ticker. Everything is read from a single `LiveMetrics` value
//  so the HUD re-renders once per update rather than once per property.
//

import SwiftUI

struct HUDView: View {

    let metrics: LiveMetrics
    let thermalLabel: String
    let isThrottling: Bool
    let modelIsAvailable: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            topRow
            if metrics.windowFill < 1.0 {
                warmUpBar
            }
            Spacer()
            feedbackTicker
        }
        .padding(20)
        .allowsHitTesting(false)   // never steal taps from the controls below
    }

    // MARK: Top row

    private var topRow: some View {
        HStack(alignment: .top) {
            repCard
            Spacer()
            VStack(alignment: .trailing, spacing: 8) {
                QualityGauge(quality: metrics.qualityScore,
                             isCompensatory: metrics.isCompensatory,
                             isCalibrated: metrics.qualityIsCalibrated)
                if let similarity = metrics.similarityScore {
                    badge("Match \(Int(similarity.rounded()))%", tint: .cyan)
                }
                if isThrottling {
                    badge("Thermal: \(thermalLabel)", tint: .orange)
                }
                if !modelIsAvailable {
                    badge("Model not bundled", tint: .red)
                } else if !metrics.qualityIsCalibrated {
                    // Untrained weights saturate the softmax, so the score
                    // would read a confident, meaningless 0%. Say why.
                    badge("Uncalibrated model", tint: .indigo)
                }
            }
        }
    }

    private var repCard: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(metrics.repCount)")
                .font(.system(size: 56, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(.white)
            Text("REPS")
                .font(.caption.weight(.semibold))
                .tracking(2)
                .foregroundStyle(.white.opacity(0.7))
            Text(timeText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.5))
        }
        .shadow(color: .black.opacity(0.6), radius: 4)
    }

    private var timeText: String {
        let total = Int(metrics.elapsedSeconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    // MARK: Warm-up

    /// The model needs a full 64-frame window (~2.1 s at 30 FPS) before its
    /// first forward pass. Showing that explicitly stops the quality gauge
    /// reading as "perfect" during the warm-up.
    private var warmUpBar: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Building movement window…")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.8))
            ProgressView(value: metrics.windowFill)
                .tint(.white)
                .frame(maxWidth: 220)
        }
        .shadow(color: .black.opacity(0.6), radius: 3)
    }

    // MARK: Feedback

    private var feedbackTicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(metrics.feedback, id: \.self) { message in
                Text(message)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.ultraThinMaterial, in: Capsule())
                    .transition(.move(edge: .leading).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.2), value: metrics.feedback)
    }

    private func badge(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(tint.opacity(0.85), in: Capsule())
    }
}

// MARK: - Quality gauge

struct QualityGauge: View {
    let quality: Double
    let isCompensatory: Bool
    /// When false the model has untrained weights; show the score as
    /// unavailable rather than rendering a saturated 0% as a real finding.
    var isCalibrated: Bool = true

    private var tint: Color {
        guard isCalibrated else { return .gray }
        if isCompensatory { return .red }
        return quality >= 0.7 ? .green : .orange
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(.white.opacity(0.2), lineWidth: 8)
            Circle()
                .trim(from: 0, to: isCalibrated ? max(0, min(1, quality)) : 0)
                .stroke(tint, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeOut(duration: 0.25), value: quality)
            VStack(spacing: 0) {
                Text(isCalibrated ? "\(Int((quality * 100).rounded()))" : "—")
                    .font(.title3.weight(.bold).monospacedDigit())
                    .foregroundStyle(.white)
                Text("FORM")
                    .font(.system(size: 8, weight: .semibold))
                    .tracking(1)
                    .foregroundStyle(.white.opacity(0.7))
            }
        }
        .frame(width: 76, height: 76)
        .shadow(color: .black.opacity(0.5), radius: 4)
    }
}

#Preview {
    ZStack {
        Color.gray
        HUDView(
            metrics: LiveMetrics(repCount: 7, qualityScore: 0.82,
                                 isCompensatory: false,
                                 feedback: ["Rep counted", "Keep your chest up"],
                                 windowFill: 0.6, inferenceCalls: 14,
                                 similarityScore: 91.5, elapsedSeconds: 74),
            thermalLabel: "Fair",
            isThrottling: false,
            modelIsAvailable: true
        )
    }
    .ignoresSafeArea()
}
