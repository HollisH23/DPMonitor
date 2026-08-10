//
//  ResultsView.swift
//  DPMonitor
//
//  Post-session summary. Renders the payload produced by
//  Core/Synthesis.swift (port of backend/analyzer/synthesis.py):
//  headline metrics, the ROM curve with rep boundaries marked, per-rep
//  breakdown, the stability trend and the fatigue index.
//

import Charts
import SwiftUI

struct ResultsView: View {

    let summary: SessionSummary

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                headline
                if !summary.synthesis.romCurve.isEmpty { romChart }
                if !summary.synthesis.stabilityTrend.isEmpty { stabilityChart }
                if !summary.synthesis.perRepROM.isEmpty { perRepTable }
                if !summary.finalWindowKinematics.isEmpty { jointTable }
                footnote
            }
            .padding(20)
        }
        .navigationTitle(summary.exerciseType.displayName)
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: Headline

    private var headline: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(summary.date.formatted(date: .abbreviated, time: .shortened))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                MetricTile(title: "Reps", value: "\(summary.repCount)", tint: .green)
                if summary.qualityIsCalibrated {
                    MetricTile(title: "Form", value: "\(summary.qualityPercent)%",
                               tint: summary.qualityScore >= 0.7 ? .green : .orange)
                } else {
                    MetricTile(title: "Form", value: "—", tint: .gray)
                }
                MetricTile(title: "Stability", value: "\(summary.stabilityPercent)%", tint: .blue)
                MetricTile(title: "Duration", value: summary.durationText, tint: .gray)
            }

            if !summary.qualityIsCalibrated {
                // Rep count, ROM, stability and tremor are all geometric and
                // remain valid — only the learned quality signal is missing.
                Label(
                    "Form scoring is unavailable: this build uses untrained "
                    + "model weights. Rep count, range of motion and stability "
                    + "are measured geometrically and remain valid.",
                    systemImage: "info.circle"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.secondary.opacity(0.10),
                            in: RoundedRectangle(cornerRadius: 10))
            }

            HStack(spacing: 12) {
                if let accuracy = summary.synthesis.overallAccuracy {
                    // Reads as similarity against a clinician reference when
                    // one is loaded; otherwise it is the scaled mean quality.
                    smallStat("Overall accuracy", String(format: "%.1f", accuracy))
                }
                if let fatigue = summary.synthesis.fatigueIndex {
                    smallStat("Fatigue index", String(format: "%.3f", fatigue))
                }
                if summary.compensationEvents > 0 {
                    smallStat("Form drift", "\(summary.compensationEvents)×")
                }
            }
        }
    }

    private func smallStat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.callout.weight(.semibold).monospacedDigit())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Charts

    private var romChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Range of motion",
                         subtitle: summary.synthesis.primaryJoint.map(prettify))
            Chart {
                ForEach(summary.synthesis.romCurve) { point in
                    LineMark(
                        x: .value("Time", point.timestampMs / 1000.0),
                        y: .value("Angle", point.angle)
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(.green)
                }
                // Troughs are the bottom of each rep — the deepest point of
                // the movement, and where a physio looks first.
                ForEach(summary.synthesis.romCurve.filter(\.isTrough)) { point in
                    PointMark(
                        x: .value("Time", point.timestampMs / 1000.0),
                        y: .value("Angle", point.angle)
                    )
                    .foregroundStyle(.orange)
                    .symbolSize(60)
                }
            }
            .chartXAxisLabel("seconds")
            .chartYAxisLabel("degrees")
            .frame(height: 200)
        }
    }

    private var stabilityChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Stability by rep",
                         subtitle: "1 − normalised tremor")
            Chart(summary.synthesis.stabilityTrend) { point in
                BarMark(
                    x: .value("Rep", point.rep),
                    y: .value("Stability", point.stability)
                )
                .foregroundStyle(point.stability >= 0.7 ? .green : .orange)
            }
            .chartYScale(domain: 0...1)
            .frame(height: 160)
        }
    }

    // MARK: Tables

    private var perRepTable: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Per-rep peak ROM", subtitle: nil)
            VStack(spacing: 0) {
                headerRow(["Rep", "Min", "Max", "Range"])
                ForEach(summary.synthesis.perRepROM) { rep in
                    dataRow([
                        "\(rep.rep)",
                        degrees(rep.minDeg),
                        degrees(rep.maxDeg),
                        degrees(rep.rangeDeg),
                    ])
                }
            }
            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var jointTable: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Final window kinematics", subtitle: "last 64 frames")
            VStack(spacing: 0) {
                headerRow(["Joint", "ROM", "Vel RMS", "Acc RMS"])
                ForEach(summary.finalWindowKinematics) { record in
                    dataRow([
                        record.displayName,
                        degrees(record.rangeDeg),
                        String(format: "%.2f", record.velocityRMS),
                        String(format: "%.2f", record.accelerationRMS),
                    ])
                }
            }
            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var footnote: some View {
        Text("""
             \(summary.framesAnalyzed) frames analysed · \
             \(summary.inferenceCalls) model passes · \
             window \(summary.windowSize) · stride \(summary.inferenceStride)
             """)
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }

    // MARK: Building blocks

    private func sectionTitle(_ title: String, subtitle: String?) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).font(.headline)
            if let subtitle {
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func headerRow(_ cells: [String]) -> some View {
        HStack {
            ForEach(cells, id: \.self) { cell in
                Text(cell)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private func dataRow(_ cells: [String]) -> some View {
        HStack {
            ForEach(Array(cells.enumerated()), id: \.offset) { _, cell in
                Text(cell)
                    .font(.caption.monospacedDigit())
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    /// NaN means "the landmark was never visible" — say so rather than
    /// printing "nan°", which reads like a bug to a clinician.
    private func degrees(_ value: Double) -> String {
        value.isNaN ? "—" : String(format: "%.1f°", value)
    }

    private func prettify(_ joint: String) -> String {
        joint.split(separator: "_").map(\.capitalized).joined(separator: " ")
    }
}

// MARK: - Metric tile

struct MetricTile: View {
    let title: String
    let value: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title2.weight(.bold).monospacedDigit())
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
    }
}

#Preview {
    NavigationStack {
        ResultsView(summary: .empty)
    }
}
