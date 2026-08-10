//
//  SessionView.swift
//  DPMonitor
//
//  The live monitor: camera preview, AR skeleton, HUD and transport
//  controls. Stopping a session hands the summary straight to
//  ResultsView and persists it.
//

import SwiftUI

struct SessionView: View {

    @EnvironmentObject private var store: SessionStore
    @EnvironmentObject private var thermal: ThermalMonitor
    @StateObject private var analyzer = SessionAnalyzer()

    @State private var exercise: ExerciseType = .squat
    @State private var completedSummary: SessionSummary?
    @State private var isFinishing = false
    /// Set when the patient starts while off-centre. A warning, never a
    /// block — see `attemptStart()`.
    @State private var centeringWarning: String?

    /// Guides show automatically before recording; during a session they
    /// are opt-in via the toggle.
    private var showCentering: Bool {
        !analyzer.isRunning || analyzer.showCenteringDuringSession
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            CameraView(session: analyzer.extractor.session)
                .ignoresSafeArea()

            SkeletonOverlayView(
                pose: analyzer.overlayPose,
                quality: SkeletonQuality.from(
                    quality: analyzer.metrics.qualityScore,
                    isCompensatory: analyzer.metrics.isCompensatory,
                    isCalibrated: analyzer.modelIsCalibrated)
            )
            .ignoresSafeArea()

            // Between the skeleton and the HUD: guides stay legible over a
            // busy pose, but never occlude the rep counter.
            if showCentering {
                CenteringOverlayView(result: analyzer.centeringResult,
                                     isCompact: analyzer.isRunning)
                    .ignoresSafeArea(edges: .bottom)
                    .transition(.opacity)
            }

            HUDView(metrics: analyzer.metrics,
                    thermalLabel: thermal.label,
                    isThrottling: thermal.isThrottling,
                    modelIsAvailable: analyzer.modelIsAvailable)

            VStack {
                Spacer()
                controls
            }

            if let message = analyzer.errorMessage {
                errorBanner(message)
            }
        }
        .task {
            await analyzer.prepare()
        }
        .onChange(of: thermal.state) { _, newValue in
            analyzer.applyThermalState(newValue)
        }
        .onChange(of: exercise) { _, newValue in
            analyzer.exerciseType = newValue
        }
        .onDisappear {
            // Tear the camera down entirely — `stop()` alone deliberately
            // leaves it running so the patient can re-centre between sets.
            analyzer.stopCamera()
        }
        .alert("Not centered yet",
               isPresented: Binding(get: { centeringWarning != nil },
                                    set: { if !$0 { centeringWarning = nil } })) {
            Button("Start anyway") {
                centeringWarning = nil
                beginSession()
            }
            Button("Let me adjust", role: .cancel) { centeringWarning = nil }
        } message: {
            Text((centeringWarning ?? "")
                 + "\n\nYou can still record — measurements may be less "
                 + "reliable if part of your body is out of frame.")
        }
        .fullScreenCover(item: $completedSummary) { summary in
            NavigationStack {
                ResultsView(summary: summary)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Done") { completedSummary = nil }
                        }
                    }
            }
        }
    }

    // MARK: Controls

    private var controls: some View {
        VStack(spacing: 14) {
            if !analyzer.isRunning {
                Picker("Exercise", selection: $exercise) {
                    ForEach(ExerciseType.allCases) { type in
                        Text(type.displayName).tag(type)
                    }
                }
                .pickerStyle(.menu)
                .tint(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
            } else {
                Toggle(isOn: $analyzer.showCenteringDuringSession) {
                    Label("Framing guides", systemImage: "viewfinder")
                        .font(.caption.weight(.medium))
                }
                .toggleStyle(.button)
                .tint(.white)
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
            }

            HStack(spacing: 28) {
                Button {
                    analyzer.extractor.switchCamera()
                } label: {
                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                        .font(.title2)
                        .frame(width: 52, height: 52)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .disabled(isFinishing)

                Button {
                    Task { await toggleSession() }
                } label: {
                    ZStack {
                        Circle()
                            .fill(analyzer.isRunning ? Color.red : Color.green)
                            .frame(width: 76, height: 76)
                        if isFinishing {
                            ProgressView().tint(.white)
                        } else {
                            Image(systemName: analyzer.isRunning ? "stop.fill" : "play.fill")
                                .font(.title)
                                .foregroundStyle(.white)
                        }
                    }
                }
                .disabled(isFinishing)

                // Balances the row so the record button stays centred.
                Color.clear.frame(width: 52, height: 52)
            }
            .foregroundStyle(.white)
        }
        .padding(.bottom, 28)
    }

    private func toggleSession() async {
        if analyzer.isRunning {
            isFinishing = true
            analyzer.stop()
            var summary = await analyzer.makeSummary()
            summary.exerciseType = exercise
            store.saveSession(summary)
            completedSummary = summary
            isFinishing = false
        } else {
            attemptStart()
        }
    }

    /// Warn on bad framing, but never block.
    ///
    /// A hard gate would lock out exactly the patients who most need the
    /// app — someone in a wheelchair, in a small room, or standing with
    /// assistance may be physically unable to hit the ideal box. The
    /// warning explains the cost and lets them proceed.
    private func attemptStart() {
        let centering = analyzer.centeringResult
        if centering.isCentered {
            beginSession()
        } else {
            centeringWarning = centering.message
        }
    }

    private func beginSession() {
        analyzer.exerciseType = exercise
        analyzer.start()
    }

    private func errorBanner(_ message: String) -> some View {
        VStack {
            Text(message)
                .font(.footnote)
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
                .padding(12)
                .frame(maxWidth: .infinity)
                .background(.red.opacity(0.9))
            Spacer()
        }
        .ignoresSafeArea(edges: .top)
    }
}

#Preview {
    SessionView()
        .environmentObject(SessionStore.preview)
        .environmentObject(ThermalMonitor())
}
