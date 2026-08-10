//
//  DPMonitorApp.swift
//  DPMonitor
//
//  SwiftUI entry point for the fully offline, on-device rehabilitation
//  movement analyser.
//
//  Offline guarantee
//  -----------------
//  This target links no networking framework and issues no URL requests.
//  Every stage — pose extraction (MediaPipe), smoothing, normalisation,
//  CTR-GCN inference (Core ML / Neural Engine) and persistence (Core Data)
//  — runs locally. See ios/README.md for the airplane-mode acceptance test.
//

import SwiftUI

@main
struct DPMonitorApp: App {
    /// Single Core Data stack shared by the history and results screens.
    @StateObject private var sessionStore = SessionStore.shared

    /// Surfaces `ProcessInfo.thermalState` so the UI can degrade gracefully
    /// during long sessions instead of letting iOS throttle us silently.
    @StateObject private var thermalMonitor = ThermalMonitor()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(sessionStore)
                .environmentObject(thermalMonitor)
                .environment(\.managedObjectContext, sessionStore.viewContext)
                .preferredColorScheme(.dark)   // camera-first UI
        }
    }
}

// MARK: - Thermal monitoring

/// Observes the device thermal state.
///
/// Continuous 30 FPS pose extraction plus a Core ML forward pass every five
/// frames is a sustained load. Rather than let the system quietly drop our
/// frame rate, we watch `thermalState` and let `SessionAnalyzer` widen the
/// inference stride once the device reports `.serious`.
@MainActor
final class ThermalMonitor: ObservableObject {
    @Published private(set) var state: ProcessInfo.ThermalState

    init() {
        state = ProcessInfo.processInfo.thermalState
        NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.state = ProcessInfo.processInfo.thermalState
            }
        }
    }

    var isThrottling: Bool { state == .serious || state == .critical }

    var label: String {
        switch state {
        case .nominal:  return "Nominal"
        case .fair:     return "Fair"
        case .serious:  return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }
}
