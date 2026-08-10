//
//  ContentView.swift
//  DPMonitor
//
//  Root navigation: a live session tab and a history tab.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var sessionStore: SessionStore
    @State private var selectedTab: Tab = .session

    enum Tab: Hashable {
        case session
        case history
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            SessionView()
                .tabItem {
                    Label("Session", systemImage: "figure.strengthtraining.functional")
                }
                .tag(Tab.session)

            HistoryView()
                .tabItem {
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
                .tag(Tab.history)
        }
        .tint(.green)
    }
}

#Preview {
    ContentView()
        .environmentObject(SessionStore.preview)
        .environmentObject(ThermalMonitor())
}
