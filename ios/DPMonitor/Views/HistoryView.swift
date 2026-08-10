//
//  HistoryView.swift
//  DPMonitor
//
//  Past sessions, read from the local Core Data store. Nothing here
//  leaves the device — there is no share sheet and no export, by design.
//

import SwiftUI

struct HistoryView: View {

    @EnvironmentObject private var store: SessionStore

    var body: some View {
        NavigationStack {
            Group {
                if store.sessions.isEmpty {
                    emptyState
                } else {
                    list
                }
            }
            .navigationTitle("History")
            .onAppear { store.refresh() }
        }
    }

    private var list: some View {
        List {
            ForEach(store.sessions, id: \.id) { session in
                NavigationLink {
                    ResultsView(summary: store.summary(for: session))
                } label: {
                    row(session)
                }
            }
            .onDelete(perform: delete)
        }
        .listStyle(.plain)
    }

    private func row(_ session: SessionMO) -> some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text(exerciseName(session))
                    .font(.headline)
                Text(session.date.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 3) {
                Text("\(session.repCount) reps")
                    .font(.subheadline.weight(.semibold).monospacedDigit())
                Text("\(Int((session.qualityScore * 100).rounded()))% form")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(session.qualityScore >= 0.7 ? .green : .orange)
            }
        }
        .padding(.vertical, 4)
    }

    private var emptyState: some View {
        ContentUnavailableView(
            "No sessions yet",
            systemImage: "figure.strengthtraining.functional",
            description: Text("Recorded sessions appear here. Everything stays on this device.")
        )
    }

    private func exerciseName(_ session: SessionMO) -> String {
        (ExerciseType(rawValue: session.exerciseType) ?? .custom).displayName
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets where store.sessions.indices.contains(index) {
            store.delete(store.sessions[index])
        }
    }
}

#Preview {
    HistoryView()
        .environmentObject(SessionStore.preview)
}
