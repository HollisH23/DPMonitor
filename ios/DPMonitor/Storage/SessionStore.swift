//
//  SessionStore.swift
//  DPMonitor
//
//  Core Data manager. Entirely local — the store lives in the app's
//  Application Support directory and is never synced, exported or
//  uploaded. No CloudKit container is configured, deliberately.
//
//  Schema (see DPMonitor.xcdatamodeld):
//    Session       — indexed headline metrics for the history list
//    SessionDetail — 1:1 JSON archive of the full synthesis payload
//
//  The managed-object subclasses are hand-written (the model is marked
//  codeGenerationType="none") because XcodeGen regenerates the project on
//  every run and Xcode's codegen would race with it.
//

import CoreData
import Foundation
import os

private let storeLog = Logger(subsystem: "au.edu.usyd.dpmonitor", category: "store")

// MARK: - Managed objects

@objc(SessionMO)
final class SessionMO: NSManagedObject {
    @NSManaged var id: UUID
    @NSManaged var date: Date
    @NSManaged var exerciseType: String
    @NSManaged var repCount: Int16
    @NSManaged var qualityScore: Float
    @NSManaged var stabilityScore: Float
    @NSManaged var compensationEvents: Int16
    @NSManaged var durationSeconds: Float
    @NSManaged var detail: SessionDetailMO?

    @nonobjc static func fetchRequest() -> NSFetchRequest<SessionMO> {
        NSFetchRequest<SessionMO>(entityName: "Session")
    }
}

@objc(SessionDetailMO)
final class SessionDetailMO: NSManagedObject {
    @NSManaged var sessionId: UUID
    @NSManaged var summaryJSON: String
    @NSManaged var session: SessionMO?

    @nonobjc static func fetchRequest() -> NSFetchRequest<SessionDetailMO> {
        NSFetchRequest<SessionDetailMO>(entityName: "SessionDetail")
    }
}

// MARK: - Store

@MainActor
final class SessionStore: ObservableObject {

    static let shared = SessionStore()

    /// In-memory instance for SwiftUI previews and unit tests.
    static let preview = SessionStore(inMemory: true)

    @Published private(set) var sessions: [SessionMO] = []
    @Published private(set) var lastError: String?

    private let container: NSPersistentContainer

    var viewContext: NSManagedObjectContext { container.viewContext }

    init(inMemory: Bool = false) {
        container = NSPersistentContainer(name: "DPMonitor")

        if inMemory {
            container.persistentStoreDescriptions.first?.url =
                URL(fileURLWithPath: "/dev/null")
        } else if let description = container.persistentStoreDescriptions.first {
            // Complete-until-first-unlock keeps session history encrypted at
            // rest while still allowing a background write after a reboot.
            description.setOption(FileProtectionType.completeUntilFirstUserAuthentication as NSObject,
                                  forKey: NSPersistentStoreFileProtectionKey)
            description.shouldMigrateStoreAutomatically = true
            description.shouldInferMappingModelAutomatically = true
        }

        container.loadPersistentStores { [weak self] _, error in
            if let error {
                storeLog.error("persistent store failed to load: \(error.localizedDescription)")
                Task { @MainActor in
                    self?.lastError = "Could not open the local session database."
                }
            }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy

        refresh()
    }

    // MARK: Reads

    /// Reload the history list, newest first.
    func refresh() {
        let request = SessionMO.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "date", ascending: false)]
        do {
            sessions = try viewContext.fetch(request)
        } catch {
            storeLog.error("fetch failed: \(error.localizedDescription)")
            sessions = []
        }
    }

    func fetchHistory() -> [SessionMO] {
        refresh()
        return sessions
    }

    /// Decode the archived synthesis for a stored session.
    func loadDetail(for session: SessionMO) -> SessionDetailPayload? {
        guard let json = session.detail?.summaryJSON,
              let data = json.data(using: .utf8) else { return nil }
        do {
            // Must use the NaN-aware decoder — see `encodeDetail`.
            return try JSONDecoder.dpMonitor.decode(SessionDetailPayload.self, from: data)
        } catch {
            storeLog.error("detail decode failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Rehydrate a full `SessionSummary` from storage, for the results screen.
    func summary(for session: SessionMO) -> SessionSummary {
        let payload = loadDetail(for: session)
        return SessionSummary(
            id: session.id,
            date: session.date,
            exerciseType: ExerciseType(rawValue: session.exerciseType) ?? .custom,
            repCount: Int(session.repCount),
            qualityScore: Double(session.qualityScore),
            stabilityScore: Double(session.stabilityScore),
            compensationEvents: Int(session.compensationEvents),
            durationSeconds: Double(session.durationSeconds),
            framesAnalyzed: payload?.framesAnalyzed ?? 0,
            inferenceCalls: payload?.inferenceCalls ?? 0,
            windowSize: payload?.windowSize ?? 64,
            inferenceStride: payload?.inferenceStride ?? 5,
            finalWindowKinematics: payload?.finalWindowKinematics ?? [],
            qualityIsCalibrated: payload?.qualityIsCalibrated ?? true,
            synthesis: payload?.synthesis ?? .empty
        )
    }

    // MARK: Writes

    @discardableResult
    func saveSession(_ summary: SessionSummary) -> Bool {
        let context = viewContext

        let session = SessionMO(context: context)
        session.id = summary.id
        session.date = summary.date
        session.exerciseType = summary.exerciseType.rawValue
        session.repCount = Int16(clamping: summary.repCount)
        session.qualityScore = Float(summary.qualityScore)
        session.stabilityScore = Float(summary.stabilityScore)
        session.compensationEvents = Int16(clamping: summary.compensationEvents)
        session.durationSeconds = Float(summary.durationSeconds)

        let detail = SessionDetailMO(context: context)
        detail.sessionId = summary.id
        detail.summaryJSON = encodeDetail(summary)
        detail.session = session
        session.detail = detail

        do {
            try context.save()
            refresh()
            return true
        } catch {
            context.rollback()
            storeLog.error("save failed: \(error.localizedDescription)")
            lastError = "Could not save this session."
            return false
        }
    }

    func deleteSession(_ id: UUID) {
        let request = SessionMO.fetchRequest()
        request.predicate = NSPredicate(format: "id == %@", id as CVarArg)
        do {
            for object in try viewContext.fetch(request) {
                viewContext.delete(object)   // cascade removes the detail row
            }
            try viewContext.save()
            refresh()
        } catch {
            viewContext.rollback()
            storeLog.error("delete failed: \(error.localizedDescription)")
            lastError = "Could not delete that session."
        }
    }

    func delete(_ session: SessionMO) {
        deleteSession(session.id)
    }

    // MARK: Encoding

    private func encodeDetail(_ summary: SessionSummary) -> String {
        let payload = SessionDetailPayload(summary: summary)
        do {
            // NaN is legal in our numerics (a missing landmark) but not in
            // JSON. Encode it as a sentinel string rather than letting the
            // encoder throw and lose the whole session.
            let encoder = JSONEncoder()
            encoder.nonConformingFloatEncodingStrategy = .convertToString(
                positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
            let data = try encoder.encode(payload)
            return String(data: data, encoding: .utf8) ?? "{}"
        } catch {
            storeLog.error("detail encode failed: \(error.localizedDescription)")
            return "{}"
        }
    }
}

// MARK: - Decoding companion

extension JSONDecoder {
    /// Matches the encoder's NaN strategy above.
    static var dpMonitor: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(
            positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return decoder
    }
}
