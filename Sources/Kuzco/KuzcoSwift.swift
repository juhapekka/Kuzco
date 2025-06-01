// The Swift Programming Language
// https://docs.swift.org/swift-book
import Foundation

public actor Kuzco {
    public static let shared = Kuzco()

    private var activeInstances: [String: LlamaInstance] = [:]
    private var instanceLeaseCount: [String: Int] = [:]

    private init() {
        print("🦙 Kuzco Service Initialized 🦙")
    }

    public func instance(
        for profile: ModelProfile,
        settings: InstanceSettings = .standard,
        predictionConfig: PredictionConfig = .balanced,
        formatter: InteractionFormatting? = nil,
        customStopSequences: [String] = []
    ) async -> (instance: LlamaInstance, loadStream: AsyncStream<LoadUpdate>) {
        if let existingInstance = activeInstances[profile.id] {
            instanceLeaseCount[profile.id, default: 0] += 1
            let stream = AsyncStream<LoadUpdate> { continuation in
                continuation.yield(LoadUpdate(stage: .ready, detail: "Successfully retrieved from cache."))
                continuation.finish()
            }
            return (existingInstance, stream)
        }

        let newInstance = await LlamaInstance(
            profile: profile,
            settings: settings,
            predictionConfig: predictionConfig,
            formatter: formatter,
            customStopSequences: customStopSequences
        )
        activeInstances[profile.id] = newInstance
        instanceLeaseCount[profile.id, default: 0] = 1

        let loadStream = await newInstance.startup()
        return (newInstance, loadStream)
    }

    public func predict(
        dialogue: [Turn],
        with modelProfile: ModelProfile,
        instanceSettings: InstanceSettings = .standard,
        predictionConfig: PredictionConfig = .balanced,
        formatter: InteractionFormatting? = nil,
        customStopSequences: [String] = []
    ) async throws -> AsyncThrowingStream<String, Error> {

        let (llamaInstance, loadStream) = await instance(
            for: modelProfile,
            settings: instanceSettings,
            predictionConfig: predictionConfig,
            formatter: formatter,
            customStopSequences: customStopSequences
        )

        if await !llamaInstance.isReady {
            for await progress in loadStream {
                print("Kuzco Model Loading (\(modelProfile.id)): \(progress.stage) - \(progress.detail ?? "")")
                if progress.stage == .failed {
                    await releaseInstance(for: modelProfile.id, forceShutdown: true)
                    throw KuzcoError.modelInitializationFailed(details: "Loading failed: \(progress.detail ?? "Unknown reason")")
                }
                if progress.stage == .ready { break }
            }
        }

        return await llamaInstance.generate(dialogue: dialogue, overridePredictionConfig: predictionConfig)
    }

    public func releaseInstance(for profileID: String, forceShutdown: Bool = false) async {
        guard let instanceToShutdown = activeInstances[profileID] else { return }

        instanceLeaseCount[profileID, default: 1] -= 1

        if forceShutdown || instanceLeaseCount[profileID, default: 0] <= 0 {
            await instanceToShutdown.performShutdown()
            activeInstances.removeValue(forKey: profileID)
            instanceLeaseCount.removeValue(forKey: profileID)
            print("🦙 Kuzco - LlamaInstance for profile ID '\(profileID)' released and shut down 🦙")
        } else {
            print("🦙 Kuzco - LlamaInstance for profile ID '\(profileID)' lease released. Leases remaining: \(instanceLeaseCount[profileID]!) 🦙")
        }
    }

    public func shutdownAllInstances() async {
        for instance in activeInstances.values {
            await instance.performShutdown()
        }
        activeInstances.removeAll()
        instanceLeaseCount.removeAll()
        print("🦙 Kuzco - All LlamaInstances shut down 🦙")
    }

    public func interruptPrediction(for modelProfileID: String) async {
        await activeInstances[modelProfileID]?.interruptCurrentPrediction()
    }

    public func interruptAllPredictions() async {
        for instance in activeInstances.values {
            await instance.interruptCurrentPrediction()
        }
    }
}
