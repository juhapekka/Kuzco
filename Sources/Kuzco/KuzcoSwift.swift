// The Swift Programming Language
// https://docs.swift.org/swift-book
import Foundation

public actor Kuzco {
    public static let shared = Kuzco()

    private var activeInstances: [String: LlamaInstance] = [:]
    private var instanceLeaseCount: [String: Int] = [:]
    private let modelCache = ModelCache.shared

    private init() {
        print("🦙 Kuzco Service Initialized 🦙")
    }
    
    /// Configure caching behavior for models
    public func configureCaching(with settings: CacheSettings) async {
        await modelCache.configure(with: settings)
    }

    public func instance(
        for profile: ModelProfile,
        settings: InstanceSettings = .standard,
        predictionConfig: PredictionConfig = .balanced,
        formatter: InteractionFormatting? = nil,
        customStopSequences: [String] = []
    ) async -> (instance: LlamaInstance, loadStream: AsyncStream<LoadUpdate>) {
        // Check in-memory cache first
        if let existingInstance = activeInstances[profile.id] {
            instanceLeaseCount[profile.id, default: 0] += 1
            let stream = AsyncStream<LoadUpdate> { continuation in
                continuation.yield(LoadUpdate(stage: .ready, detail: "Successfully retrieved from memory cache."))
                continuation.finish()
            }
            return (existingInstance, stream)
        }

        // Check persistent cache
        if await modelCache.hasCachedInstance(for: profile, with: settings) {
            do {
                if let cachedInstance = try await modelCache.restoreCachedInstance(for: profile, with: settings) {
                    activeInstances[profile.id] = cachedInstance
                    instanceLeaseCount[profile.id, default: 0] = 1
                    
                    let stream = AsyncStream<LoadUpdate> { continuation in
                        continuation.yield(LoadUpdate(stage: .ready, detail: "Successfully restored from persistent cache."))
                        continuation.finish()
                    }
                    return (cachedInstance, stream)
                }
            } catch {
                print("🦙 Kuzco - Failed to restore cached instance: \(error.localizedDescription) 🦙")
            }
        }

        // Create new instance
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

    public func predictWithCompletionInfo(
        dialogue: [Turn],
        with modelProfile: ModelProfile,
        instanceSettings: InstanceSettings = .standard,
        predictionConfig: PredictionConfig = .balanced,
        formatter: InteractionFormatting? = nil,
        customStopSequences: [String] = []
    ) async throws -> AsyncThrowingStream<StreamResponse, Error> {

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

        return await llamaInstance.generateWithCompletionInfo(dialogue: dialogue, overridePredictionConfig: predictionConfig)
    }

    public func releaseInstance(for profileID: String, forceShutdown: Bool = false) async {
        guard let instanceToShutdown = activeInstances[profileID] else { return }

        instanceLeaseCount[profileID, default: 1] -= 1

        if forceShutdown || instanceLeaseCount[profileID, default: 0] <= 0 {
            // Try to cache the instance before shutting down
            do {
                try await modelCache.cacheInstance(instanceToShutdown)
            } catch {
                print("🦙 Kuzco - Failed to cache instance before shutdown: \(error.localizedDescription) 🦙")
            }
            
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
    
    // MARK: - Cache Management
    
    /// Clear all persistent cached models
    public func clearModelCache() async {
        await modelCache.clearAllCache()
    }
    
    /// Remove a specific model from persistent cache
    public func removeCachedModel(for profile: ModelProfile, with settings: InstanceSettings = .standard) async {
        await modelCache.removeCachedInstance(for: profile, with: settings)
    }
    
    /// Check if a model is available in persistent cache
    public func isModelCached(for profile: ModelProfile, with settings: InstanceSettings = .standard) async -> Bool {
        return await modelCache.hasCachedInstance(for: profile, with: settings)
    }
}
