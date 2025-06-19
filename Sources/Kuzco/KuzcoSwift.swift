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

        // Auto-detect architecture if unknown
        var finalProfile = profile
        if profile.architecture == .unknown {
            let detectedArch = ModelArchitecture.detectFromPath(profile.sourcePath)
            finalProfile = ModelProfile(
                id: profile.id,
                sourcePath: profile.sourcePath, 
                architecture: detectedArch
            )
            print("🦙 Kuzco - Auto-detected architecture: \(detectedArch.rawValue) for model: \(profile.sourcePath)")
        }

        // Create new instance with fallback handling
        var currentInstance = await LlamaInstance(
            profile: finalProfile,
            settings: settings,
            predictionConfig: predictionConfig,
            formatter: formatter,
            customStopSequences: customStopSequences
        )
        activeInstances[finalProfile.id] = currentInstance
        instanceLeaseCount[finalProfile.id, default: 0] = 1

        let loadStream = await currentInstance.startup()
        
        // Check if startup failed due to unsupported architecture and try fallback
        let fallbackStream = AsyncStream<LoadUpdate> { continuation in
            Task {
                var lastUpdate: LoadUpdate?
                for await update in loadStream {
                    lastUpdate = update
                    continuation.yield(update)
                    
                    // If loading failed due to unsupported architecture, try fallback
                    if update.stage == .failed && 
                       (update.detail?.contains("unsupported model architecture") == true ||
                        update.detail?.contains("qwen3") == true ||
                        update.detail?.contains("qwen2") == true) {
                        
                        print("🦙 Kuzco - Attempting fallback for unsupported architecture: \(finalProfile.architecture.rawValue)")
                        
                        // Try with a fallback architecture
                        let fallbackArch = finalProfile.architecture.getFallbackArchitecture()
                        
                        let fallbackProfile = ModelProfile(
                            id: finalProfile.id,
                            sourcePath: finalProfile.sourcePath,
                            architecture: fallbackArch
                        )
                        
                        continuation.yield(LoadUpdate(stage: .preparing, detail: "Retrying with fallback architecture: \(fallbackArch.rawValue)"))
                        
                        // Remove the failed instance
                        activeInstances.removeValue(forKey: finalProfile.id)
                        instanceLeaseCount.removeValue(forKey: finalProfile.id)
                        await currentInstance.performShutdown()
                        
                        // Create new instance with fallback
                        let fallbackInstance = await LlamaInstance(
                            profile: fallbackProfile,
                            settings: settings,
                            predictionConfig: predictionConfig,
                            formatter: formatter,
                            customStopSequences: customStopSequences
                        )
                        activeInstances[finalProfile.id] = fallbackInstance
                        instanceLeaseCount[finalProfile.id, default: 0] = 1
                        currentInstance = fallbackInstance
                        
                        // Try loading with fallback
                        let fallbackLoadStream = await fallbackInstance.startup()
                        for await fallbackUpdate in fallbackLoadStream {
                            continuation.yield(fallbackUpdate)
                            if fallbackUpdate.stage == .ready || fallbackUpdate.stage == .failed {
                                break
                            }
                        }
                        break
                    }
                    
                    if update.stage == .ready || update.stage == .failed {
                        break
                    }
                }
                continuation.finish()
            }
        }
        
        return (currentInstance, fallbackStream)
    }

    /// Predict with a given dialogue and a required system prompt.
    ///
    /// - Parameters:
    ///   - dialogue: The conversation turns to base the prediction on.
    ///   - systemPrompt: A required system prompt to guide the model's behavior.
    ///   - modelProfile: The model profile to use.
    ///   - instanceSettings: Settings for the model instance.
    ///   - predictionConfig: Configuration for prediction behavior.
    ///   - formatter: Optional formatter for interaction formatting.
    ///   - customStopSequences: Optional custom stop sequences.
    ///
    /// - Returns: An asynchronous throwing stream of prediction output strings.
    public func predict(
        dialogue: [Turn],
        systemPrompt: String,
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

        return await llamaInstance.generate(dialogue: dialogue, overrideSystemPrompt: systemPrompt, overridePredictionConfig: predictionConfig)
    }

    /// Predict with completion info with a given dialogue and a required system prompt.
    ///
    /// - Parameters:
    ///   - dialogue: The conversation turns to base the prediction on.
    ///   - systemPrompt: A required system prompt to guide the model's behavior.
    ///   - modelProfile: The model profile to use.
    ///   - instanceSettings: Settings for the model instance.
    ///   - predictionConfig: Configuration for prediction behavior.
    ///   - formatter: Optional formatter for interaction formatting.
    ///   - customStopSequences: Optional custom stop sequences.
    ///
    /// - Returns: An asynchronous throwing stream of StreamResponse with completion details.
    public func predictWithCompletionInfo(
        dialogue: [Turn],
        systemPrompt: String,
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

        return await llamaInstance.generateWithCompletionInfo(dialogue: dialogue, overrideSystemPrompt: systemPrompt, overridePredictionConfig: predictionConfig)
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
