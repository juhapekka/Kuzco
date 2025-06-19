// The Swift Programming Language
// https://docs.swift.org/swift-book
import Foundation

/// Result type for safe model loading operations
public enum LoadResult {
    case success(LlamaInstance)
    case failure(KuzcoError)
    
    /// Returns true if the load was successful
    public var isSuccess: Bool {
        switch self {
        case .success: return true
        case .failure: return false
        }
    }
    
    /// Returns the error if load failed, nil if successful
    public var error: KuzcoError? {
        switch self {
        case .success: return nil
        case .failure(let error): return error
        }
    }
    
    /// Returns the instance if load was successful, nil if failed
    public var instance: LlamaInstance? {
        switch self {
        case .success(let instance): return instance
        case .failure: return nil
        }
    }
}

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
        do {
            let newInstance = await LlamaInstance(
                profile: finalProfile,
                settings: settings,
                predictionConfig: predictionConfig,
                formatter: formatter,
                customStopSequences: customStopSequences
            )
            
            let loadStream = await newInstance.startup()
            activeInstances[finalProfile.id] = newInstance
            instanceLeaseCount[finalProfile.id, default: 0] = 1
            
            return (newInstance, loadStream)
            
        } catch let error as KuzcoError {
            // Handle specific error types with helpful guidance
            print("🦙 Kuzco - Instance creation failed: \(error.localizedDescription) 🦙")
            
            // Create error stream with recovery suggestions
            let errorStream = AsyncStream<LoadUpdate> { continuation in
                var errorDetail = error.localizedDescription
                if let suggestion = error.recoverySuggestion {
                    errorDetail += "\n\nSuggestion: \(suggestion)"
                }
                
                continuation.yield(LoadUpdate(
                    stage: .failed,
                    detail: errorDetail,
                    hasError: true
                ))
                continuation.finish()
            }
            
            // Create a placeholder instance that will fail gracefully
            let failedInstance = await LlamaInstance(
                profile: finalProfile,
                settings: InstanceSettings(contextLength: 512, processingBatchSize: 32),
                predictionConfig: predictionConfig,
                formatter: formatter,
                customStopSequences: customStopSequences
            )
            
            return (failedInstance, errorStream)
            
        } catch {
            print("🦙 Kuzco - Unexpected error during instance creation: \(error.localizedDescription) 🦙")
            
            let errorStream = AsyncStream<LoadUpdate> { continuation in
                continuation.yield(LoadUpdate(
                    stage: .failed,
                    detail: "Unexpected error: \(error.localizedDescription)",
                    hasError: true
                ))
                continuation.finish()
            }
            
            let failedInstance = await LlamaInstance(
                profile: finalProfile,
                settings: InstanceSettings(contextLength: 512, processingBatchSize: 32),
                predictionConfig: predictionConfig,
                formatter: formatter,
                customStopSequences: customStopSequences
            )
            
            return (failedInstance, errorStream)
        }
    }

    /// Safe model loading with comprehensive error handling and fallback options
    public static func loadModelSafely(
        profile: ModelProfile,
        settings: InstanceSettings = .standard,
        predictionConfig: PredictionConfig = .balanced,
        formatter: InteractionFormatting? = nil,
        customStopSequences: [String] = []
    ) async -> (instance: LlamaInstance?, loadResult: LoadResult) {
        
        do {
            // Pre-validate the model
            try LlamaKitBridge.validateModelFile(path: profile.sourcePath)
            
            let (instance, stream) = await Kuzco.shared.instance(
                for: profile,
                settings: settings,
                predictionConfig: predictionConfig,
                formatter: formatter,
                customStopSequences: customStopSequences
            )
            
            // Collect the load results
            var finalUpdate: LoadUpdate?
            for await update in stream {
                finalUpdate = update
                if update.stage == .ready || update.stage == .failed {
                    break
                }
            }
            
            if let update = finalUpdate {
                if update.stage == .ready {
                    return (instance, .success(instance))
                } else {
                    return (nil, .failure(KuzcoError.modelInitializationFailed(details: update.detail ?? "")))
                }
            } else {
                return (nil, .failure(KuzcoError.unknown(details: "No load update received")))
            }
            
        } catch let error as KuzcoError {
            return (nil, .failure(error))
        } catch {
            return (nil, .failure(KuzcoError.unknown(details: error.localizedDescription)))
        }
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
