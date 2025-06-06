//
//  ModelCache.swift
//  Kuzco
//
//  Created by Kuzco on 2025-01-01.
//

import Foundation

/// Settings for controlling model caching behavior
public struct CacheSettings: Codable {
    public var maxCachedModels: Int
    public var enablePersistentCache: Bool
    public var cacheDirectory: String?
    
    public init(
        maxCachedModels: Int = 1,
        enablePersistentCache: Bool = true,
        cacheDirectory: String? = nil
    ) {
        self.maxCachedModels = maxCachedModels
        self.enablePersistentCache = enablePersistentCache
        self.cacheDirectory = cacheDirectory
    }
    
    public static var `default`: CacheSettings {
        CacheSettings()
    }
}

/// Metadata about a cached model instance
struct CachedModelMetadata: Codable {
    let profileId: String
    let sourcePath: String
    let architecture: ModelArchitecture
    let settings: InstanceSettings
    let lastAccessed: Date
    let cacheFilePath: String
    
    init(from profile: ModelProfile, settings: InstanceSettings, cacheFilePath: String) {
        self.profileId = profile.id
        self.sourcePath = profile.sourcePath
        self.architecture = profile.architecture
        self.settings = settings
        self.lastAccessed = Date()
        self.cacheFilePath = cacheFilePath
    }
}

/// Manages persistent caching of model instances across app sessions
@globalActor
public actor ModelCache {
    public static let shared = ModelCache()
    
    private var cacheSettings: CacheSettings
    private var cachedMetadata: [String: CachedModelMetadata] = [:]
    private let cacheDirectory: URL
    private let metadataFileName = "kuzco_cache_metadata.json"
    
    private init() {
        self.cacheSettings = .default
        
        // Create cache directory in app support directory
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.cacheDirectory = appSupport.appendingPathComponent("KuzcoModelCache")
        
        // Create directory if it doesn't exist
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        
        // Load existing cache metadata
        loadCacheMetadata()
    }
    
    public func configure(with settings: CacheSettings) {
        self.cacheSettings = settings
        if !settings.enablePersistentCache {
            clearAllCache()
        }
    }
    
    /// Check if a model instance is cached and can be restored
    public func hasCachedInstance(for profile: ModelProfile, with settings: InstanceSettings) -> Bool {
        guard cacheSettings.enablePersistentCache else { return false }
        
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        if let metadata = cachedMetadata[cacheKey] {
            // Verify the cache file still exists and original model hasn't changed
            let cacheFileExists = FileManager.default.fileExists(atPath: metadata.cacheFilePath)
            let originalModelExists = FileManager.default.fileExists(atPath: metadata.sourcePath)
            
            if cacheFileExists && originalModelExists {
                // Check if original model has been modified since caching
                if let originalModifiedDate = try? FileManager.default.attributesOfItem(atPath: metadata.sourcePath)[.modificationDate] as? Date,
                   originalModifiedDate <= metadata.lastAccessed {
                    return true
                }
            }
            
            // Cache is invalid, remove metadata
            cachedMetadata.removeValue(forKey: cacheKey)
            saveCacheMetadata()
        }
        
        return false
    }
    
    /// Cache a model instance for future use
    public func cacheInstance(_ instance: LlamaInstance) async throws {
        guard cacheSettings.enablePersistentCache else { return }
        
        let profile = instance.profile
        let settings = await instance.getCurrentSettings()
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        
        // Create cache file path
        let cacheFileName = "\(cacheKey).kuzco_cache"
        let cacheFilePath = cacheDirectory.appendingPathComponent(cacheFileName)
        
        // Serialize the model state (this would require changes to LlamaInstance)
        try await serializeModelInstance(instance, to: cacheFilePath)
        
        // Update metadata
        let metadata = CachedModelMetadata(
            from: profile,
            settings: settings,
            cacheFilePath: cacheFilePath.path
        )
        
        cachedMetadata[cacheKey] = metadata
        
        // Enforce cache size limit
        await enforceCacheLimit()
        
        // Save metadata
        saveCacheMetadata()
        
        print("🦙 Kuzco - Cached model instance for \(profile.id) 🦙")
    }
    
    /// Restore a cached model instance
    public func restoreCachedInstance(for profile: ModelProfile, with settings: InstanceSettings) async throws -> LlamaInstance? {
        guard cacheSettings.enablePersistentCache else { return nil }
        
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        guard let metadata = cachedMetadata[cacheKey] else { return nil }
        
        let cacheFilePath = URL(fileURLWithPath: metadata.cacheFilePath)
        
        // Deserialize the model instance
        let instance = try await deserializeModelInstance(from: cacheFilePath, profile: profile, settings: settings)
        
        // Update last accessed time
        var updatedMetadata = metadata
        updatedMetadata = CachedModelMetadata(from: profile, settings: settings, cacheFilePath: metadata.cacheFilePath)
        cachedMetadata[cacheKey] = updatedMetadata
        saveCacheMetadata()
        
        print("🦙 Kuzco - Restored cached model instance for \(profile.id) 🦙")
        return instance
    }
    
    /// Clear all cached models
    public func clearAllCache() {
        // Remove all cache files
        for metadata in cachedMetadata.values {
            try? FileManager.default.removeItem(atPath: metadata.cacheFilePath)
        }
        
        // Clear metadata
        cachedMetadata.removeAll()
        saveCacheMetadata()
        
        print("🦙 Kuzco - Cleared all cached models 🦙")
    }
    
    /// Remove a specific cached model
    public func removeCachedInstance(for profile: ModelProfile, with settings: InstanceSettings) {
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        
        if let metadata = cachedMetadata.removeValue(forKey: cacheKey) {
            try? FileManager.default.removeItem(atPath: metadata.cacheFilePath)
            saveCacheMetadata()
            print("🦙 Kuzco - Removed cached model for \(profile.id) 🦙")
        }
    }
    
    // MARK: - Private Methods
    
    private func generateCacheKey(profile: ModelProfile, settings: InstanceSettings) -> String {
        // Create a unique key based on model path, architecture, and critical settings
        let criticalSettings = "\(settings.contextLength)_\(settings.offloadedGpuLayers)_\(settings.useFlashAttention)"
        return "\(profile.id)_\(profile.architecture.rawValue)_\(criticalSettings)".replacingOccurrences(of: "/", with: "_")
    }
    
    private func enforceCacheLimit() async {
        guard cachedMetadata.count > cacheSettings.maxCachedModels else { return }
        
        // Sort by last accessed date and remove oldest
        let sortedMetadata = cachedMetadata.sorted { $0.value.lastAccessed < $1.value.lastAccessed }
        let itemsToRemove = sortedMetadata.prefix(cachedMetadata.count - cacheSettings.maxCachedModels)
        
        for (cacheKey, metadata) in itemsToRemove {
            try? FileManager.default.removeItem(atPath: metadata.cacheFilePath)
            cachedMetadata.removeValue(forKey: cacheKey)
            print("🦙 Kuzco - Evicted cached model \(metadata.profileId) (LRU) 🦙")
        }
    }
    
    private func loadCacheMetadata() {
        let metadataURL = cacheDirectory.appendingPathComponent(metadataFileName)
        
        guard let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode([String: CachedModelMetadata].self, from: data) else {
            return
        }
        
        self.cachedMetadata = metadata
        print("🦙 Kuzco - Loaded cache metadata for \(metadata.count) models 🦙")
    }
    
    private func saveCacheMetadata() {
        let metadataURL = cacheDirectory.appendingPathComponent(metadataFileName)
        
        guard let data = try? JSONEncoder().encode(cachedMetadata) else { return }
        try? data.write(to: metadataURL)
    }
    
    // These methods would need to be implemented based on llama.cpp's state serialization capabilities
    private func serializeModelInstance(_ instance: LlamaInstance, to url: URL) async throws {
        // This is a placeholder - actual implementation would depend on llama.cpp's
        // ability to serialize/deserialize model state
        throw KuzcoError.unknown(details: "Model serialization not yet implemented")
    }
    
    private func deserializeModelInstance(from url: URL, profile: ModelProfile, settings: InstanceSettings) async throws -> LlamaInstance {
        // This is a placeholder - actual implementation would depend on llama.cpp's
        // ability to serialize/deserialize model state
        throw KuzcoError.unknown(details: "Model deserialization not yet implemented")
    }
}

// Extension to LlamaInstance to support caching
extension LlamaInstance {
    func getCurrentSettings() async -> InstanceSettings {
        return self.settings
    }
} 