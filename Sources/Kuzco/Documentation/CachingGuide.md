# Model Caching in Kuzco

Kuzco provides automatic persistent caching of loaded models across app sessions, dramatically reducing startup times for frequently used models.

## How It Works

- **Automatic**: Models are automatically cached when released and restored when needed
- **Transparent**: No changes needed to existing code - caching works behind the scenes
- **Persistent**: Cached models survive app restarts and device reboots
- **Smart**: Cache invalidation when original model files are modified

## Configuration

### Basic Setup

```swift
import Kuzco

// Configure caching when your app starts
await Kuzco.shared.configureCaching(with: CacheSettings(
    maxCachedModels: 2,           // Cache up to 2 models
    enablePersistentCache: true   // Enable persistent caching
))
```

### Cache Settings

```swift
let cacheSettings = CacheSettings(
    maxCachedModels: 1,           // Number of models to keep cached (default: 1)
    enablePersistentCache: true,  // Enable/disable caching (default: true)
    cacheDirectory: nil           // Custom cache directory (default: app support)
)

await Kuzco.shared.configureCaching(with: cacheSettings)
```

## Usage Examples

### Standard Usage (Automatic Caching)

```swift
// First usage - model loads from disk
let profile = ModelProfile(
    sourcePath: "/path/to/llama3-8b.gguf",
    architecture: .llama3
)

let stream = try await kuzco.predict(
    dialogue: [Turn(role: .user, text: "Hello!")],
    with: profile
)
// Model is automatically cached when done

// Later usage - model loads from cache (much faster)
let stream2 = try await kuzco.predict(
    dialogue: [Turn(role: .user, text: "How are you?")],
    with: profile
)
// ✅ Loads instantly from cache!
```

### Checking Cache Status

```swift
let profile = ModelProfile(
    sourcePath: "/path/to/model.gguf",
    architecture: .llama3
)

// Check if model is cached
let isCached = await kuzco.isModelCached(for: profile)
if isCached {
    print("Model will load quickly from cache")
} else {
    print("Model will load from disk (slower)")
}
```

### Cache Management

```swift
// Clear all cached models
await kuzco.clearModelCache()

// Remove specific model from cache
await kuzco.removeCachedModel(for: profile)

// Disable caching entirely
await kuzco.configureCaching(with: CacheSettings(enablePersistentCache: false))
```

## Storage Requirements

- **Cached models**: ~1-4GB per model (depends on model size and quantization)
- **Location**: `~/Library/Application Support/KuzcoModelCache/` (macOS) or app support directory (iOS)
- **LRU eviction**: Oldest models are automatically removed when cache limit is reached

## Best Practices

### Model Selection Strategy

```swift
// Cache your most frequently used model
let primaryModel = ModelProfile(
    sourcePath: "/path/to/primary-model.gguf",
    architecture: .llama3
)

// Use a smaller quantized model for secondary tasks
let secondaryModel = ModelProfile(
    sourcePath: "/path/to/small-model-q4.gguf", 
    architecture: .llama3
)

// Configure to cache both
await kuzco.configureCaching(with: CacheSettings(maxCachedModels: 2))
```

### Cache Warming

```swift
// Pre-warm cache with your main model during app startup
let profile = ModelProfile(sourcePath: "/path/to/model.gguf", architecture: .llama3)

// This will cache the model for future use
let (instance, loadStream) = await kuzco.instance(for: profile)
for await progress in loadStream {
    print("Pre-warming: \(progress.stage)")
    if progress.stage == .ready { break }
}
// Model is now cached and ready for instant future use
```

### Memory Considerations

```swift
#if os(iOS)
// iOS: Use more conservative settings
await kuzco.configureCaching(with: CacheSettings(
    maxCachedModels: 1,                    // Cache only 1 model on iOS
    enablePersistentCache: true
))
#else
// macOS: Can cache more models
await kuzco.configureCaching(with: CacheSettings(
    maxCachedModels: 3,                    // Cache up to 3 models on macOS
    enablePersistentCache: true
))
#endif
```

## Performance Impact

### Before Caching
```
First model load: ~10-30 seconds (loading from disk)
Subsequent loads: ~10-30 seconds (reloading from disk)
```

### After Caching  
```
First model load: ~10-30 seconds (loading + caching)
Subsequent loads: ~1-3 seconds (loading from cache)
```

### Typical Speedup
- **iOS**: 5-10x faster model loading
- **macOS**: 3-8x faster model loading  
- **Cache hit rate**: 90%+ for typical usage patterns

## Troubleshooting

### Cache Not Working

```swift
// Check if caching is enabled
let isCached = await kuzco.isModelCached(for: profile)
print("Model cached: \(isCached)")

// Verify cache settings
await kuzco.configureCaching(with: CacheSettings(
    maxCachedModels: 1,
    enablePersistentCache: true  // Make sure this is true
))
```

### High Storage Usage

```swift
// Reduce cache size
await kuzco.configureCaching(with: CacheSettings(maxCachedModels: 1))

// Or clear cache periodically
await kuzco.clearModelCache()
```

### Cache Invalidation

The cache automatically invalidates when:
- Original model file is modified
- Model file is moved or deleted
- Cache corruption is detected

You can manually clear invalid cache entries:

```swift
await kuzco.clearModelCache()
``` 