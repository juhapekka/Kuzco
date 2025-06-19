# Kuzco 🦙

[![Swift Package Manager](https://img.shields.io/badge/Swift%20Package%20Manager-compatible-brightgreen.svg)](https://swift.org/package-manager/)
[![Platform](https://img.shields.io/badge/platform-iOS%2015%2B%20|%20macOS%2012%2B%20|%20Mac%20Catalyst%2015%2B-blue.svg)](https://swift.org)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Kuzco** is a powerful, easy-to-use Swift package that brings local Large Language Model (LLM) inference to iOS and macOS apps. Built on top of the battle-tested `llama.cpp`, Kuzco enables you to run AI models directly on-device with zero network dependency, ensuring privacy, speed, and reliability.

> 🔒 **Privacy First**: All inference happens locally on-device  
> ⚡ **High Performance**: Optimized for Apple Silicon and Intel Macs  
> 🎯 **Production Ready**: Built for real-world iOS and macOS applications

## ✨ Key Features

### 🚀 **Core Capabilities**
- **Local LLM Execution**: Run powerful language models entirely on-device using `llama.cpp`
- **Multiple Model Architectures**: Support for LLaMA, Mistral, Phi, Gemma, Qwen, and more
- **Async/Await Native**: Modern Swift concurrency with streaming responses
- **Cross-Platform**: Works seamlessly on iOS, macOS, and Mac Catalyst

### ⚙️ **Advanced Configuration**
- **Flexible Model Settings**: Fine-tune context length, batch size, GPU layers, and CPU threads
- **Customizable Sampling**: Control temperature, top-K, top-P, repetition penalties, and more
- **Smart Resource Management**: Efficient instance caching and automatic context handling
- **Automatic Architecture Detection**: Auto-detect model architectures from filenames

### 🎨 **Developer Experience**
- **Simple API**: Get started with just a few lines of code
- **Comprehensive Error Handling**: Detailed error messages and recovery suggestions
- **Memory Efficient**: Optimized for mobile device constraints
- **Thread Safe**: Concurrent prediction support
- **Fallback Support**: Automatic fallback to compatible architectures

## 📋 Requirements

- **iOS**: 15.0+
- **macOS**: 12.0+
- **Mac Catalyst**: 15.0+
- **Swift**: 5.9+
- **Xcode**: 15.0+

## 📦 Installation

### Swift Package Manager

Add Kuzco to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(path: "./path/to/Kuzco") // For local development
    // or for remote repository:
    // .package(url: "https://github.com/yourusername/Kuzco.git", from: "1.0.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["Kuzco"]
)
```

## 🚀 Quick Start

### Basic Usage

```swift
import Kuzco

class ChatService {
    private let kuzco = Kuzco.shared
    
    func generateResponse(to userMessage: String) async throws {
        // 1. Create a model profile with automatic architecture detection
        let profile = ModelProfile(
            id: "my-model",
            sourcePath: "/path/to/your/model.gguf"
            // architecture is auto-detected from filename
        )
        
        // 2. Get model instance with safe loading
        let (instance, loadStream) = await kuzco.instance(for: profile)
        
        // 3. Monitor loading progress
        for await progress in loadStream {
            print("Loading: \(progress.stage)")
            if progress.stage == .ready {
                break
            } else if progress.stage == .failed {
                print("Failed to load: \(progress.detail ?? "Unknown error")")
                return
            }
        }
        
        // 4. Create conversation turns
        let turns = [Turn(role: .user, text: userMessage)]
        
        // 5. Generate response with streaming
        let predictionStream = try await instance.predict(
            turns: turns,
            systemPrompt: "You are a helpful assistant."
        )
        
        // 6. Process the streaming response
        for try await (content, isComplete, _) in predictionStream {
            print(content, terminator: "")
            if isComplete { break }
        }
        print() // New line after completion
    }
}
```

### Safe Model Loading

```swift
// Use the safe loading method for better error handling
let (instance, result) = await Kuzco.loadModelSafely(
    profile: profile,
    settings: .standard
)

switch result {
case .success(let loadedInstance):
    print("✅ Model loaded successfully!")
    // Use the instance for predictions
    
case .failure(let error):
    print("❌ Model loading failed: \(error.localizedDescription)")
    if let suggestion = error.recoverySuggestion {
        print("💡 Suggestion: \(suggestion)")
    }
}
```

### Advanced Configuration

```swift
// Custom instance settings for better performance
let customSettings = InstanceSettings(
    contextLength: 4096,
    processingBatchSize: 512,
    gpuOffloadLayers: 35,
    cpuThreadCount: 8
)

// Fine-tuned prediction config
let customConfig = PredictionConfig(
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repeatPenalty: 1.1,
    maxNewTokens: 1024
)

// Use custom configurations
let (instance, loadStream) = await kuzco.instance(
    for: profile,
    settings: customSettings,
    predictionConfig: customConfig
)
```

## 🧠 Supported Model Architectures

Kuzco supports multiple popular LLM architectures with automatic detection and optimized prompt formatting:

| Architecture | Models | Auto-Detection Keywords | Prompt Format |
|-------------|---------|------------------------|---------------|
| **LLaMA 3** | Llama 3, Llama 3.1, Llama 3.2 | `llama-3`, `llama3` | LLaMA 3 format |
| **LLaMA General** | Llama 2, Code Llama | `llama`, `codellama` | Standard LLaMA format |
| **Qwen** | Qwen2, Qwen3 | `qwen2`, `qwen3` | ChatML format |
| **Mistral** | Mistral 7B, Mixtral 8x7B | `mistral`, `mixtral` | Mistral chat format |
| **Phi** | Phi-3, Phi-3.5 | `phi` | Microsoft Phi format |
| **Gemma** | Gemma 2B, Gemma 7B | `gemma` | Google Gemma format |
| **DeepSeek** | DeepSeek models | `deepseek` | DeepSeek format |
| **Command-R** | Command-R models | `command-r`, `commandr` | Command-R format |
| **Yi** | Yi models | `yi-` | ChatML format |
| **OpenChat** | OpenChat models | `openchat` | ChatML format |

### Manual Architecture Specification

```swift
// Explicitly specify architecture when auto-detection isn't sufficient
let profile = ModelProfile(
    id: "my-model",
    sourcePath: "/path/to/model.gguf",
    architecture: .qwen3
)
```

### Fallback Support

```swift
// Use the safer initialization for better compatibility
let profile = ModelProfile.createWithFallback(
    id: "my-model",
    sourcePath: "/path/to/qwen3-model.gguf"
    // Automatically falls back to qwen2 if qwen3 is unsupported
)
```

## ⚙️ Configuration Reference

### InstanceSettings

Controls how the model is loaded and executed:

```swift
let settings = InstanceSettings(
    contextLength: 4096,           // Context window size (tokens)
    processingBatchSize: 512,      // Batch size for processing
    gpuOffloadLayers: 35,          // Layers to offload to GPU (Metal)
    cpuThreadCount: 8              // CPU threads to use
)
```

### PredictionConfig

Fine-tune the text generation behavior:

```swift
let config = PredictionConfig(
    temperature: 0.7,              // Randomness (0.0 = deterministic, 1.0+ = creative)
    topK: 40,                     // Top-K sampling
    topP: 0.9,                    // Nucleus sampling
    repeatPenalty: 1.1,           // Repetition penalty
    maxNewTokens: 1024,           // Maximum tokens to generate
    stopSequences: ["</s>"]       // Stop generation at these sequences
)
```

## 🔧 Troubleshooting

### Common Issues

**Q: My model isn't loading / crashes on load**
- Ensure your `.gguf` model file is compatible with llama.cpp
- Check that the file path is correct and accessible
- Verify you have enough available RAM for the model
- Use `profile.validateModelFile()` to check file integrity

**Q: "unknown model architecture" Error**
- Let Kuzco auto-detect the architecture by not specifying it explicitly
- Use `ModelProfile.createWithFallback()` for better compatibility
- Ensure your model filename contains recognizable architecture keywords

**Q: Inference is slow**
- Increase `gpuOffloadLayers` for Apple Silicon devices
- Reduce `contextLength` if you don't need large contexts
- Try `InstanceSettings.standard` or customize settings for your hardware

**Q: Getting memory warnings on iOS**
- Use smaller quantized models (Q4_0, Q4_1)
- Reduce `contextLength` and `processingBatchSize`
- Monitor memory usage and implement proper cleanup

### Performance Tips

1. **Model Selection**: Use appropriately sized models for your target devices
2. **Quantization**: Q4_0 and Q4_1 models offer good quality/size balance
3. **Context Management**: Only include necessary conversation history
4. **Caching**: Leverage Kuzco's automatic instance caching

## 📱 Example Implementation

This package includes `ChatPage.swift` as an example of how to integrate Kuzco into a real SwiftUI application, demonstrating:

- Safe model loading with error handling
- Streaming response generation
- Conversation continuation
- Memory management
- User-friendly error messages

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone the repository
git clone /path/to/Kuzco
cd Kuzco

# Build and test
swift build
swift test
```

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - The foundational C++ library that makes this possible
- **[Georgi Gerganov](https://github.com/ggerganov)** - Creator of llama.cpp
- **Open Source Community** - For making efficient on-device AI a reality

---

<div align="center">
  <strong>Built with ❤️ for the Swift community</strong><br>
  <sub>Made by <a href="https://github.com/jaredcassoutt">Jared Cassoutt</a></sub>
</div>
