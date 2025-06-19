# Kuzco 🦙

[![Swift Package Manager](https://img.shields.io/badge/Swift%20Package%20Manager-compatible-brightgreen.svg)](https://swift.org/package-manager/)
[![Platform](https://img.shields.io/badge/platform-iOS%2015%2B%20|%20macOS%2012%2B%20|%20Mac%20Catalyst%2015%2B-blue.svg)](https://swift.org)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Kuzco** is a powerful, easy-to-use Swift package that brings local Large Language Model (LLM) inference to iOS and macOS apps. Built on top of the battle-tested `llama.cpp`, Kuzco enables you to run AI models directly on-device with zero network dependency, ensuring privacy, speed, and reliability.

> 🔒 **Privacy First**: All inference happens locally on-device  
> ⚡ **High Performance**: Optimized for Apple Silicon and Intel Macs  
> 🎯 **Production Ready**: Built for real-world iOS and macOS applications

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Your iOS/macOS App                         │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Kuzco.shared                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   predict()     │  │   instance()    │  │ Cache Management    │  │
│  │   ↓             │  │   ↓             │  │ ↓                   │  │
│  │ • Dialogue      │  │ • ModelProfile  │  │ • clearCache()      │  │
│  │ • Streaming     │  │ • Settings      │  │ • isModelCached()   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────┬───────────────────┬───────────────────────────┘
                      │                   │
              ┌───────▼──────┐   ┌────────▼────────┐
              │ Memory Cache │   │   ModelCache    │
              │  (Runtime)   │   │  (Persistent)   │
              │              │   │                 │
              │              │   │                 │ 
              │ Active       │   │ • Disk Storage  │
              │ Instances    │   │ • LRU Eviction  │
              │              │   │ • Validation    │
              └──────┬───────┘   └────────┬────────┘
                     │                    │
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          LlamaInstance                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Model Loading   │  │ Context Mgmt    │  │ Token Generation    │  │
│  │ • Startup       │  │ • KV Cache      │  │ • Streaming         │  │
│  │ • Pre-warming   │  │ • Batching      │  │ • Stop Sequences    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LlamaKitBridge                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Model Ops       │  │ Context Ops     │  │ Token Ops           │  │
│  │ • Load/Free     │  │ • Create/Free   │  │ • Tokenize          │  │
│  │ • Memory Map    │  │ • Batch Proc    │  │ • Detokenize        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           llama.cpp                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ GGUF Parser     │  │ Inference Eng   │  │ Hardware Accel      │  │
│  │ • Model Format  │  │ • Transformer   │  │ • Metal (GPU)       │  │
│  │ • Quantization  │  │ • Attention     │  │ • CPU Vectorization │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            File System                              │
│                                                                     │
│  Your App Bundle/Documents     App Support (Cache)                  │
│  ├── model1.gguf (4.2GB)       ├── KuzcoModelCache/                 │
│  ├── model2.gguf (7.1GB)       │   ├── cached_model_1.kuzco_cache   │
│  └── model3.gguf (2.8GB)       │   ├── cached_model_2.kuzco_cache   │
│                                │   └── kuzco_cache_metadata.json    │
└─────────────────────────────────────────────────────────────────────┘

📊 Data Flow:
┌─────┐    ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────┐
│User │───▶│ Kuzco   │───▶│ Instance │───▶│ Bridge      │───▶│llama.cpp│
│Call │    │ Manager │    │ (Cached) │    │ (C++ Wrap)  │    │ Engine  │
└─────┘    └─────────┘    └──────────┘    └─────────────┘    └─────────┘
              │ ▲
              ▼ │ 🔄 Cache Hit: ~1-3 seconds
        ┌─────────────┐ 
        │ ModelCache  │ 
        │ (Persistent)│ 
        └─────────────┘ 
              │
              ▼ 💾 Cache Miss: Load from .gguf (~10-30 seconds)
        ┌─────────────┐
        │ Disk Storage│
        │(.gguf files)│
        └─────────────┘
```

## ✨ Key Features

### 🚀 **Core Capabilities**
- **Local LLM Execution**: Run powerful language models entirely on-device using `llama.cpp`
- **Multiple Model Architectures**: Support for LLaMA, Mistral, Phi, Gemma, and OpenChat models
- **Async/Await Native**: Modern Swift concurrency with streaming responses
- **Cross-Platform**: Works seamlessly on iOS, macOS, and Mac Catalyst

### ⚙️ **Advanced Configuration**
- **Flexible Model Settings**: Fine-tune context length, batch size, GPU layers, and CPU threads
- **Customizable Sampling**: Control temperature, top-K, top-P, repetition penalties, and Mirostat
- **Smart Resource Management**: Efficient instance caching and automatic context handling
- **Pre-warming Engine**: Optional model preloading for faster inference

### 🎨 **Developer Experience**
- **Simple API**: Get started with just a few lines of code
- **Comprehensive Error Handling**: Detailed error messages and recovery suggestions
- **Memory Efficient**: Optimized for mobile device constraints
- **Thread Safe**: Concurrent prediction support

## 📋 Requirements

- **iOS**: 15.0+
- **macOS**: 12.0+
- **Mac Catalyst**: 15.0+
- **Swift**: 5.9+
- **Xcode**: 15.0+

## 📦 Installation

### Swift Package Manager (Recommended)

#### Via Xcode
1. Open your project in Xcode
2. Go to **File** → **Add Package Dependencies**
3. Enter the repository URL:
   ```
   https://github.com/jcassoutt/Kuzco.git
   ```
4. Select **Up to Next Major Version** and click **Add Package**
5. Add `Kuzco` to your target

#### Via Package.swift
Add Kuzco to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/jcassoutt/Kuzco.git", from: "1.0.0")
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
        // 1. Create a model profile
        let profile = ModelProfile(
            sourcePath: "/path/to/your/model.gguf",
            architecture: .llama3
        )
        
        // 2. Prepare the conversation
        let dialogue: [Turn] = [
            Turn(role: .user, text: userMessage)
        ]
        
        // 3. Generate response with streaming
        let stream = try await kuzco.predict(
            dialogue: dialogue,
            with: profile,
            instanceSettings: .performanceFocused,
            predictionConfig: .creative
        )
        
        // 4. Process the streaming response
        for try await token in stream {
            print(token, terminator: "")
        }
        print() // New line after completion
    }
}
```

### Advanced Configuration

```swift
// Custom instance settings for better performance
let customSettings = InstanceSettings(
    contextLength: 4096,
    batchSize: 512,
    gpuOffloadLayers: 35,
    cpuThreads: 8,
    useFlashAttention: true
)

// Fine-tuned prediction config
let customConfig = PredictionConfig(
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repeatPenalty: 1.1,
    maxTokens: 1024
)

// Use custom configurations
let stream = try await kuzco.predict(
    dialogue: dialogue,
    with: profile,
    instanceSettings: customSettings,
    predictionConfig: customConfig
)
```

### Multi-turn Conversation

```swift
var conversation: [Turn] = []

// Add user message
conversation.append(Turn(role: .user, text: "What is machine learning?"))

// Generate and collect AI response
var aiResponse = ""
let stream = try await kuzco.predict(
    dialogue: conversation,
    with: profile
)

for try await token in stream {
    aiResponse += token
}

// Add AI response to conversation history
conversation.append(Turn(role: .assistant, text: aiResponse))

// Continue the conversation...
conversation.append(Turn(role: .user, text: "Can you give me an example?"))
```

## 🧠 Supported Model Architectures

Kuzco supports multiple popular LLM architectures with optimized prompt formatting:

| Architecture | Models | Prompt Format |
|-------------|---------|---------------|
| **LLaMA 3** | Llama 3, Llama 3.1, Llama 3.2 | Optimized Llama 3 format |
| **LLaMA General** | Llama 2, Code Llama | Standard Llama format |
| **Mistral Instruct** | Mistral 7B, Mixtral 8x7B | Mistral chat format |
| **Phi** | Phi-3, Phi-3.5 | Microsoft Phi format |
| **Gemma** | Gemma 2B, Gemma 7B | Google Gemma format |
| **OpenChat** | OpenChat models | OpenChat conversation format |

### Adding Your Model

```swift
// For a LLaMA 3 model
let profile = ModelProfile(
    sourcePath: "/path/to/llama3-8b.gguf",
    architecture: .llama3
)

// For a Mistral model
let profile = ModelProfile(
    sourcePath: "/path/to/mistral-7b-instruct.gguf",
    architecture: .mistralInstruct
)
```

## ⚙️ Configuration Reference

### InstanceSettings

Controls how the model is loaded and executed:

```swift
let settings = InstanceSettings(
    contextLength: 4096,        // Context window size (tokens)
    batchSize: 512,            // Batch size for processing
    gpuOffloadLayers: 35,      // Layers to offload to GPU (Metal)
    cpuThreads: 8,             // CPU threads to use
    useFlashAttention: true,   // Enable flash attention optimization
    enableMemoryMapping: true  // Use memory mapping for efficiency
)
```

### PredictionConfig

Fine-tune the text generation behavior:

```swift
let config = PredictionConfig(
    temperature: 0.7,          // Randomness (0.0 = deterministic, 1.0+ = creative)
    topK: 40,                 // Top-K sampling
    topP: 0.9,                // Nucleus sampling
    repeatPenalty: 1.1,       // Repetition penalty
    maxTokens: 1024,          // Maximum tokens to generate
    stopSequences: ["</s>"],  // Stop generation at these sequences
    enableMirostat: false,    // Enable Mirostat sampling
    mirostatTau: 5.0,        // Mirostat target entropy
    mirostatEta: 0.1         // Mirostat learning rate
)
```

## 📱 Platform Considerations

### iOS Optimization
- **Memory Management**: Models are automatically unloaded when memory pressure is detected
- **Background Handling**: Inference pauses when app enters background
- **Battery Optimization**: Smart CPU/GPU usage based on device capabilities

### macOS Features
- **Apple Silicon**: Full Metal GPU acceleration on M1/M2/M3 Macs
- **Intel Macs**: Optimized CPU inference with vectorization
- **Memory Mapping**: Efficient large model loading

## 🔧 Troubleshooting

### Common Issues

**Q: My model isn't loading / crashes on load**
- Ensure your `.gguf` model file is compatible with llama.cpp
- Check that the file path is correct and accessible
- Verify you have enough available RAM for the model

**Q: Inference is slow**
- Increase `gpuOffloadLayers` for Apple Silicon devices
- Reduce `contextLength` if you don't need large contexts
- Try `instanceSettings: .performanceFocused`

**Q: Getting memory warnings on iOS**
- Use smaller quantized models (Q4_0, Q4_1)
- Reduce `contextLength` and `batchSize`
- Consider `instanceSettings: .memoryEfficient`

### Performance Tips

1. **Model Selection**: Use appropriately sized models for your target devices
2. **Quantization**: Q4_0 and Q4_1 models offer good quality/size balance
3. **Context Management**: Only include necessary conversation history
4. **Pre-warming**: Use `kuzco.prewarm()` for faster first inference

### "unknown model architecture: 'qwen3'" Error

This error typically occurs when:
1. The underlying llama.cpp version doesn't support the model architecture
2. The model file is corrupted or incompatible

**Solutions:**
- Ensure you're using a GGUF format model
- Try using the automatic architecture detection by not specifying the architecture explicitly
- Check that the model file is not corrupted
- For Qwen3 models specifically, ensure you have a recent enough llama.cpp version

**New: Automatic Fallback Support**
Kuzco now includes automatic fallback mechanisms for unsupported architectures:

```swift
// Use the safer initialization method for better compatibility
let profile = ModelProfile.createWithFallback(
    id: "my-qwen3-model",
    sourcePath: "/path/to/qwen3-model.gguf"
    // This will automatically use qwen2 formatting if qwen3 is unsupported
)

// Or if loading fails, Kuzco will automatically retry with fallback architectures
let (instance, loadStream) = await Kuzco.shared.instance(for: profile)
for await progress in loadStream {
    print("Loading: \(progress.stage) - \(progress.detail ?? "")")
    if progress.stage == .ready { break }
}
```

The fallback system will:
1. First try the detected architecture (qwen3)
2. If that fails, automatically retry with qwen2
3. If qwen2 fails, fall back to unknown/ChatML formatting
4. Provide clear logging about which fallback is being used

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/jcassoutt/Kuzco.git
cd Kuzco
swift build
swift test
```

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - The foundational C++ library that makes this possible
- **[Georgi Gerganov](https://github.com/ggerganov)** - Creator of llama.cpp
- **Open Source Community** - For making efficient on-device AI a reality

## 📚 Additional Resources

- 📖 **[API Documentation](https://jcassoutt.github.io/Kuzco/documentation/kuzco/)** - Complete API reference
- 🧪 **[Example Projects](https://github.com/jcassoutt/Kuzco/tree/main/Examples)** - Sample implementations
- 💬 **[Discussions](https://github.com/jcassoutt/Kuzco/discussions)** - Community Q&A and feature requests
- 🐛 **[Issues](https://github.com/jcassoutt/Kuzco/issues)** - Bug reports and feature requests

---

<div align="center">
  <strong>Built with ❤️ for the Swift community</strong><br>
  <sub>Made by <a href="https://github.com/jaredcassoutt">Jared Cassoutt</a></sub>
</div>

## Recent Updates

✨ **Enhanced Model Architecture Support**: Kuzco now supports a wide range of model architectures including:
- Qwen2 and Qwen3 models
- CodeLlama variants  
- DeepSeek models
- Command-R models
- Yi models
- Mixtral models
- And more with automatic architecture detection

🔧 **Improved Error Handling**: Better error messages for unsupported model architectures with helpful suggestions.

🎯 **Dynamic Model Loading**: Automatic architecture detection from model file names with fallback support.

## Features

- **Easy Model Loading**: Automatic architecture detection and configuration
- **Comprehensive Architecture Support**: Wide range of model architectures with proper prompt formatting
- **Dynamic Error Handling**: Helpful error messages and suggestions for troubleshooting
- **Caching System**: Intelligent model caching for faster subsequent loads
- **SwiftUI Integration**: Ready for use in SwiftUI applications

## Quick Start

### Loading a Model with Automatic Architecture Detection

```swift
import Kuzco

// Automatic architecture detection from filename
let modelProfile = ModelProfile(
    id: "my-qwen3-model",
    sourcePath: "/path/to/qwen3-8b-instruct.gguf"
    // architecture will be auto-detected as .qwen3
)

let (instance, loadStream) = await Kuzco.shared.instance(for: modelProfile)

// Monitor loading progress
for await progress in loadStream {
    print("Loading: \(progress.stage) - \(progress.detail ?? "")")
    if progress.stage == .ready { break }
}
```

### Explicit Architecture Specification

```swift
// Explicitly specify architecture for better control
let modelProfile = ModelProfile(
    id: "my-model",
    sourcePath: "/path/to/model.gguf",
    architecture: .qwen3
)
```

### Handling Different Model Architectures

```swift
// The library automatically handles prompt formatting for different architectures
let dialogue = [
    Turn(role: .user, text: "Hello, how are you?")
]

let predictionStream = try await Kuzco.shared.predict(
    dialogue: dialogue,
    systemPrompt: "You are a helpful assistant.",
    with: modelProfile
)

for try await token in predictionStream {
    print(token, terminator: "")
}
```

### Model Validation

```swift
// Validate model file before loading
do {
    try modelProfile.validateModelFile()
    print("Model file is valid GGUF format")
} catch {
    print("Model validation failed: \(error.localizedDescription)")
    print("Suggestion: \(modelProfile.getArchitectureSuggestions())")
}
```

## Supported Model Architectures

| Architecture | Models | Prompt Format |
|--------------|--------|---------------|
| `.qwen2`, `.qwen3` | Qwen series | ChatML format |
| `.llama3` | LLaMA 3 | LLaMA 3 format |
| `.codellama` | CodeLlama | LLaMA format |
| `.mistralInstruct`, `.mixtral` | Mistral models | Mistral format |
| `.deepseek` | DeepSeek models | DeepSeek format |
| `.commandR` | Command-R models | Command-R format |
| `.yi` | Yi models | ChatML format |
| `.phiGeneric` | Phi models | Phi format |
| `.gemmaInstruct` | Gemma models | Gemma format |
| `.openChat` | OpenChat models | ChatML format |
| `.llamaGeneral` | General LLaMA | LLaMA format |
| `.unknown` | Auto-fallback | ChatML format |

## Error Handling

If you encounter model loading errors, Kuzco provides helpful error messages:

```swift
do {
    let (instance, loadStream) = await Kuzco.shared.instance(for: modelProfile)
    // ... handle loading
} catch let error as KuzcoError {
    switch error {
    case .unsupportedModelArchitecture(let arch, let suggestion):
        print("Unsupported architecture '\(arch)': \(suggestion)")
    case .modelInitializationFailed(let details):
        print("Model failed to load: \(details)")
    default:
        print("Error: \(error.localizedDescription)")
    }
}
```

## Common Issues and Solutions

### "unknown model architecture: 'qwen3'" Error

This error typically occurs when:
1. The underlying llama.cpp version doesn't support the model architecture
2. The model file is corrupted or incompatible

**Solutions:**
- Ensure you're using a GGUF format model
- Try using the automatic architecture detection by not specifying the architecture explicitly
- Check that the model file is not corrupted
- For Qwen3 models specifically, ensure you have a recent enough llama.cpp version

### Model File Not Loading

1. **Check file path**: Ensure the file path is correct and accessible
2. **Validate format**: Use `modelProfile.validateModelFile()` to check if it's a valid GGUF file
3. **Check architecture**: Let Kuzco auto-detect the architecture or specify it explicitly

### Performance Optimization

- Use model caching to avoid reloading the same models
- Adjust instance settings for your hardware capabilities
- Consider using smaller quantized models for faster inference

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## License

This project is licensed under the Apache 2.0 License.
